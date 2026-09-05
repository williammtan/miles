"""Privileged teacher scoring, separate from outcome rewards and advantages."""

import json
import math
import uuid
from argparse import Namespace
from typing import Any

import torch

from miles.rollout.ptd_scoring import request_scores, validate_score_entries
from miles.utils.types import Sample


def validate_teacher_route(args: Namespace) -> None:
    """The pinned Rust router drops PTD's sparse IDs and cache-isolation salt."""
    if getattr(args, "ptd_coef", 0) > 0 and not args.ptd_teacher_url and not args.use_miles_router:
        raise ValueError(
            "PTD requires --use-miles-router when --ptd-teacher-url is unset: "
            "the SGLang Rust router drops token_ids_logprob_positions and cache_salt. "
            "An explicit teacher URL must reach a patched engine through a proxy that preserves these fields "
            "or connect directly to the engine."
        )


def is_tutor_target(args: Namespace, sample: Sample) -> bool:
    """Infrastructure failures are never evidence that an answer is incorrect."""
    return (
        getattr(args, "ptd_coef", 0) > 0
        and sample.metadata.get("grade_valid") is True
        and sample.get_reward_value(args) == 0
        and sample.response_length > 0
        and not sample.remove_sample
        and sample.status in (Sample.Status.COMPLETED, Sample.Status.TRUNCATED)
    )


def hint_text(hint: str | dict) -> str:
    if isinstance(hint, str):
        text = hint.strip()
    elif isinstance(hint, dict):
        text = json.dumps(hint, ensure_ascii=False, sort_keys=True)
    else:
        raise TypeError("PTD hint callback must return a nonempty str or dict")
    if not text or text == "{}":
        raise ValueError("PTD hint callback returned an empty hint")
    return text


def teacher_score_context(args: Namespace, sample: Sample, tokenizer, hint: str | dict) -> dict[str, Any]:
    """Prepend a teacher-only system message, preserving all original token IDs.

    In particular the multimodal prompt is NOT re-tokenized and the sampled
    continuation, including EOS or a truncated final token, is unchanged.
    """
    instruction = (
        "Use these privileged, answer-free tutoring hints to interpret the image and question. "
        "Continue the assistant response; do not quote the hints.\n" + hint_text(hint)
    )
    # Qwen's template rejects a system-only conversation and injects a default
    # reasoning system message. Render with a sentinel user and retain exactly
    # the system prefix, including its closing special token. Explicitly request
    # IDs: Transformers 5 otherwise returns a BatchEncoding for tokenize=True.
    user = {"role": "user", "content": "__MILES_PTD_PREFIX_BOUNDARY__"}
    with_system = list(tokenizer.apply_chat_template(
        [{"role": "system", "content": instruction}, user], tokenize=True, return_dict=False,
        add_generation_prompt=False,
    ))
    marker = tokenizer.encode("<|im_start|>user\n", add_special_tokens=False)
    starts = [i for i in range(len(with_system)) if with_system[i:i + len(marker)] == marker]
    if not starts or starts[-1] == 0:
        raise ValueError("PTD teacher prefix currently requires the Qwen ChatML template")
    prefix = with_system[:starts[-1]]
    input_ids = list(prefix) + list(sample.tokens)
    prompt_length = len(input_ids) - sample.response_length
    if prompt_length <= 0:
        raise ValueError("PTD teacher scoring needs at least one prefix token")
    media = sample.metadata.get("ptd_media_payload")
    if media is None:
        raise ValueError("PTD requires the original generation media payload, including an explicit empty dict for text")
    payload = {
        "input_ids": input_ids,
        "sampling_params": {"temperature": 0, "max_new_tokens": 0, "skip_special_tokens": False},
        "return_logprob": True,
        "logprob_start_len": prompt_length - 1,
        # Explicit null selects the frozen base, never the current student LoRA.
        "lora_path": None,
        **media,
    }
    url = args.ptd_teacher_url or f"http://{args.sglang_router_ip}:{args.sglang_router_port}/generate"
    return {"payload": payload, "url": url, "response_tokens": list(sample.tokens[-sample.response_length:]),
            "vocab_size": getattr(args, "ptd_vocab_size", None)}


def extract_score_rows(response: dict, context: dict, field: str) -> list:
    """Validate both count and token IDs before accepting positional scores."""
    meta = response["meta_info"]
    expected = context["response_tokens"]
    scored = meta.get("input_token_logprobs", [])
    # logprob_start_len starts one token before response; row 0 is a null placeholder.
    if len(scored) != len(expected) + 1:
        raise ValueError(f"PTD score count mismatch: {len(scored)} != {len(expected) + 1}")
    if any(type(row[1]) is not int for row in scored[1:]) or [row[1] for row in scored[1:]] != expected:
        raise ValueError("PTD teacher response token IDs do not match the unchanged sampled continuation")
    if any(not isinstance(row[0], (int, float)) or isinstance(row[0], bool) or not math.isfinite(row[0]) or row[0] > 0 for row in scored[1:]):
        raise ValueError("PTD teacher returned an invalid response-token log probability")
    rows = meta.get(field)
    if rows is None or len(rows) != len(expected) + 1:
        raise ValueError(f"PTD teacher missing or misaligned {field}")
    return rows[1:]


async def collect_teacher_targets(args: Namespace, sample: Sample, tokenizer) -> None:
    # A reused sample must not retain tutoring targets after its eligibility changes.
    sample.ptd_teacher_ids = None
    sample.ptd_teacher_log_probs = None
    sample.ptd_teacher_context = None
    if not is_tutor_target(args, sample):
        return
    # Keep pure payload/alignment helpers usable without Ray installed.
    from miles.utils.misc import load_function

    hint_fn = load_function(args.ptd_hint_function_path)
    # The callback owns exact-rollout retry caching; no per-question hint reuse here.
    hint = await hint_fn(args, sample)
    if hint is None:
        sample.metadata.setdefault("ptd_hint_status", "needs_review")
        return
    sample.metadata[args.ptd_hint_key] = hint
    sample.ptd_teacher_context = teacher_score_context(args, sample, tokenizer, hint)
    # Preserve the ragged codec fields without caching probabilities from another
    # teacher forward. Training collects the entire union in one joint request.
    sample.ptd_teacher_ids = torch.empty((0, args.ptd_top_k), dtype=torch.int32)
    sample.ptd_teacher_log_probs = torch.empty((0, args.ptd_top_k), dtype=torch.float32)


def score_teacher_joint(
    context: dict, ids_by_position: list[list[int]], top_k: int, timeout: float,
) -> tuple[list[list[int]], list[list[float]]]:
    """Score teacher Top-K and current student Top-K in one teacher forward.

    Reusing teacher Top-K from an earlier request can combine different numerical
    distributions when a hybrid model's prefix cache changes its execution path.
    Every probability returned here comes from the same response. The union is
    padded to 2K IDs with -1; padding log probabilities are negative infinity.
    """
    vocab_size = context["vocab_size"]
    if type(vocab_size) is not int or not 0 < top_k <= vocab_size:
        raise ValueError("PTD joint scoring requires 0 < Top-K <= vocabulary size")
    if len(ids_by_position) != len(context["response_tokens"]):
        raise ValueError("PTD student Top-K rows do not match the response length")
    for ids in ids_by_position:
        if not isinstance(ids, list) or len(ids) != top_k:
            raise ValueError("PTD joint scoring requires one student Top-K row per response token")
        if any(type(token) is not int or not 0 <= token < vocab_size for token in ids) or len(set(ids)) != top_k:
            raise ValueError("PTD student Top-K contains invalid or duplicate token IDs")
    payload = {
        **context["payload"],
        "top_logprobs_num": top_k,
        "token_ids_logprob_positions": [[], *ids_by_position],
        # Isolate hybrid-model prefix state across logical teacher requests.
        # Transport refreshes the salt on retries while preserving semantic input.
        "cache_salt": uuid.uuid4().hex,
    }
    response = request_scores(context["url"], payload, timeout)
    top_rows = extract_score_rows(response, context, "input_top_logprobs")
    student_rows = extract_score_rows(response, context, "input_token_ids_logprobs")
    union_ids, union_log_probs = [], []
    for top, student, requested in zip(top_rows, student_rows, ids_by_position, strict=True):
        teacher_probs = validate_score_entries(top, vocab_size, count=top_k)
        cross_probs = validate_score_entries(student, vocab_size, requested=requested)
        for token in teacher_probs.keys() & cross_probs.keys():
            # Same-forward kernels may differ by fp32 rounding, not cache-state drift.
            if not math.isclose(teacher_probs[token], cross_probs[token], rel_tol=0, abs_tol=2e-4):
                raise ValueError("PTD joint teacher scores disagree on overlapping token IDs")
        merged = {**cross_probs, **teacher_probs}
        validate_score_entries([[lp, token] for token, lp in merged.items()], vocab_size)
        ids = sorted(merged)
        padding = 2 * top_k - len(ids)
        union_ids.append([*ids, *([-1] * padding)])
        union_log_probs.append([*(merged[token] for token in ids), *([-math.inf] * padding)])
    return union_ids, union_log_probs


def score_teacher_missing_ids(context: dict, ids_by_position: list[list[int]], timeout: float) -> list[dict[int, float]]:
    """Sparse exact teacher cross-probabilities at current student Top-K IDs.

    The additive SGLang patch defines position rows relative to logprob_start_len,
    including the initial placeholder. No global union over response positions.
    """
    payload = {**context["payload"], "token_ids_logprob_positions": [[], *ids_by_position]}
    result = request_scores(context["url"], payload, timeout)
    rows = extract_score_rows(result, context, "input_token_ids_logprobs")
    return [validate_score_entries(row, context["vocab_size"], requested=requested)
            for requested, row in zip(ids_by_position, rows, strict=True)]
