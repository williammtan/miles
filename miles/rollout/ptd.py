"""Privileged teacher scoring, separate from outcome rewards and advantages."""

import json
import math
from argparse import Namespace
from typing import Any

import torch

from miles.rollout.ptd_scoring import request_scores, request_scores_async, validate_score_entries
from miles.utils.types import Sample


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
    context = teacher_score_context(args, sample, tokenizer, hint)
    payload = {**context["payload"], "top_logprobs_num": args.ptd_top_k}
    response = await request_scores_async(context["url"], payload, args.ptd_score_timeout)
    rows = extract_score_rows(response, context, "input_top_logprobs")
    for row in rows:
        validate_score_entries(row, context["vocab_size"], count=args.ptd_top_k)
    sample.ptd_teacher_ids = torch.tensor([[int(e[1]) for e in row] for row in rows], dtype=torch.int32)
    sample.ptd_teacher_log_probs = torch.tensor([[float(e[0]) for e in row] for row in rows], dtype=torch.float32)
    sample.ptd_teacher_context = context


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
