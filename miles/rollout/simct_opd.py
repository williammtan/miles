"""SimCT cross-tokenizer on-policy distillation reward path (arXiv 2605.07711).

A second cross-tokenizer OPD method alongside the DPCA path in
:mod:`miles.rollout.cross_tokenizer_opd`. Where DPCA aligns the *realized*
student/teacher token sequences, SimCT ("Simple Cross-Tokenizer OPD") enlarges
the supervision space to a common set of **text units** and applies the standard
reverse-KL OPD loss there:

    U = (V_T ∩ V_S) ∪ A

where ``A`` are "minimal aligned units" -- short text spans both tokenizers can
realize. At each *aligned* student position (a shared token boundary in byte
space) the candidate set is the union of student and teacher top-k next tokens
decoded to text. Each candidate text ``u`` with tokenizer-M realization
``(v_1..v_k)`` is scored by the length-normalized continuation log-likelihood
``s_M(u) = (1/k) Σ_j log p_M(v_j | x<t, v<j)``; softmax over the candidate set
gives ``q_S, q_T`` and the per-position penalty is the reverse-KL
``D(q_S ‖ q_T)``.

Like the DPCA path it is entirely rollout-side: the per-token reverse-KL is
stored on ``sample.opd_reverse_kl`` and consumed unchanged by the *precomputed*
branch of ``apply_opd_kl_to_advantages``. It reuses the teacher chat-template /
raw-prompt machinery from :mod:`miles.rollout.cross_tokenizer_opd`.

Selected via config (no dispatch edits):

    --custom-rm-path                  miles.rollout.simct_opd.reward_func
    --custom-reward-post-process-path miles.rollout.simct_opd.post_process_rewards
    --use-opd --opd-type sglang --opd-ct-method simct
    --opd-teacher-tokenizer <teacher tokenizer path>
    --opd-ct-candidate-k 20 --opd-ct-max-continuation-len 4
    --rm-url http://<teacher>:<port>/generate
"""

import asyncio
import math
from argparse import Namespace
from typing import Any

import torch

from miles.rollout.cross_tokenizer_opd import (
    RAW_PROMPT_METADATA_KEY,
    _student_tokenizer,
    _teacher_prompt_ids,
    _teacher_tokenizer,
)
from miles.rollout.on_policy_distillation import _post_json
from miles.utils.types import Sample

# ---------------------------------------------------------------------------
# Pure helpers (CPU-testable, no HTTP)
# ---------------------------------------------------------------------------


def _byte_boundaries(tokenizer, token_ids: list[int]) -> list[int]:
    """Cumulative UTF-8 byte length of the decoded prefixes.

    Entry ``i`` is the number of UTF-8 bytes in ``decode(token_ids[:i])``; length
    ``len(token_ids) + 1``. Boundary math is in bytes so partial-UTF8 tokens have
    well-defined positions. Decodes each prefix (O(n^2)); fine for a first cut.
    """
    bounds = [0]
    for i in range(1, len(token_ids) + 1):
        text = tokenizer.decode(token_ids[:i], skip_special_tokens=False)
        bounds.append(len(text.encode("utf-8")))
    return bounds


def _align_positions(
    student_resp_ids: list[int],
    teacher_resp_ids: list[int],
    student_tok,
    teacher_tok,
) -> dict[int, int]:
    """Map aligned student response positions to teacher token indices.

    Student position ``t`` predicts the token starting at response byte offset
    ``sb[t]``. It is *aligned* iff the teacher also has a token boundary there;
    the value is the teacher token index ``j_t`` whose prefix ends at that byte.
    """
    sb = _byte_boundaries(student_tok, student_resp_ids)
    tb = _byte_boundaries(teacher_tok, teacher_resp_ids)
    teacher_boundary_to_idx = {b: j for j, b in enumerate(tb)}
    aligned: dict[int, int] = {}
    for t in range(len(student_resp_ids)):
        j = teacher_boundary_to_idx.get(sb[t])
        if j is not None and j < len(teacher_resp_ids):
            aligned[t] = j
    return aligned


def _continuation_text(tokenizer, prefix_ids: list[int], token_id: int) -> str | None:
    """Text contributed by appending ``token_id`` to ``prefix_ids`` (diff of cumulative decodes)."""
    base = tokenizer.decode(prefix_ids, skip_special_tokens=False)
    full = tokenizer.decode(list(prefix_ids) + [token_id], skip_special_tokens=False)
    if not full.startswith(base):
        return None
    suffix = full[len(base) :]
    return suffix or None


def _realize(tokenizer, prefix_ids: list[int], unit_text: str, prefix_text: str) -> list[int] | None:
    """Token ids realizing ``unit_text`` as a continuation of ``prefix_text``.

    Returns ``None`` if the encoding does not extend the prefix cleanly (the
    re-encoding changed the prefix segmentation, so length-normalization is unsafe).
    """
    full_ids = tokenizer.encode(prefix_text + unit_text, add_special_tokens=False)
    if len(full_ids) <= len(prefix_ids) or full_ids[: len(prefix_ids)] != list(prefix_ids):
        return None
    return full_ids[len(prefix_ids) :]


def _softmax(log_scores: list[float]) -> list[float]:
    m = max(log_scores)
    exps = [math.exp(s - m) for s in log_scores]
    z = sum(exps)
    if z == 0.0:
        return [0.0 for _ in exps]
    return [e / z for e in exps]


def _reverse_kl(scores_s: dict[str, float], scores_t: dict[str, float]) -> float:
    """Reverse KL ``D(q_S ‖ q_T)`` over the shared candidate set."""
    keys = [k for k in scores_s if k in scores_t]
    if len(keys) < 2:
        return 0.0
    q_s = _softmax([scores_s[k] for k in keys])
    q_t = _softmax([scores_t[k] for k in keys])
    kl = 0.0
    for ps, pt in zip(q_s, q_t, strict=True):
        if ps > 0.0 and pt > 0.0:
            kl += ps * (math.log(ps) - math.log(pt))
    return kl


def _simct_reverse_kls(positions: list[dict[str, Any]], response_length: int) -> list[float]:
    """Assemble a per-token reverse-KL list of length ``response_length`` (non-aligned positions = 0)."""
    reverse_kls = [0.0] * response_length
    for pos in positions:
        reverse_kls[pos["t"]] = _reverse_kl(pos["scores_s"], pos["scores_t"])
    return reverse_kls


# ---------------------------------------------------------------------------
# SGLang scoring payloads
# ---------------------------------------------------------------------------


def _base_score_payload(input_ids: list[int], logprob_start_len: int, top_k: int) -> dict[str, Any]:
    return {
        "input_ids": input_ids,
        "sampling_params": {"temperature": 0, "max_new_tokens": 0, "skip_special_tokens": False},
        "return_logprob": True,
        "logprob_start_len": logprob_start_len,
        "top_logprobs_num": top_k,
    }


def _continuation_score_payload(full_ids: list[int], num_cont: int) -> dict[str, Any]:
    start = len(full_ids) - num_cont - 1
    return {
        "input_ids": full_ids,
        "sampling_params": {"temperature": 0, "max_new_tokens": 0, "skip_special_tokens": False},
        "return_logprob": True,
        "logprob_start_len": max(0, start),
    }


def _trim_placeholder(values: list[Any]) -> list[Any]:
    # SGLang's first returned input logprob is a placeholder for logprob_start_len.
    return values[1:] if values else values


def _mean_continuation_logprob(response: dict[str, Any], num_cont: int) -> float:
    entries = _trim_placeholder(response["meta_info"]["input_token_logprobs"])
    tail = entries[-num_cont:]
    return sum(float(e[0]) for e in tail) / max(1, len(tail))


def _student_score_url(args: Namespace) -> str:
    return f"http://{args.sglang_router_ip}:{args.sglang_router_port}/generate"


# ---------------------------------------------------------------------------
# Public surface
# ---------------------------------------------------------------------------


async def reward_func(args: Namespace, sample: Sample, **kwargs: Any) -> dict[str, Any]:
    """Compute per-token SimCT reverse-KL for one sample.

    Returns ``{"ctopd_reverse_kl": list[float]}`` of length ``response_length``;
    ``post_process_rewards`` moves it onto ``sample.opd_reverse_kl``.
    """
    response_length = sample.response_length
    if response_length == 0:
        return {"ctopd_reverse_kl": []}

    student_tok = _student_tokenizer(args)
    teacher_tok = _teacher_tokenizer(args)
    candidate_k = int(getattr(args, "opd_ct_candidate_k", 20) or 20)
    max_cont_len = int(getattr(args, "opd_ct_max_continuation_len", 4) or 4)

    student_resp_ids = sample.tokens[-response_length:]
    student_prompt_ids = sample.tokens[: len(sample.tokens) - response_length]
    response_text = sample.response
    teacher_resp_ids = teacher_tok.encode(response_text, add_special_tokens=False)
    if len(teacher_resp_ids) == 0:
        return {"ctopd_reverse_kl": [0.0] * response_length}

    aligned = _align_positions(student_resp_ids, teacher_resp_ids, student_tok, teacher_tok)
    if not aligned:
        return {"ctopd_reverse_kl": [0.0] * response_length}

    # Bound continuation-scoring cost: cap the number of aligned positions scored per
    # sample (evenly subsampled). 0 = score all. Unscored positions are masked to 0.
    max_positions = int(getattr(args, "opd_ct_max_positions", 0) or 0)
    if max_positions > 0 and len(aligned) > max_positions:
        ts = sorted(aligned)
        step = len(ts) / max_positions
        keep = {ts[int(i * step)] for i in range(max_positions)}
        aligned = {t: j for t, j in aligned.items() if t in keep}

    student_top = sample.metadata.get("opd_student_top_logprobs")
    if student_top is None:
        raise ValueError(
            "SimCT cross-tokenizer OPD requires student output_top_logprobs. "
            "Ensure --opd-ct-method=simct is set before rollout generation starts."
        )
    student_top = student_top[-response_length:]

    # Teacher prompt re-templated with the teacher's own chat template (raw prompt).
    raw_prompt = sample.metadata.get(RAW_PROMPT_METADATA_KEY, sample.prompt)
    teacher_prompt_ids = _teacher_prompt_ids(teacher_tok, raw_prompt, sample.metadata.get("tools"))
    teacher_full_ids = list(teacher_prompt_ids) + list(teacher_resp_ids)
    teacher_base = await _post_json(
        args.rm_url,
        _base_score_payload(teacher_full_ids, max(0, len(teacher_prompt_ids) - 1), candidate_k),
    )
    teacher_top = _trim_placeholder(teacher_base["meta_info"]["input_top_logprobs"])

    positions: list[dict[str, Any]] = []
    cross_jobs: list[dict[str, Any]] = []
    for t, j_t in aligned.items():
        student_prefix = student_resp_ids[:t]
        student_prefix_text = student_tok.decode(student_prefix, skip_special_tokens=False)
        teacher_prefix = list(teacher_resp_ids[:j_t])
        teacher_prefix_text = teacher_tok.decode(teacher_prefix, skip_special_tokens=False)

        scores_s: dict[str, float] = {}
        scores_t: dict[str, float] = {}
        for entry in student_top[t][:candidate_k]:
            unit = _continuation_text(student_tok, student_prefix, int(entry[1]))
            if unit:
                scores_s[unit] = float(entry[0])
        for entry in (teacher_top[j_t] or [])[:candidate_k]:
            unit = _continuation_text(teacher_tok, teacher_prefix, int(entry[1]))
            if unit:
                scores_t[unit] = float(entry[0])

        pos_idx = len(positions)
        positions.append({"t": t, "scores_s": scores_s, "scores_t": scores_t})

        for unit in set(scores_s) | set(scores_t):
            if unit not in scores_t:
                cont = _realize(teacher_tok, teacher_prefix, unit, teacher_prefix_text)
                if cont and len(cont) <= max_cont_len:
                    cross_jobs.append(
                        {
                            "side": "teacher",
                            "pos_idx": pos_idx,
                            "unit": unit,
                            "full_ids": list(teacher_prompt_ids) + teacher_prefix + cont,
                            "num_cont": len(cont),
                        }
                    )
            if unit not in scores_s:
                cont = _realize(student_tok, student_prefix, unit, student_prefix_text)
                if cont and len(cont) <= max_cont_len:
                    cross_jobs.append(
                        {
                            "side": "student",
                            "pos_idx": pos_idx,
                            "unit": unit,
                            "full_ids": list(student_prompt_ids) + student_prefix + cont,
                            "num_cont": len(cont),
                        }
                    )

    await _run_cross_scoring(args, cross_jobs, positions)

    return {"ctopd_reverse_kl": _simct_reverse_kls(positions, response_length)}


async def _run_cross_scoring(
    args: Namespace, cross_jobs: list[dict[str, Any]], positions: list[dict[str, Any]]
) -> None:
    if not cross_jobs:
        return
    student_url = _student_score_url(args)

    async def _score(job: dict[str, Any]) -> float:
        url = args.rm_url if job["side"] == "teacher" else student_url
        resp = await _post_json(url, _continuation_score_payload(job["full_ids"], job["num_cont"]))
        return _mean_continuation_logprob(resp, job["num_cont"])

    results = await asyncio.gather(*[_score(job) for job in cross_jobs])
    for job, score in zip(cross_jobs, results, strict=True):
        target = "scores_t" if job["side"] == "teacher" else "scores_s"
        positions[job["pos_idx"]][target][job["unit"]] = score


def post_process_rewards(args: Namespace, samples: list[Sample], **kwargs: Any) -> tuple[list[float], list[float]]:
    """Move the precomputed SimCT reverse-KL onto ``sample.opd_reverse_kl`` (zero scalar reward)."""
    for sample in samples:
        reward = sample.get_reward_value(args)
        reverse_kls = reward["ctopd_reverse_kl"] if isinstance(reward, dict) else []
        sample.opd_reverse_kl = torch.tensor(reverse_kls, dtype=torch.float32)

    scalar_rewards = [0.0] * len(samples)
    return scalar_rewards, scalar_rewards
