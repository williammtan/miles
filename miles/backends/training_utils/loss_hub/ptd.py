"""PTD-PO integration for Megatron TP and CP=1 packed response logits."""

import torch
import torch.distributed as dist

from miles.backends.training_utils.loss_hub.logit_processors import _iter_response_chunks
from miles.backends.training_utils.loss_hub.ptd_math import sparse_vocab_parallel_jsd, support_union, vocab_parallel_topk
from miles.backends.training_utils.parallel import get_parallel_state
from miles.rollout.ptd import score_teacher_missing_ids


def _teacher_union_log_probs(context, ids, teacher_ids, teacher_log_probs, *, timeout, tp):
    """Only TP rank zero queries the teacher; failures propagate to every rank."""
    result = torch.empty(ids.shape, dtype=torch.float32, device=ids.device)
    error = [None]
    if tp.rank == 0:
        try:
            rows = ids.cpu().tolist()
            known = [dict(zip(ii, pp, strict=True)) for ii, pp in zip(
                teacher_ids.tolist(), teacher_log_probs.tolist(), strict=True,
            )]
            missing = [[i for i in row if i >= 0 and i not in cache] for row, cache in zip(rows, known, strict=True)]
            if any(missing):
                cross = score_teacher_missing_ids(context, missing, timeout)
                for cache, extra in zip(known, cross, strict=True):
                    cache.update(extra)
            values = [[cache[i] if i >= 0 else -torch.inf for i in row] for row, cache in zip(rows, known, strict=True)]
            result.copy_(torch.tensor(values, dtype=result.dtype, device=result.device))
        except Exception as exc:
            error[0] = f"{type(exc).__name__}: {exc}"
    if tp.size > 1:
        src = dist.get_global_rank(tp.group, 0)
        dist.broadcast_object_list(error, src=src, group=tp.group)
    if error[0] is not None:
        raise RuntimeError(f"PTD teacher sparse scoring failed: {error[0]}")
    if tp.size > 1:
        dist.broadcast(result, src=src, group=tp.group)
    return result.detach()


def ptd_loss_sum(args, batch, logits):
    """Unnormalized selected-token JSD sum, with gradients through student only."""
    state = get_parallel_state()
    assert state.cp.size == 1 and args.qkv_format == "thd"
    total = logits.reshape(-1)[:1].float().sum() * 0
    targets = batch.get("ptd_teacher_context")
    if targets is None:
        raise ValueError("PTD enabled but teacher-target fields are missing from the training batch")
    tp_group = state.tp.group if state.tp.size > 1 else None
    vocab_start = state.tp.rank * logits.shape[-1]
    chunks = _iter_response_chunks(
        logits, args=args, unconcat_tokens=batch["unconcat_tokens"], total_lengths=batch["total_lengths"],
        response_lengths=batch["response_lengths"], include_response_indices=False,
    )
    for index, (response_logits, response_tokens, _) in enumerate(chunks):
        context = targets[index]
        if context is None or response_tokens.numel() == 0:
            continue
        if response_tokens.tolist() != context["response_tokens"]:
            raise ValueError("PTD packed-training response positions differ from the teacher continuation")
        student_ids = vocab_parallel_topk(
            response_logits, args.ptd_top_k, vocab_start=vocab_start, vocab_size=args.ptd_vocab_size,
            tp_group=tp_group, chunk_size=args.ptd_logits_chunk_size,
        )
        teacher_ids = torch.as_tensor(batch["ptd_teacher_ids"][index])
        teacher_log_probs = torch.as_tensor(batch["ptd_teacher_log_probs"][index])
        ids = support_union(student_ids, teacher_ids.to(device=logits.device, dtype=torch.long))
        teacher_union = _teacher_union_log_probs(
            context, ids, teacher_ids, teacher_log_probs, timeout=args.ptd_score_timeout, tp=state.tp,
        )
        token_losses = sparse_vocab_parallel_jsd(
            response_logits, ids, teacher_union, vocab_start=vocab_start, vocab_size=args.ptd_vocab_size,
            tp_group=tp_group, chunk_size=args.ptd_logits_chunk_size,
        )
        total = total + (token_losses * batch["loss_masks"][index]).sum()
    return total


def add_ptd_loss(args, batch, logits, policy_loss):
    """Cancel the GRPO token denominator to implement equation 11 exactly.

    Megatron sums microbatches and DP gradients, then divides by the global
    GRPO token count. Multiplying the PTD numerator by N_GRPO/N_selected here
    yields lambda * sum_selected(JSD)/N_selected after that existing reduction.
    Counts cover the entire optimizer step, never just this microbatch.
    """
    numerator = ptd_loss_sum(args, batch, logits)
    normalizers = batch["ptd_normalizers"]
    if not normalizers or any(pair != normalizers[0] for pair in normalizers):
        raise ValueError("PTD microbatch mixes optimizer-step normalizers")
    grpo_tokens, selected_tokens = normalizers[0]
    scaled = numerator * (grpo_tokens / max(selected_tokens, 1))
    combined = policy_loss + args.ptd_coef * scaled
    local_tokens = sum(torch.clamp_min(mask.sum(), 1) for mask in batch["loss_masks"])
    metrics = {
        "ptd_jsd": scaled.detach(),
        "ptd_loss": (args.ptd_coef * scaled).detach(),
        "ptd_selected_fraction": local_tokens * selected_tokens / max(grpo_tokens, 1),
    }
    return combined, metrics


def attach_ptd_normalizers(args, rollout_data, data_iterators, num_microbatches):
    """One DP collective per optimizer step, before any forward/backward work."""
    if getattr(args, "ptd_coef", 0) == 0:
        return
    state = get_parallel_state()
    iterator = data_iterators[0]
    offset = 0
    result = [None] * len(rollout_data["tokens"])
    for count in num_microbatches:
        if iterator.micro_batch_indices is not None:
            indices = [i for mb in iterator.micro_batch_indices[offset:offset + count] for i in mb]
        else:
            start = offset * iterator.micro_batch_size
            indices = range(start, start + count * iterator.micro_batch_size)
        masks = rollout_data["loss_masks"]
        contexts = rollout_data["ptd_teacher_context"]
        grpo_tokens = sum(max(int(masks[i].sum()), 1) for i in indices)
        selected_tokens = sum(int(masks[i].sum()) for i in indices if contexts[i] is not None)
        counts = torch.tensor([grpo_tokens, selected_tokens], dtype=torch.int64, device=masks[0].device)
        if state.effective_dp.size > 1:
            dist.all_reduce(counts, group=state.effective_dp.group)
        pair = counts.tolist()
        for i in indices:
            result[i] = pair
        offset += count
    rollout_data["ptd_normalizers"] = result
