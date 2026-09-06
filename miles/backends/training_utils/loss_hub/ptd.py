"""PTD-PO integration for Megatron TP and CP=1 packed response logits."""

import torch
import torch.distributed as dist
from miles.backends.training_utils.loss_hub.logit_processors import _iter_response_chunks
from miles.backends.training_utils.loss_hub.ptd_math import (
    sparse_vocab_parallel_jsd,
    teacher_log_probs_at_student_topk,
    vocab_parallel_topk,
)
from miles.backends.training_utils.parallel import get_parallel_state


def ptd_loss_sum(args, batch, logits):
    """Unnormalized selected-token JSD sum, with gradients through student only."""
    state = get_parallel_state()
    assert state.cp.size == 1 and args.qkv_format == "thd"
    total = logits.reshape(-1)[:1].float().sum() * 0
    teacher_ids = batch.get("ptd_teacher_ids")
    teacher_log_probs = batch.get("ptd_teacher_log_probs")
    contexts = batch.get("ptd_teacher_context")
    if teacher_ids is None or teacher_log_probs is None or contexts is None:
        raise ValueError("PTD enabled but teacher-target fields are missing from the training batch")
    tp_group = state.tp.group if state.tp.size > 1 else None
    vocab_start = state.tp.rank * logits.shape[-1]
    chunks = _iter_response_chunks(
        logits, args=args, unconcat_tokens=batch["unconcat_tokens"], total_lengths=batch["total_lengths"],
        response_lengths=batch["response_lengths"], include_response_indices=False,
    )
    for index, (response_logits, response_tokens, _) in enumerate(chunks):
        stored_ids = teacher_ids[index]
        stored_log_probs = teacher_log_probs[index]
        context = contexts[index]
        if context is None:
            if stored_ids.numel() or stored_log_probs.numel():
                raise ValueError("PTD inactive response unexpectedly contains teacher targets")
            continue
        if context.get("score_mode") != "precomputed_teacher_topk_tail_v1":
            raise ValueError("PTD teacher targets use an incompatible scoring objective")
        if response_tokens.tolist() != context.get("response_tokens"):
            raise ValueError("PTD packed-training response positions differ from the teacher continuation")
        if stored_ids.numel() == 0 or response_tokens.numel() == 0:
            raise ValueError("PTD active response is missing precomputed teacher Top-K targets")
        if stored_ids.shape != stored_log_probs.shape or stored_ids.shape[0] != response_tokens.numel():
            raise ValueError("PTD packed-training response positions differ from the stored teacher targets")
        student_ids = vocab_parallel_topk(
            response_logits, args.ptd_top_k, vocab_start=vocab_start, vocab_size=args.ptd_vocab_size,
            tp_group=tp_group, chunk_size=args.ptd_logits_chunk_size,
        )
        teacher_at_student = teacher_log_probs_at_student_topk(
            student_ids, stored_ids, stored_log_probs, vocab_size=args.ptd_vocab_size,
            chunk_size=args.ptd_logits_chunk_size,
        )
        token_losses = sparse_vocab_parallel_jsd(
            response_logits, student_ids, teacher_at_student,
            vocab_start=vocab_start, vocab_size=args.ptd_vocab_size,
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
        teacher_ids = rollout_data["ptd_teacher_ids"]
        grpo_tokens = sum(max(int(masks[i].sum()), 1) for i in indices)
        selected_tokens = sum(int(masks[i].sum()) for i in indices if teacher_ids[i].numel() > 0)
        counts = torch.tensor([grpo_tokens, selected_tokens], dtype=torch.int64, device=masks[0].device)
        if state.effective_dp.size > 1:
            dist.all_reduce(counts, group=state.effective_dp.group)
        pair = counts.tolist()
        for i in indices:
            result[i] = pair
        offset += count
    rollout_data["ptd_normalizers"] = result
