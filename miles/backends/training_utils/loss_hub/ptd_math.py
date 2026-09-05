"""Exact coarsened JSD on vocabulary-sharded logits (PTD-PO, equations 13–17)."""

import torch
import torch.distributed as dist
from torch.utils.checkpoint import checkpoint


class _SumVocabularyShards(torch.autograd.Function):
    """Sum disjoint vocab contributions; every TP rank evaluates the same loss.

    Backward is the identity, not another all-reduce: the upstream derivative
    is already replicated. A differentiable generic all-reduce would multiply
    that derivative by TP size.
    """

    @staticmethod
    def forward(ctx, value, group):
        result = value.clone()
        dist.all_reduce(result, group=group)
        return result

    @staticmethod
    def backward(ctx, grad):
        return grad, None


def _sum_shards(value, tp_group):
    return value if tp_group is None else _SumVocabularyShards.apply(value, tp_group)


def _local_vocab_logits(logits, vocab_start, vocab_size):
    values = logits if logits.dtype == torch.float64 else logits.float()
    if vocab_size < vocab_start + values.shape[-1]:
        ids = torch.arange(values.shape[-1], device=values.device) + vocab_start
        values = values.masked_fill(ids >= vocab_size, -torch.inf)
    return values


@torch.no_grad()
def vocab_parallel_topk(logits, k, *, vocab_start=0, vocab_size=None, tp_group=None, chunk_size=128):
    """Global top-K IDs, communicating only local top-K candidates per row."""
    vocab_size = vocab_size or logits.shape[-1]
    chunks = []
    for chunk in logits.split(chunk_size):
        values = _local_vocab_logits(chunk, vocab_start, vocab_size)
        local_k = min(k, values.shape[-1])
        top_values, top_ids = values.topk(local_k, dim=-1)
        top_ids = top_ids + vocab_start
        if tp_group is not None:
            world = dist.get_world_size(tp_group)
            all_values = [torch.empty_like(top_values) for _ in range(world)]
            all_ids = [torch.empty_like(top_ids) for _ in range(world)]
            dist.all_gather(all_values, top_values.contiguous(), group=tp_group)
            dist.all_gather(all_ids, top_ids.contiguous(), group=tp_group)
            top_values, top_ids = torch.cat(all_values, -1), torch.cat(all_ids, -1)
        selected = top_values.topk(min(k, vocab_size), dim=-1).indices
        chunks.append(top_ids.gather(-1, selected))
    return torch.cat(chunks) if chunks else torch.empty((0, min(k, vocab_size)), device=logits.device, dtype=torch.long)


def support_union(student_ids, teacher_ids):
    """Padded union: duplicate IDs become -1, never double-counted in the tail."""
    ids = torch.cat((student_ids, teacher_ids), dim=-1).sort(dim=-1).values
    duplicate = torch.zeros_like(ids, dtype=torch.bool)
    duplicate[..., 1:] = ids[..., 1:] == ids[..., :-1]
    return ids.masked_fill(duplicate, -1)


def _validate_probability_mass(probs, valid):
    selected = probs.detach().masked_fill(~valid, 0)
    if (not torch.isfinite(selected).all() or (selected < 0).any()
            or (selected.sum(-1) > 1 + 1e-5).any()):
        raise ValueError("PTD sparse probability mass must be finite, nonnegative, and at most one")


def _validate_sparse_support(ids, teacher_log_probs, vocab_size):
    if ids.dtype not in (torch.int32, torch.int64) or ((ids < -1) | (ids >= vocab_size)).any():
        raise ValueError("PTD support IDs must be integer vocabulary IDs or -1 padding")
    if ids.shape != teacher_log_probs.shape:
        raise ValueError("PTD support and teacher probabilities must have identical shapes")
    ordered = ids.sort(-1).values
    if ((ordered[..., 1:] == ordered[..., :-1]) & (ordered[..., 1:] >= 0)).any():
        raise ValueError("PTD support contains duplicate vocabulary IDs")
    valid = ids >= 0
    selected = teacher_log_probs.detach()[valid]
    if not torch.isfinite(selected).all() or (selected > 0).any():
        raise ValueError("PTD teacher log probabilities must be finite and nonpositive")
    _validate_probability_mass(teacher_log_probs.detach().exp(), valid)


def coarsened_jsd(student_probs, teacher_probs, valid):
    """True JSD; only teacher probabilities are stop-gradient.

    Zero buckets use the continuous extension 0 log(0) = 0. Clamping is only
    inside logarithms, so full-vocabulary support and empty tails stay exact.
    """
    _validate_probability_mass(student_probs, valid)
    _validate_probability_mass(teacher_probs, valid)
    p = student_probs.masked_fill(~valid, 0)
    q = teacher_probs.detach().masked_fill(~valid, 0)
    p = torch.cat((p, (1 - p.sum(-1, keepdim=True)).clamp_min(0)), -1)
    q = torch.cat((q, (1 - q.sum(-1, keepdim=True)).clamp_min(0)), -1)
    m = (p + q) * 0.5
    tiny = torch.finfo(p.dtype).tiny
    log_m = m.clamp_min(tiny).log()
    return 0.5 * ((p * (p.clamp_min(tiny).log() - log_m)).sum(-1)
                  + (q * (q.clamp_min(tiny).log() - log_m)).sum(-1))


def _sparse_jsd_chunk(logits, ids, teacher_log_probs, vocab_start, vocab_size, tp_group):
    values = _local_vocab_logits(logits, vocab_start, vocab_size)
    maximum = values.detach().amax(-1, keepdim=True)
    if tp_group is not None:
        dist.all_reduce(maximum, op=dist.ReduceOp.MAX, group=tp_group)
    log_z = _sum_shards((values - maximum).exp().sum(-1, keepdim=True), tp_group).log() + maximum
    local_ids = ids - vocab_start
    local = (local_ids >= 0) & (local_ids < logits.shape[-1]) & (ids < vocab_size)
    selected = values.gather(-1, local_ids.clamp(0, logits.shape[-1] - 1)).masked_fill(~local, 0)
    selected = _sum_shards(selected, tp_group)
    valid = ids >= 0
    p = (selected - log_z).exp().masked_fill(~valid, 0)
    q = teacher_log_probs.detach().to(values.dtype).exp()
    return coarsened_jsd(p, q, valid)


def sparse_vocab_parallel_jsd(
    logits, ids, teacher_log_probs, *, vocab_start=0, vocab_size=None, tp_group=None, chunk_size=128,
):
    """Per-response-token loss; no full-vocabulary teacher or gathered student.

    Checkpointing retains only native model logits and O(RK) sparse tensors;
    the fp32 vocabulary normalization workspace is bounded by chunk_size.
    """
    vocab_size = vocab_size or logits.shape[-1]
    _validate_sparse_support(ids, teacher_log_probs, vocab_size)
    if logits.shape[0] != ids.shape[0]:
        raise ValueError("PTD logits and support row counts differ")
    if logits.shape[0] == 0:
        return logits.sum(-1) * 0
    chunks = []
    for start in range(0, logits.shape[0], chunk_size):
        end = start + chunk_size
        chunks.append(checkpoint(
            _sparse_jsd_chunk, logits[start:end], ids[start:end], teacher_log_probs[start:end],
            vocab_start, vocab_size, tp_group, use_reentrant=False,
        ))
    return torch.cat(chunks)
