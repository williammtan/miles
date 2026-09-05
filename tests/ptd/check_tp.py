"""torchrun --standalone --nproc-per-node=2 tests/ptd/check_tp.py [--backend nccl]."""

import argparse
import json
import os
from types import SimpleNamespace

import torch
import torch.distributed as dist

from miles.backends.training_utils.loss_hub.ptd import attach_ptd_normalizers
from miles.backends.training_utils.loss_hub.ptd_math import sparse_vocab_parallel_jsd, support_union, vocab_parallel_topk
from miles.backends.training_utils.parallel import set_parallel_state


def dense_reference(logits, q_logits, ids):
    p_full, q_full = logits.softmax(-1), q_logits.softmax(-1).detach()
    values = []
    for p, q, row in zip(p_full, q_full, ids, strict=True):
        support = row[row >= 0]
        tail = torch.ones(p.shape, dtype=torch.bool, device=p.device)
        tail[support] = False
        p = torch.cat((p[support], p[tail].sum().view(1)))
        q = torch.cat((q[support], q[tail].sum().view(1)))
        m = (p + q) / 2
        tiny = torch.finfo(p.dtype).tiny
        values.append(((p * (p.clamp_min(tiny).log() - m.clamp_min(tiny).log())).sum()
                       + (q * (q.clamp_min(tiny).log() - m.clamp_min(tiny).log())).sum()) / 2)
    return torch.stack(values)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--backend", default="gloo", choices=("gloo", "nccl"))
    args = parser.parse_args()
    if args.backend == "nccl":
        torch.cuda.set_device(int(os.environ["LOCAL_RANK"]))
    device = torch.device("cuda", int(os.environ["LOCAL_RANK"])) if args.backend == "nccl" else torch.device("cpu")
    dist.init_process_group(args.backend)
    rank, world = dist.get_rank(), dist.get_world_size()
    assert world == 2
    torch.manual_seed(204)
    full = torch.randn(5, 11, dtype=torch.float64, device=device, requires_grad=True)
    teacher = torch.randn_like(full)
    padded = torch.cat((full.detach(), torch.full((5, 1), 500., device=device, dtype=full.dtype)), -1)
    local = padded[:, rank * 6:(rank + 1) * 6].clone().requires_grad_()
    student_ids = vocab_parallel_topk(local, 3, vocab_start=rank * 6, vocab_size=11, tp_group=dist.group.WORLD, chunk_size=2)
    assert torch.equal(student_ids, full.detach().topk(3).indices)
    ids = support_union(student_ids, teacher.topk(3).indices)
    q_lp = teacher.log_softmax(-1).gather(-1, ids.clamp_min(0))
    sparse = sparse_vocab_parallel_jsd(local, ids, q_lp, vocab_start=rank * 6, vocab_size=11,
                                      tp_group=dist.group.WORLD, chunk_size=2)
    reference = dense_reference(full, teacher, ids)
    weights = torch.tensor([1., 0., 0.2, 1.5, 0.8], dtype=full.dtype, device=device)
    (sparse * weights).sum().backward()
    (reference * weights).sum().backward()
    reference_grad = torch.cat((full.grad, torch.zeros(5, 1, device=device, dtype=full.dtype)), -1)
    expected_local = reference_grad[:, rank * 6:(rank + 1) * 6]
    torch.testing.assert_close(sparse, reference, atol=2e-14, rtol=2e-12)
    torch.testing.assert_close(local.grad, expected_local, atol=2e-14, rtol=2e-11)
    # Exercise the actual DP collective used to make optimizer-step normalizers.
    set_parallel_state(SimpleNamespace(effective_dp=SimpleNamespace(size=2, group=dist.group.WORLD)))
    data = {"tokens": [None, None], "loss_masks": [torch.ones(3 + rank, device=device), torch.ones(2, device=device)],
            "ptd_teacher_context": [None if rank == 0 else {}, {}]}
    attach_ptd_normalizers(SimpleNamespace(ptd_coef=0.5), data, [SimpleNamespace(micro_batch_indices=[[0], [1]])], [2])
    assert data["ptd_normalizers"] == [[11, 8], [11, 8]]
    report = {"backend": args.backend, "rank": rank, "world_size": world, "loss_max_error": (sparse-reference).abs().max().item(),
              "grad_max_error": (local.grad-expected_local).abs().max().item(), "grad_norm": local.grad.norm().item(),
              "dp_normalizers": data["ptd_normalizers"], "device": str(device)}
    print(json.dumps(report), flush=True)
    dist.destroy_process_group()


if __name__ == "__main__":
    main()
