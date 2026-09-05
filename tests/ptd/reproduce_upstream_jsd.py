"""Reproduce the pinned upstream JSD gradient issue without importing verl."""

import argparse
import ast
import json
import math
from pathlib import Path

import torch


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--upstream", type=Path, required=True)
    args = parser.parse_args()
    path = args.upstream / "verl/trainer/core_algos.py"
    source = path.read_text()
    functions = [node for node in ast.parse(source).body if isinstance(node, ast.FunctionDef)
                 and node.name in {"_topk_match_and_gather", "compute_topk_kl"}]
    assert len(functions) == 2
    namespace = {"torch": torch, "math": math}
    exec(compile(ast.Module(body=functions, type_ignores=[]), str(path), "exec"), namespace)
    logits = torch.tensor([[[2., 1., -1., 0.5]]], requires_grad=True)
    teacher = torch.tensor([[[-1., 3., 0.5, 0.2]]]).log_softmax(-1)
    student_lp, student_ids = logits.log_softmax(-1).topk(2)
    teacher_lp, teacher_ids = teacher.topk(2)
    upstream = namespace["compute_topk_kl"](
        student_lp, student_lp.exp(), student_ids, teacher_lp, teacher_ids, kl_direction="jsd_kl",
    )
    upstream_grad = torch.autograd.grad(upstream.sum(), logits, retain_graph=True)[0]
    # Same approximate probabilities/support as upstream: isolate only autodiff.
    p = student_lp.exp()
    q = namespace["_topk_match_and_gather"](student_ids, teacher_lp, teacher_ids).exp().detach()
    p = torch.cat((p, (1 - p.sum(-1, keepdim=True)).clamp_min(1e-8)), -1)
    q = torch.cat((q, (1 - q.sum(-1, keepdim=True)).clamp_min(1e-8)), -1)
    mixture = (p + q) / 2
    reference = ((p * (p.log() - mixture.log())).sum(-1)
                 + (q * (q.log() - mixture.log())).sum(-1)) / 2
    reference_grad = torch.autograd.grad(reference.sum(), logits)[0]
    assert torch.allclose(upstream, reference, atol=1e-6)
    assert reference_grad.norm() > 1e-3
    assert upstream_grad.norm() < reference_grad.norm() * 1e-5
    print(json.dumps({"upstream_loss": upstream.item(), "reference_loss": reference.item(),
                      "upstream_grad_norm": upstream_grad.norm().item(),
                      "reference_grad_norm": reference_grad.norm().item(),
                      "finding": "Detached student weights suppress the intended JSD gradient"}))


if __name__ == "__main__":
    main()
