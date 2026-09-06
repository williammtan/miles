"""Bounded GPU probe of the public LoRA checkpoint APIs and installed Megatron.

Run with two otherwise idle GPUs and an empty shared checkpoint directory::

    CUDA_VISIBLE_DEVICES=0,1 torchrun --standalone --nproc_per_node=2 \
        tests/manual/check_lora_checkpoint_resume.py --checkpoint-dir /tmp/lora-resume-probe

This tiny BF16 DP2 test proves optimizer continuity, not full-model recovery.
HF export is deliberately skipped; this probe checks the native training format.
"""

import argparse
import copy
import os
import random
from types import SimpleNamespace

import numpy as np
import torch
import torch.distributed as dist
from megatron.bridge import AutoBridge
from megatron.core import parallel_state
from megatron.core.distributed import DistributedDataParallel, DistributedDataParallelConfig
from megatron.core.optimizer import OptimizerConfig, get_megatron_optimizer
from megatron.core.tensor_parallel.random import model_parallel_cuda_manual_seed
from megatron.core.transformer import TransformerConfig

from miles.backends.megatron_utils.lora_utils import load_lora_adapter, save_lora_checkpoint
from miles.backends.training_utils.parallel import set_parallel_state


class _Net(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.lora_in = torch.nn.Linear(256, 256, bias=False, device="cuda", dtype=torch.bfloat16)
        self.lora_out = torch.nn.Linear(256, 16, bias=False, device="cuda", dtype=torch.bfloat16)

    def forward(self, inputs):
        return self.lora_out(self.lora_in(inputs))


def _make():
    torch.manual_seed(72)
    model = DistributedDataParallel(
        TransformerConfig(num_attention_heads=1, num_layers=1, hidden_size=256),
        DistributedDataParallelConfig(use_distributed_optimizer=True),
        _Net(),
    )
    optimizer = get_megatron_optimizer(OptimizerConfig(optimizer="adam", lr=0.001, bf16=True, use_distributed_optimizer=True), [model])
    return model, optimizer


def _step(model, optimizer, index):
    model.zero_grad_buffer()
    optimizer.zero_grad()
    torch.manual_seed(123 + index)
    inputs = torch.randn(8, 256, device="cuda", dtype=torch.bfloat16)
    model(inputs).float().square().mean().backward()
    model.finish_grad_sync()
    optimizer.step()
    model.start_param_sync()


def _assert_equal(left, right):
    if isinstance(left, torch.Tensor):
        torch.testing.assert_close(left, right, rtol=0, atol=0)
    elif isinstance(left, dict):
        assert left.keys() == right.keys()
        for key in left:
            _assert_equal(left[key], right[key])
    elif isinstance(left, (tuple, list)):
        assert len(left) == len(right)
        for first, second in zip(left, right, strict=True):
            _assert_equal(first, second)
    else:
        assert left == right


def _skip_export(*args, **kwargs):
    raise RuntimeError("HF export intentionally skipped by native recovery probe")


def _probe(path):
    model, optimizer = _make()
    model_parallel_cuda_manual_seed(321)
    for index in range(3):
        _step(model, optimizer, index)
    AutoBridge.from_hf_pretrained = _skip_export
    save_lora_checkpoint([model], SimpleNamespace(hf_checkpoint="unused"), path, optimizer=optimizer, iteration=3)
    rng_expected = (random.random(), np.random.random(), torch.rand(10), torch.rand(10, device="cuda"))
    saved_inner = copy.deepcopy(optimizer.chained_optimizers[0].optimizer.state_dict())
    _step(model, optimizer, 3)
    expected_weights = copy.deepcopy(model.state_dict())
    expected_inner = copy.deepcopy(optimizer.chained_optimizers[0].optimizer.state_dict())
    restored_model, restored_optimizer = _make()
    assert load_lora_adapter([restored_model], path, optimizer=restored_optimizer, checkpoint_args=SimpleNamespace(hf_checkpoint="unused")) == (True, 3)
    _assert_equal((random.random(), np.random.random(), torch.rand(10), torch.rand(10, device="cuda")), rng_expected)
    _assert_equal(saved_inner, restored_optimizer.chained_optimizers[0].optimizer.state_dict())
    _step(restored_model, restored_optimizer, 3)
    _assert_equal(expected_weights, restored_model.state_dict())
    _assert_equal(expected_inner, restored_optimizer.chained_optimizers[0].optimizer.state_dict())
    print(f"PASS rank {dist.get_rank()}: public LoRA save/load, RNG, Adam moments, next update bitwise identical", flush=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint-dir", required=True)
    args = parser.parse_args()
    torch.cuda.set_device(int(os.environ["LOCAL_RANK"]))
    dist.init_process_group("nccl")
    try:
        assert dist.get_world_size() == 2, "This bounded probe requires exactly two ranks"
        parallel_state.initialize_model_parallel()
        group = SimpleNamespace(rank=0, size=1)
        dp = SimpleNamespace(rank=dist.get_rank(), size=dist.get_world_size())
        set_parallel_state(
            SimpleNamespace(
                **{name: group for name in ("tp", "pp", "cp", "ep", "etp", "indep_dp")},
                intra_dp=dp,
                effective_dp=dp,
                vpp_size=1,
            )
        )
        _probe(args.checkpoint_dir)
    finally:
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
