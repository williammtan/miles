"""Exercise the actual LoRA provider setup and global PTD reduction contract."""

import importlib.util
import sys
from argparse import Namespace
from pathlib import Path
from types import ModuleType, SimpleNamespace

import pytest
import torch
import torch.distributed as dist
import torch.multiprocessing as mp

from miles.backends.training_utils.loss_hub.ptd import add_ptd_loss, attach_ptd_normalizers
from miles.backends.training_utils.parallel import set_parallel_state


@pytest.mark.parametrize("requested,ptd_coef,override,raises", [
    (True, 0.05, None, False), (False, 0, None, False),
    (False, 0.05, None, True), (True, 0.05, False, True),
])
def test_lora_provider_preserves_requested_loss_normalization(monkeypatch, requested, ptd_coef, override, raises):
    events = []

    class Provider:
        calculate_per_token_loss = False

        def finalize(self):
            events.append(("finalize", self.calculate_per_token_loss))
            if override is not None:
                self.calculate_per_token_loss = override

        def register_pre_wrap_hook(self, hook):
            events.append(("hook", hook))

        def provide_distributed_model(self, **kwargs):
            events.append(("build", self.calculate_per_token_loss))
            return [self]

    provider = Provider()

    def module(name, **attributes):
        fake = ModuleType(name)
        fake.__dict__.update(attributes)
        monkeypatch.setitem(sys.modules, name, fake)

    bridge = SimpleNamespace(to_megatron_provider=lambda **kw: provider)
    module("megatron.core.utils", get_attr_wrapped_model=lambda *a, **kw: None)
    module("megatron.bridge", AutoBridge=SimpleNamespace(from_hf_pretrained=lambda *a, **kw: bridge))
    module("megatron.bridge.training.config", DistributedDataParallelConfig=lambda **kw: SimpleNamespace(finalize=lambda: None))
    module("miles.utils.hf_config", load_hf_config=lambda _: SimpleNamespace(architectures=["Qwen3_5ForConditionalGeneration"]))
    module("miles.utils.multi_lora", is_multi_lora_enabled=lambda _: False, targets_expert_leaves=lambda _: False)
    module("miles.backends.megatron_utils.lora_utils", convert_target_modules_to_hf=lambda x: x,
           patch_param_grad_buffer_for_colocate_mode_lora=lambda: None, create_lora_instance=lambda _: object())
    name = "miles.backends.megatron_utils._ptd_test_bridge"
    path = Path(__file__).parents[2] / "miles/backends/megatron_utils/bridge_lora_helpers.py"
    spec = importlib.util.spec_from_file_location(name, path)
    helper = importlib.util.module_from_spec(spec)
    monkeypatch.setitem(sys.modules, name, helper)
    spec.loader.exec_module(helper)
    args = Namespace(
        hf_checkpoint="unused", tensor_model_parallel_size=2, pipeline_model_parallel_size=1,
        expert_model_parallel_size=1, expert_tensor_parallel_size=1, sequence_parallel=True,
        virtual_pipeline_model_parallel_size=None, context_parallel_size=1, gradient_accumulation_fusion=True,
        recompute_granularity=None, recompute_method=None, recompute_num_layers=None, recompute_modules=[],
        distribute_saved_activations=False, attention_backend="auto", calculate_per_token_loss=requested,
        ptd_coef=ptd_coef, optimizer="adam", accumulate_allreduce_grads_in_fp32=True, offload_train=False,
    )
    if raises:
        with pytest.raises(RuntimeError, match="PTD requires calculate_per_token_loss=True"):
            helper._setup_lora_model_via_bridge(args)
        assert not any(event[0] == "build" for event in events)
    else:
        assert helper._setup_lora_model_via_bridge(args) == [provider]
        assert events[-1] == ("build", requested)
    assert events[0] == ("finalize", requested)


def _global_reduction_worker(rank, rendezvous):
    dist.init_process_group("gloo", init_method=rendezvous, rank=rank, world_size=2)
    try:
        set_parallel_state(SimpleNamespace(
            cp=SimpleNamespace(size=1), tp=SimpleNamespace(size=1, rank=0, group=None),
            effective_dp=SimpleNamespace(size=2, group=dist.group.WORLD),
        ))
        args = Namespace(ptd_coef=0.05, qkv_format="thd", ptd_top_k=3, ptd_vocab_size=3,
                         ptd_logits_chunk_size=2, ptd_score_timeout=1, true_on_policy_mode=True, rollout_temperature=1)
        lengths = [1, 4] if rank == 0 else [2, 5]
        masks = [torch.ones(n) for n in lengths]
        data = {"tokens": [None, None], "loss_masks": masks,
                "ptd_teacher_context": [None, None] if rank == 0 else [{}, {}]}
        attach_ptd_normalizers(args, data, [SimpleNamespace(micro_batch_indices=[[0], [1]])], [2])
        assert data["ptd_normalizers"] == [[12, 7], [12, 7]]
        weight = torch.tensor(0.7, dtype=torch.float64, requires_grad=True)
        q = torch.tensor([0.2, 0.3, 0.5], dtype=torch.float64)
        features = torch.tensor([-1., 0.5, 2.], dtype=torch.float64)
        local_loss = weight * 0
        tutor_sum = weight * 0
        for n, mask in zip(lengths, masks, strict=True):
            logits = (weight * features * n / 3).expand(1, n + 1, 3)
            batch = {
                "unconcat_tokens": [torch.ones(n + 1, dtype=torch.long)], "total_lengths": [n + 1],
                "response_lengths": [n], "loss_masks": [mask], "ptd_normalizers": [[12, 7]],
                "ptd_teacher_context": [None if rank == 0 else {"response_tokens": [1] * n}],
                "ptd_teacher_ids": [torch.arange(3).expand(n, -1)],
                "ptd_teacher_log_probs": [q.log().expand(n, -1)],
            }
            # A differentiable stand-in policy numerator isolates PTD's reduction contract.
            policy_sum = weight.square() * n
            combined, _ = add_ptd_loss(args, batch, logits, policy_sum)
            tutor_sum = tutor_sum + combined - policy_sum
            local_loss = local_loss + combined
        if rank == 0:
            assert tutor_sum.item() == 0
            assert torch.autograd.grad(tutor_sum, weight, retain_graph=True)[0].item() == 0
        # Megatron's per-token path sums microbatch/DP gradients, then divides by global tokens.
        local_loss.backward()
        actual = torch.stack((local_loss.detach(), weight.grad))
        dist.all_reduce(actual)
        actual /= 12
        ref_weight = weight.detach().clone().requires_grad_()
        selected_sum = ref_weight * 0
        for n in [2, 5]:
            p = (ref_weight * features * n / 3).softmax(-1)
            mixture = (p + q) / 2
            dense_jsd = ((p * (p.log() - mixture.log())).sum() + (q * (q.log() - mixture.log())).sum()) / 2
            selected_sum = selected_sum + n * dense_jsd
        reference = ref_weight.square() + args.ptd_coef * selected_sum / 7
        reference.backward()
        torch.testing.assert_close(actual[0], reference, atol=1e-9, rtol=1e-8)
        torch.testing.assert_close(actual[1], ref_weight.grad, atol=1e-9, rtol=1e-8)
    finally:
        dist.destroy_process_group()


def test_unequal_dp_microbatches_and_zero_selected_rank_match_global_reference(tmp_path):
    mp.spawn(_global_reduction_worker, args=(f"file://{tmp_path / 'gloo-init'}",), nprocs=2, join=True)
