"""Recovery tests: Adam continuity, collective payloads, and fail-closed files."""

import copy
from types import SimpleNamespace

import pytest
import torch

from miles.backends.megatron_utils import lora_checkpoint_state as checkpoint


@pytest.fixture
def topology(monkeypatch):
    group = SimpleNamespace(rank=0, size=1)
    state = SimpleNamespace(**{name: group for name in ("tp", "pp", "cp", "ep", "etp", "intra_dp", "indep_dp")}, vpp_size=1)
    monkeypatch.setattr(checkpoint, "get_parallel_state", lambda: state)
    return state


def test_adam_next_update_is_exact_after_roundtrip(tmp_path):
    parameter = torch.nn.Parameter(torch.tensor([1.0, -2.0, 3.0]))
    optimizer = torch.optim.AdamW([parameter], lr=0.003)
    for gradient in ([0.1, 0.2, -0.3], [-0.4, 0.5, 0.6], [0.2, -0.1, 0.8]):
        parameter.grad = torch.tensor(gradient)
        optimizer.step()
    weights = parameter.detach().clone()
    state = {"optimizer": optimizer.state_dict(), "optimizer_layout": checkpoint.optimizer_layout(optimizer), "parameter_states": checkpoint.parameter_states(optimizer)}
    checkpoint.atomic_save(state, tmp_path / "state.pt")
    before = copy.deepcopy(optimizer.state[parameter])
    parameter.grad = torch.tensor([0.7, -0.2, 0.3])
    optimizer.step()

    restored = torch.nn.Parameter(weights)
    new_optimizer = torch.optim.AdamW([restored], lr=0.9)
    checkpoint.restore_optimizer(new_optimizer, torch.load(tmp_path / "state.pt", weights_only=False))
    for key, value in before.items():
        torch.testing.assert_close(value, new_optimizer.state[restored][key], rtol=0, atol=0)
    restored.grad = parameter.grad.clone()
    new_optimizer.step()
    torch.testing.assert_close(parameter, restored, rtol=0, atol=0)
    for key, value in optimizer.state[parameter].items():
        torch.testing.assert_close(value, new_optimizer.state[restored][key], rtol=0, atol=0)


def test_distributed_children_restore_metadata_before_scattering():
    events = []

    class Child:
        def get_parameter_state_dp_zero(self):
            return {"exp_avg": torch.tensor([0.5])}

        def load_parameter_state_from_dp_zero(self, state):
            events.append(state)

    class Chain:
        chained_optimizers = [Child(), Child()]

        def load_state_dict(self, state):
            events.append("metadata")

    optimizer = Chain()
    payloads = checkpoint.parameter_states(optimizer)
    payloads[1] = None  # A nonzero DP rank must still participate in scatter.
    checkpoint.restore_optimizer(optimizer, {"optimizer_layout": checkpoint.optimizer_layout(optimizer), "optimizer": {}, "parameter_states": payloads})
    assert events[0] == "metadata"
    assert events[1]["exp_avg"].item() == 0.5
    assert events[2] is None


def _write_checkpoint(path):
    checkpoint.begin_checkpoint(path)
    checkpoint.atomic_save({"lora_weight": torch.ones(3)}, path / "adapter_megatron_rank0.pt")
    checkpoint.atomic_save({"iteration": 3}, path / "training_state_rank0.pt")
    checkpoint.complete_checkpoint(path, iteration=3, training=True)


def test_checkpoint_missing_completion_fails_closed(tmp_path, topology):
    checkpoint.begin_checkpoint(tmp_path)
    with pytest.raises(RuntimeError, match="incomplete"):
        checkpoint.validate_checkpoint(tmp_path, training=True)


def test_checkpoint_corruption_fails_closed(tmp_path, topology):
    _write_checkpoint(tmp_path)
    manifest = checkpoint.validate_checkpoint(tmp_path, training=True)
    assert manifest["iteration"] == 3
    (tmp_path / "training_state_rank0.pt").write_bytes(b"truncated")
    with pytest.raises(RuntimeError, match="checksum"):
        checkpoint.validate_checkpoint(tmp_path, training=True)


def test_checkpoint_missing_rank_fails_closed(tmp_path, topology):
    _write_checkpoint(tmp_path)
    (tmp_path / "adapter_megatron_rank0.pt").unlink()
    with pytest.raises(RuntimeError, match="incomplete"):
        checkpoint.validate_checkpoint(tmp_path, training=True)


def test_changed_topology_fails_closed(tmp_path, topology):
    _write_checkpoint(tmp_path)
    topology.tp = SimpleNamespace(rank=0, size=2)
    with pytest.raises(RuntimeError, match="topology"):
        checkpoint.validate_checkpoint(tmp_path, training=True)


def test_completed_checkpoint_cannot_be_overwritten(tmp_path, topology):
    _write_checkpoint(tmp_path)
    with pytest.raises(RuntimeError, match="overwrite"):
        checkpoint.begin_checkpoint(tmp_path)


def test_legacy_checkpoint_is_explicitly_identified(tmp_path, topology):
    assert checkpoint.validate_checkpoint(tmp_path, training=True) is None


def test_rng_streams_roundtrip(monkeypatch):
    import random
    import sys
    from types import ModuleType

    import numpy as np

    tracker_state = {"model-parallel-rng": torch.tensor([2, 3])}
    tracker = SimpleNamespace(get_states=lambda: tracker_state.copy(), set_states=lambda state: tracker_state.update(state))
    module = ModuleType("megatron.core.tensor_parallel.random")
    module.get_cuda_rng_tracker = lambda: tracker
    monkeypatch.setitem(sys.modules, module.__name__, module)
    cuda_state = [torch.get_rng_state()]
    monkeypatch.setattr(torch.cuda, "get_rng_state", lambda: cuda_state[0])
    monkeypatch.setattr(torch.cuda, "set_rng_state", lambda state: cuda_state.__setitem__(0, state))
    state = checkpoint.capture_rng()
    expected = (random.random(), np.random.random(), torch.rand(5))
    tracker_state["model-parallel-rng"] = torch.tensor([9, 9])
    checkpoint.restore_rng(state)
    assert random.random() == expected[0]
    assert np.random.random() == expected[1]
    torch.testing.assert_close(torch.rand(5), expected[2], rtol=0, atol=0)
    torch.testing.assert_close(tracker_state["model-parallel-rng"], torch.tensor([2, 3]))


def test_preflight_rejects_missing_distributed_moments(tmp_path, topology):
    class Child:
        data_parallel_group = SimpleNamespace(rank=lambda: 0)

        def get_parameter_state_dp_zero(self):
            return {}

    optimizer = Child()
    state = {"version": 2, "iteration": 4, "optimizer_layout": checkpoint.optimizer_layout(optimizer), "parameter_states": [None]}
    checkpoint.atomic_save(state, tmp_path / "training_state_rank0.pt")
    with pytest.raises(RuntimeError, match="master parameters and moments"):
        checkpoint.preflight_training_state(tmp_path, {"iteration": 4}, optimizer, None)


@pytest.mark.parametrize(
    "restored,checkpoint_flag,iteration,expected",
    [
        (True, False, 4, []),
        (True, True, 4, []),
        (False, True, 4, []),
        (False, False, 4, [32]),
        (False, False, 0, [0]),
    ],
)
def test_scheduler_is_not_advanced_twice(restored, checkpoint_flag, iteration, expected):
    increments = []
    scheduler = SimpleNamespace(step=lambda *, increment: increments.append(increment))
    optimizer = SimpleNamespace(_lora_checkpoint_scheduler_restored=restored)
    checkpoint.advance_scheduler_after_load(scheduler, optimizer, iteration=iteration, global_batch_size=8, use_checkpoint_scheduler=checkpoint_flag)
    assert increments == expected


def test_actor_restores_rng_after_wake_and_only_once(monkeypatch):
    # Execute the real method without importing Ray/Megatron's actor dependencies.
    import ast
    from contextlib import ExitStack, nullcontext
    from pathlib import Path

    source = Path(__file__).parents[2] / "miles/backends/megatron_utils/actor.py"
    tree = ast.parse(source.read_text())
    actor = next(node for node in tree.body if isinstance(node, ast.ClassDef) and node.name == "MegatronTrainRayActor")
    method = copy.deepcopy(next(node for node in actor.body if isinstance(node, ast.FunctionDef) and node.name == "train"))
    method.decorator_list = []
    module = ast.Module(body=[ast.ImportFrom(module="__future__", names=[ast.alias(name="annotations")], level=0), method], type_ignores=[])
    ast.fix_missing_locations(module)
    events = []
    monkeypatch.setattr(checkpoint, "restore_rng", lambda state: events.append("restore"))
    namespace = {"ExitStack": ExitStack, "timer": lambda _: nullcontext(), "get_rollout_data": lambda *a, **kw: ({}, nullcontext()), "restore_pending_rng": checkpoint.restore_pending_rng}
    exec(compile(module, str(source), "exec"), namespace)
    instance = SimpleNamespace(
        _heartbeat=SimpleNamespace(bump=lambda: None), args=SimpleNamespace(offload_train=True, debug_rollout_only=False), _asleep=True, wake_up=lambda: events.append("wake"), optimizer=SimpleNamespace(_lora_checkpoint_rng_state={}), role="actor", train_actor=lambda *a, **kw: events.append("forward")
    )
    namespace["train"](instance, 4, None)
    namespace["train"](instance, 5, None)
    assert events == ["wake", "restore", "forward", "wake", "forward"]


@pytest.mark.parametrize(
    "payload",
    [
        {},
        {"buckets_coalesced": True},
        {"buckets_coalesced": True, 0: {(torch.bfloat16, torch.float32): {"numel_unpadded": 3, "param": torch.zeros(2)}}},
    ],
)
def test_preflight_rejects_malformed_distributed_payload(payload):
    dtype = (torch.bfloat16, torch.float32)
    optimizer = SimpleNamespace(data_parallel_group=SimpleNamespace(rank=lambda: 0), gbuf_ranges=[{dtype: []}], buffers=[SimpleNamespace(numel_unpadded=3)])
    with pytest.raises((ValueError, KeyError)):
        checkpoint._validate_parameter_payload(optimizer, payload)


def test_base_metadata_and_lora_config_are_pinned(tmp_path, topology):
    (tmp_path / "config.json").write_text('{"model_type":"qwen3_5"}')
    args = SimpleNamespace(hf_checkpoint=str(tmp_path), lora_alpha=32)
    original = checkpoint.checkpoint_configuration(args)
    checkpoint.begin_checkpoint(tmp_path)
    checkpoint.atomic_save({}, tmp_path / "adapter_megatron_rank0.pt")
    checkpoint.complete_checkpoint(tmp_path, iteration=1, training=False, configuration=original)
    args.lora_alpha = 64
    with pytest.raises(RuntimeError, match="configuration changed"):
        checkpoint.validate_checkpoint(tmp_path, training=False, configuration=checkpoint.checkpoint_configuration(args))
    args.lora_alpha = 32
    (tmp_path / "config.json").write_text('{"model_type":"different"}')
    with pytest.raises(RuntimeError, match="configuration changed"):
        checkpoint.validate_checkpoint(tmp_path, training=False, configuration=checkpoint.checkpoint_configuration(args))


def test_legacy_adapter_and_optimizer_still_load(tmp_path, topology, monkeypatch):
    from miles.backends.megatron_utils import lora_utils

    monkeypatch.setattr(lora_utils, "get_parallel_state", lambda: topology)
    model = torch.nn.Module()
    model.register_parameter("lora_weight", torch.nn.Parameter(torch.zeros(3)))
    optimizer = torch.optim.AdamW(model.parameters(), lr=.003)
    torch.save({"lora_weight": torch.ones(3)}, tmp_path / "adapter_megatron_rank0.pt")
    torch.save({"iteration": 5, "optimizer": optimizer.state_dict(), "opt_param_scheduler": None},
               tmp_path / "training_state_rank0.pt")
    assert lora_utils.load_lora_adapter([model], str(tmp_path), optimizer=optimizer) == (True, 5)
    torch.testing.assert_close(model.lora_weight, torch.ones(3))
    assert not hasattr(optimizer, "_lora_checkpoint_rng_state")


def test_adapter_shape_error_is_collectively_reported(tmp_path, topology):
    model = torch.nn.Module()
    model.register_parameter("lora_weight", torch.nn.Parameter(torch.zeros(3)))
    torch.save({"lora_weight": torch.ones(2)}, tmp_path / "adapter.pt")
    with pytest.raises(RuntimeError, match="shape/dtype mismatch"):
        checkpoint.load_validated_adapter(tmp_path / "adapter.pt", [model])
