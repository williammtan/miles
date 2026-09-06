"""Versioned, topology-pinned LoRA training checkpoints (trusted local files)."""

import hashlib
import json
import os
import random
from pathlib import Path

import numpy as np
import torch
import torch.distributed as dist

from miles.backends.training_utils.parallel import get_parallel_state

_FORMAT = 2
_MARKER = "checkpoint_format.json"
_COMPLETE = "checkpoint_complete.json"


def _rank():
    return dist.get_rank() if dist.is_initialized() else 0


def _gather(value):
    if not dist.is_initialized():
        return [value]
    values = [None] * dist.get_world_size()
    dist.all_gather_object(values, value)
    return values


def _atomic_write(path, write):
    temporary = path.with_name(path.name + ".tmp")
    with temporary.open("wb") as stream:
        write(stream)
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(temporary, path)
    directory = os.open(path.parent, os.O_DIRECTORY)
    try:
        os.fsync(directory)
    finally:
        os.close(directory)


def atomic_save(value, path):
    _atomic_write(path, lambda stream: torch.save(value, stream))


def _json_save(value, path):
    _atomic_write(path, lambda stream: stream.write(json.dumps(value, sort_keys=True).encode()))


def _topology():
    state = get_parallel_state()
    return {
        "world_size": dist.get_world_size() if dist.is_initialized() else 1,
        "groups": {name: [getattr(state, name).rank, getattr(state, name).size] for name in ("tp", "pp", "cp", "ep", "etp", "intra_dp", "indep_dp")},
        "vpp_size": state.vpp_size,
    }


def capture_rng():
    # Megatron is an optional dependency outside the training backend.
    from megatron.core.tensor_parallel.random import get_cuda_rng_tracker

    return {"python": random.getstate(), "numpy": np.random.get_state(), "torch": torch.get_rng_state(), "cuda": torch.cuda.get_rng_state(), "megatron": get_cuda_rng_tracker().get_states()}


def restore_rng(state):
    from megatron.core.tensor_parallel.random import get_cuda_rng_tracker

    random.setstate(state["python"])
    np.random.set_state(state["numpy"])
    torch.set_rng_state(state["torch"])
    torch.cuda.set_rng_state(state["cuda"])
    get_cuda_rng_tracker().set_states(state["megatron"])


def _optimizers(optimizer):
    return list(getattr(optimizer, "chained_optimizers", [optimizer]))


def parameter_states(optimizer):
    """Collectively gather masters and moments through Megatron's DP-zero API."""
    return [child.get_parameter_state_dp_zero() if hasattr(child, "get_parameter_state_dp_zero") and not getattr(child, "is_stub_optimizer", False) else None for child in _optimizers(optimizer)]


def optimizer_layout(optimizer):
    layouts = []
    for child in _optimizers(optimizer):
        buffers = []
        for buffer in getattr(child, "buffers", []):
            buffers.append(
                {
                    "param_dtype": str(buffer.param_dtype),
                    "grad_dtype": str(buffer.grad_dtype),
                    "parameters": [(tuple(param.shape), tuple(indices)) for param, indices in buffer.param_index_map.items()],
                    "buckets": [(bucket.grad_data.numel(), bucket.numel_unpadded) for bucket in buffer.buckets],
                }
            )
        layouts.append({"class": f"{type(child).__module__}.{type(child).__qualname__}", "distributed": hasattr(child, "get_parameter_state_dp_zero"), "buffers": buffers})
    return layouts


def restore_optimizer(optimizer, state):
    if state["optimizer_layout"] != optimizer_layout(optimizer):
        raise RuntimeError("LoRA checkpoint optimizer layout differs from the current optimizer")
    children = _optimizers(optimizer)
    if len(state["parameter_states"]) != len(children):
        raise RuntimeError("Incomplete LoRA optimizer parameter states")
    error = None
    try:
        optimizer.load_state_dict(state["optimizer"])
    except Exception as exc:
        error = str(exc)
    errors = _gather(error)
    if any(errors):
        raise RuntimeError(f"Cannot restore LoRA optimizer metadata: {errors}")
    for child, payload in zip(children, state["parameter_states"], strict=True):
        if hasattr(child, "load_parameter_state_from_dp_zero") and not getattr(child, "is_stub_optimizer", False):
            child.load_parameter_state_from_dp_zero(payload)


def begin_checkpoint(path):
    """Mark the directory as new format before writing any resumable shard."""
    if any(_gather((path / _COMPLETE).exists())):
        raise RuntimeError(f"Refusing to overwrite a completed LoRA checkpoint: {path}")
    if _rank() == 0:
        _json_save({"version": _FORMAT}, path / _MARKER)
    if dist.is_initialized():
        dist.barrier()


def _digest(path):
    with path.open("rb") as stream:
        digest = hashlib.sha256()
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
        return digest.hexdigest()


def complete_checkpoint(path, *, iteration, training, configuration=None):
    rank = _rank()
    names = [f"adapter_megatron_rank{rank}.pt"]
    if training:
        names.append(f"training_state_rank{rank}.pt")
    record = {"topology": _topology(), "files": {name: _digest(path / name) for name in names}}
    records = _gather(record)
    if rank == 0:
        _json_save({"version": _FORMAT, "iteration": iteration, "training": training, "configuration": configuration, "ranks": records}, path / _COMPLETE)
    if dist.is_initialized():
        dist.barrier()


def validate_checkpoint(path, *, training, configuration=None):
    """Validate collectively before any optimizer scatter or model mutation."""
    manifest = None
    error = None
    try:
        if (path / _MARKER).exists() or (path / _COMPLETE).exists():
            manifest = json.loads((path / _COMPLETE).read_text())
            if manifest.get("configuration") != configuration:
                raise ValueError("LoRA checkpoint base model, adapter, or training configuration changed")
            if manifest["version"] != _FORMAT:
                raise ValueError("Unsupported LoRA checkpoint version")
            if len(manifest["ranks"]) != _topology()["world_size"]:
                raise ValueError("LoRA checkpoint world size changed")
            record = manifest["ranks"][_rank()]
            if record["topology"] != _topology():
                raise ValueError("LoRA checkpoint topology changed")
            if training and not manifest["training"]:
                raise ValueError("LoRA checkpoint has no training state")
            required = {f"adapter_megatron_rank{_rank()}.pt"}
            if manifest["training"]:
                required.add(f"training_state_rank{_rank()}.pt")
            if set(record["files"]) != required:
                raise ValueError("Incomplete LoRA checkpoint file manifest")
            for name, digest in record["files"].items():
                if _digest(path / name) != digest:
                    raise ValueError(f"LoRA checkpoint checksum mismatch: {name}")
    except Exception as exc:
        error = str(exc)
    errors = _gather(error)
    if any(errors):
        raise RuntimeError(f"Invalid or incomplete LoRA checkpoint: {errors}")
    return manifest


def validate_adapter(model, tensors):
    # Imported lazily to avoid the public utility module's dependency cycle.
    from miles.backends.megatron_utils.lora_utils import _is_adapter_param_name

    parameters = [(name, param) for chunk in model for name, param in chunk.named_parameters() if _is_adapter_param_name(name)]
    names = [name for name, _ in parameters]
    if len(names) != len(set(names)):
        raise RuntimeError("LoRA checkpoint does not support duplicate adapter names across model chunks")
    if not names or set(names) != set(tensors):
        raise RuntimeError("LoRA checkpoint adapter parameter names differ from the model")
    for name, param in parameters:
        if param.shape != tensors[name].shape or param.dtype != tensors[name].dtype:
            raise RuntimeError(f"LoRA checkpoint adapter shape/dtype mismatch: {name}")


def preflight_training_state(path, manifest, optimizer, scheduler):
    error = None
    try:
        state = torch.load(path / f"training_state_rank{_rank()}.pt", map_location="cpu", weights_only=False)
        if state["version"] != _FORMAT or state["iteration"] != manifest["iteration"]:
            raise ValueError("LoRA checkpoint training version/iteration mismatch")
        if state["optimizer_layout"] != optimizer_layout(optimizer):
            raise ValueError("LoRA checkpoint optimizer layout differs")
        children = _optimizers(optimizer)
        if len(state["parameter_states"]) != len(children):
            raise ValueError("Incomplete optimizer parameter states")
        for child, payload in zip(children, state["parameter_states"], strict=True):
            if hasattr(child, "get_parameter_state_dp_zero") and not getattr(child, "is_stub_optimizer", False):
                _validate_parameter_payload(child, payload)
        if scheduler is not None and state["opt_param_scheduler"] is None:
            raise ValueError("Missing scheduler state")
        if set(state["rng"]) != {"python", "numpy", "torch", "cuda", "megatron"}:
            raise ValueError("Incomplete RNG state")
    except Exception as exc:
        error = str(exc)
    errors = _gather(error)
    if any(errors):
        raise RuntimeError(f"Invalid LoRA training state: {errors}")


def load_validated_adapter(path, model):
    tensors = None
    error = None
    try:
        tensors = torch.load(path, map_location="cpu", weights_only=True)
        validate_adapter(model, tensors)
    except Exception as exc:
        error = str(exc)
    errors = _gather(error)
    if any(errors):
        raise RuntimeError(f"Invalid LoRA adapter state: {errors}")
    return tensors


def _validate_parameter_payload(optimizer, payload):
    """Check the installed Megatron coalesced DP-zero schema before scatter."""
    if optimizer.data_parallel_group.rank() != 0:
        if payload is not None:
            raise ValueError("Unexpected optimizer parameter payload on a nonzero DP rank")
        return
    if payload is None:
        raise ValueError("Missing optimizer master parameters and moments")
    if payload.get("buckets_coalesced") is not True:
        raise ValueError("Unsupported optimizer parameter state format")
    if set(payload) != {"buckets_coalesced", *range(len(optimizer.gbuf_ranges))}:
        raise ValueError("Incomplete optimizer parameter buffers")
    for index, ranges in enumerate(optimizer.gbuf_ranges):
        if set(payload[index]) != set(ranges):
            raise ValueError("Optimizer parameter buffer dtypes differ")
        for dtype in ranges:
            tensors = payload[index][dtype]
            size = optimizer.buffers[index].numel_unpadded
            if tensors["numel_unpadded"] != size:
                raise ValueError("Optimizer parameter buffer size differs")
            for key in ("param", "exp_avg", "exp_avg_sq"):
                tensor = tensors[key]
                if not isinstance(tensor, torch.Tensor) or tensor.shape != (size,) or tensor.dtype != torch.float32:
                    raise ValueError(f"Invalid optimizer parameter tensor: {key}")


def restore_pending_rng(optimizer):
    """Replay the saved streams once after actor initialization and weight sync."""
    state = getattr(optimizer, "_lora_checkpoint_rng_state", None)
    if state is not None:
        restore_rng(state)
        del optimizer._lora_checkpoint_rng_state


def advance_scheduler_after_load(scheduler, optimizer, *, iteration, global_batch_size, use_checkpoint_scheduler):
    restored = getattr(optimizer, "_lora_checkpoint_scheduler_restored", False)
    if scheduler is not None and not restored and not (use_checkpoint_scheduler and iteration > 0):
        scheduler.step(increment=iteration * global_batch_size)


def checkpoint_configuration(args):
    """Pin local base metadata and resume-sensitive adapter/training settings.

    The base weights remain frozen and external. Config/index hashes identify
    their layout; operators must preserve the referenced weight files unchanged.
    """
    if args is None:
        return None
    names = (
        "hf_checkpoint",
        "lora_type",
        "lora_rank",
        "lora_alpha",
        "lora_dropout",
        "target_modules",
        "exclude_modules",
        "experts_shared_outer_loras",
        "optimizer",
        "lr",
        "min_lr",
        "lr_decay_style",
        "lr_decay_iters",
        "lr_warmup_iters",
        "lr_warmup_fraction",
        "lr_warmup_init",
        "adam_beta1",
        "adam_beta2",
        "adam_eps",
        "weight_decay",
        "start_weight_decay",
        "end_weight_decay",
        "weight_decay_incr_style",
        "clip_grad",
        "bf16",
        "fp16",
        "global_batch_size",
        "num_steps_per_rollout",
        "rollout_batch_size",
        "n_samples_per_prompt",
        "ptd_coef",
        "loss_type",
        "calculate_per_token_loss",
        "use_distributed_optimizer",
    )
    values = {name: getattr(args, name, None) for name in names}
    base = Path(getattr(args, "hf_checkpoint", "") or "")
    values["base_metadata"] = {
        name: _digest(base / name)
        for name in (
            "config.json",
            "model.safetensors.index.json",
            "pytorch_model.bin.index.json",
        )
        if (base / name).is_file()
    }
    return json.loads(json.dumps(values, sort_keys=True))
