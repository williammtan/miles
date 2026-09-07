"""Runtime proof that full-parameter VLM training updates rollout-visible vision weights."""

from __future__ import annotations

import logging
import os
from collections.abc import Sequence
from dataclasses import dataclass

import torch

logger = logging.getLogger(__name__)

_VISION_MARKER = ".vision_model."
_PROJECTION_MARKERS = (".vision_model.merger.", ".vision_model.decoder.deepstack_merger_list.")
_MAX_SAMPLES_PER_TENSOR = 4096


def _category(name: str) -> str | None:
    qualified = f".{name}."
    if _VISION_MARKER not in qualified:
        return None
    if any(marker in qualified for marker in _PROJECTION_MARKERS):
        return "projection"
    return "encoder"


def _gradient(param: torch.nn.Parameter) -> torch.Tensor | None:
    grad = getattr(param, "main_grad", None)
    return param.grad if grad is None else grad


def _sample_indices(numel: int, *, device: torch.device) -> torch.Tensor:
    count = min(numel, _MAX_SAMPLES_PER_TENSOR)
    if count == 1:
        return torch.zeros(1, device=device, dtype=torch.int64)
    if count == numel:
        return torch.arange(numel, device=device)
    # Integer arithmetic includes both ends without constructing a full-size tensor.
    return torch.div(
        torch.arange(count, device=device, dtype=torch.int64) * (numel - 1),
        count - 1,
        rounding_mode="floor",
    )


def _optimizer_tensor_ids(optimizer) -> set[int]:
    """Collect direct/master tensors owned by this DP rank's optimizer."""

    pending = list(getattr(optimizer, "chained_optimizers", [optimizer]))
    tensor_ids: set[int] = set()
    while pending:
        current = pending.pop()
        children = getattr(current, "chained_optimizers", None)
        if children is not None:
            pending.extend(children)
            continue
        inner = getattr(current, "optimizer", current)
        for group in getattr(inner, "param_groups", ()):  # torch optimizer or HDO wrapper
            tensor_ids.update(id(param) for param in group.get("params", ()))
        for mapping_name in ("param_to_fp32_param", "gpu_params_map_cpu_copy"):
            mapping = getattr(inner, mapping_name, None)
            if mapping:
                tensor_ids.update(id(param) for param in mapping)
                tensor_ids.update(id(param) for param in mapping.values())
    return tensor_ids


@dataclass
class _Sample:
    name: str
    category: str
    parameter: torch.nn.Parameter
    indices: torch.Tensor
    before: torch.Tensor


@dataclass
class VisionUpdateSnapshot:
    samples: list[_Sample]
    tensor_counts: dict[str, int]
    element_counts: dict[str, int]
    max_gradients: dict[str, float]


@torch.no_grad()
def capture_vision_update(model: Sequence[torch.nn.Module], optimizer) -> VisionUpdateSnapshot:
    """Validate vision trainability/gradients immediately before ``optimizer.step``."""

    cache_size = os.environ.get("SGLANG_VLM_CACHE_SIZE_MB")
    if cache_size != "0":
        raise RuntimeError(
            "--audit-vision-updates requires SGLANG_VLM_CACHE_SIZE_MB=0; cached vision embeddings "
            f"would make rollout outputs stale after FFT updates (got {cache_size!r})"
        )

    optimizer_ids = _optimizer_tensor_ids(optimizer)
    samples: list[_Sample] = []
    tensor_counts = {"encoder": 0, "projection": 0}
    element_counts = {"encoder": 0, "projection": 0}
    max_gradient_tensors = {"encoder": [], "projection": []}
    frozen: list[str] = []
    uncovered: list[str] = []

    for chunk_index, chunk in enumerate(model):
        for raw_name, param in chunk.named_parameters():
            name = f"pp{chunk_index}.{raw_name}"
            category = _category(name)
            if category is None:
                continue
            tensor_counts[category] += 1
            element_counts[category] += param.numel()
            if not param.requires_grad:
                frozen.append(name)
                continue

            main_param = getattr(param, "main_param", None)
            optimizer_owns = id(param) in optimizer_ids or (main_param is not None and id(main_param) in optimizer_ids)
            if not optimizer_owns and not getattr(param, "main_param_sharded", False):
                uncovered.append(name)

            grad = _gradient(param)
            if grad is None:
                max_gradient_tensors[category].append(torch.zeros((), device=param.device, dtype=torch.float32))
                continue
            max_gradient_tensors[category].append(grad.detach().abs().amax().float())
            indices = _sample_indices(param.numel(), device=param.device)
            before = param.detach().reshape(-1).index_select(0, indices).clone()
            samples.append(_Sample(name, category, param, indices, before))

    missing = [name for name, count in tensor_counts.items() if count == 0]
    if missing:
        raise RuntimeError(f"vision update audit found no parameters for categories: {missing}")
    if frozen:
        raise RuntimeError(f"vision update audit found frozen parameters: {frozen[:8]}")
    if uncovered:
        raise RuntimeError(f"vision update audit found parameters absent from the optimizer: {uncovered[:8]}")

    max_gradients = {}
    for category, values in max_gradient_tensors.items():
        maximum = torch.stack(values).amax().item() if values else 0.0
        max_gradients[category] = maximum
        if not torch.isfinite(torch.tensor(maximum)) or maximum <= 0:
            raise RuntimeError(f"vision update audit found no finite nonzero {category} gradient")

    return VisionUpdateSnapshot(samples, tensor_counts, element_counts, max_gradients)


@torch.no_grad()
def verify_vision_update(snapshot: VisionUpdateSnapshot, *, rollout_id: int, step_id: int) -> None:
    """Require sampled BF16/FP16 model values to change after ``optimizer.step``."""

    changed = {"encoder": 0, "projection": 0}
    checked = {"encoder": 0, "projection": 0}
    for sample in snapshot.samples:
        after = sample.parameter.detach().reshape(-1).index_select(0, sample.indices)
        checked[sample.category] += after.numel()
        changed[sample.category] += torch.count_nonzero(after != sample.before).item()

    unchanged = [category for category, count in changed.items() if count == 0]
    if unchanged:
        raise RuntimeError(
            "vision update audit found no rollout-visible low-precision weight change after optimizer.step "
            f"for categories {unchanged}; sampled={checked}, max_gradients={snapshot.max_gradients}"
        )

    logger.info(
        "vision_update_audit rollout=%d step=%d tensors=%s elements=%s sampled=%s changed=%s max_gradients=%s",
        rollout_id,
        step_id,
        snapshot.tensor_counts,
        snapshot.element_counts,
        checked,
        changed,
        snapshot.max_gradients,
    )


def verify_vision_sync_checksums(
    expected: dict[str, str],
    engine_bodies: list[dict],
    *,
    previous: dict[str, str] | None,
    require_change: bool,
) -> None:
    """Compare exported vision tensors with each TP=1 rollout engine."""

    if not expected:
        raise RuntimeError("vision weight sync audit saw no exported vision tensors")
    if not any("merger" in name for name in expected):
        raise RuntimeError("vision weight sync audit saw no exported vision projection tensors")
    if require_change and previous is not None:
        changed = {name for name, digest in expected.items() if previous.get(name) != digest}
        if not changed:
            raise RuntimeError("vision weight sync audit found no changed exported vision tensors")
        if not any("merger" in name for name in changed):
            raise RuntimeError("vision weight sync audit found no changed exported vision projection tensors")

    for engine_index, body in enumerate(engine_bodies):
        if body is None:
            continue
        if not body.get("success", False):
            raise RuntimeError(f"rollout engine {engine_index} failed its vision checksum request")
        ranks = body.get("ranks") or []
        if len(ranks) != 1:
            raise RuntimeError(
                f"vision weight sync audit requires TP=1 rollout engines; engine {engine_index} returned {len(ranks)} ranks"
            )
        actual = ranks[0].get("checksums") or {}
        # The HF iterator emits Qwen VLM keys such as
        # ``model.visual.blocks.0...``. SGLang registers the same module at the
        # top level and therefore reports ``visual.blocks.0...`` from
        # ``named_parameters()``. Match only this known wrapper-prefix
        # difference and still require an exact checksum for every tensor.
        resolved = {}
        ambiguous = []
        for name in expected:
            candidates = [name]
            if name.startswith("model.visual.") or name.startswith("model.vision_model."):
                sglang_name = name.removeprefix("model.")
                # Qwen3VLForConditionalGeneration.hf_to_sglang_mapper also
                # renames the vision attention's packed projection.
                sglang_name = sglang_name.replace(".attn.qkv.", ".attn.qkv_proj.")
                candidates.append(sglang_name)
            matches = [candidate for candidate in candidates if candidate in actual]
            if len(matches) > 1:
                ambiguous.append(name)
            elif matches:
                resolved[name] = matches[0]
        missing = sorted(set(expected) - set(resolved))
        mismatched = sorted(
            name for name, digest in expected.items() if name in resolved and actual[resolved[name]] != digest
        )
        if missing or mismatched or ambiguous:
            raise RuntimeError(
                f"rollout engine {engine_index} vision weights differ after sync: "
                f"missing={missing[:8]}, mismatched={mismatched[:8]}, ambiguous={ambiguous[:8]}"
            )
