"""Bounded same-batch replay for validating exact PTD checkpoint recovery."""

from pathlib import Path

import torch

from miles.utils.types import Sample


def _equal(left, right):
    if torch.is_tensor(left) or torch.is_tensor(right):
        return torch.is_tensor(left) and torch.is_tensor(right) and torch.equal(left, right)
    return left == right


def _identity(sample: Sample):
    metadata = sample.metadata or {}
    return {
        "index": sample.index,
        "group_index": sample.group_index,
        "prompt": sample.prompt,
        "label": sample.label,
        "task_id": metadata.get("task_id"),
        "deferred_media_refs": metadata.get("_deferred_media_refs"),
    }


def _validate_configuration(args, data_source) -> None:
    if not getattr(args, "ptd_exact_checkpoints", False) or getattr(args, "ptd_coef", 0) <= 0:
        raise ValueError("--ptd-replay-rollout-data requires exact PTD checkpoints")
    if not getattr(args, "rollout_global_dataset", False):
        raise ValueError("PTD replay requires the global rollout dataset")
    if getattr(args, "load_debug_rollout_data", None) is not None:
        raise ValueError("PTD replay cannot be combined with debug-only rollout loading")
    if getattr(args, "fully_async", False) or getattr(args, "partial_rollout", False):
        raise ValueError("PTD replay supports only synchronous, complete rollouts")
    if getattr(args, "dynamic_sampling_filter_path", None) is not None:
        raise ValueError("PTD replay does not support dynamic sampling filters")
    if getattr(args, "over_sampling_batch_size", args.rollout_batch_size) != args.rollout_batch_size:
        raise ValueError("PTD replay requires over-sampling batch size to equal rollout batch size")
    pending = data_source.get_buffer_length()
    if pending not in (None, 0):
        raise ValueError("PTD replay requires an empty rollout buffer")


def load_ptd_replay_rollout(args, data_source, rollout_id: int):
    """Load a trusted postprocessed batch and advance the real source exactly once.

    The saved samples retain their sampled tokens, rollout probabilities, online
    hints and teacher contexts. The freshly read prompt batch is used only to
    prove that restoring the dataset cursor selects the same inputs.
    """
    _validate_configuration(args, data_source)
    path = Path(args.ptd_replay_rollout_data.format(rollout_id=rollout_id))
    payload = torch.load(path, map_location="cpu", weights_only=False)
    if not isinstance(payload, dict) or payload.get("rollout_id") != rollout_id:
        raise ValueError("PTD replay dump rollout ID does not match the requested rollout")
    raw_samples = payload.get("samples")
    if not isinstance(raw_samples, list):
        raise ValueError("PTD replay dump has no sample list")
    replayed = [Sample.from_dict(sample) for sample in raw_samples]
    teacher_url = args.ptd_teacher_url or f"http://{args.sglang_router_ip}:{args.sglang_router_port}/generate"
    for sample in replayed:
        if sample.ptd_teacher_context is not None:
            # The URL names a process, not semantic rollout data. A fresh
            # recovery process must score against its own frozen-base fleet.
            sample.ptd_teacher_context["url"] = teacher_url
    expected_count = args.rollout_batch_size * args.n_samples_per_prompt
    if len(replayed) != expected_count:
        raise ValueError(f"PTD replay expected {expected_count} saved samples, found {len(replayed)}")

    fresh_groups = data_source.get_samples(args.rollout_batch_size)
    fresh = [sample for group in fresh_groups for sample in group]
    if len(fresh) != expected_count:
        raise ValueError(f"PTD replay data source produced {len(fresh)} samples, expected {expected_count}")
    fresh_by_index = {sample.index: sample for sample in fresh}
    replay_indices = [sample.index for sample in replayed]
    if len(fresh_by_index) != expected_count or set(fresh_by_index) != set(replay_indices):
        raise ValueError("PTD replay sample indices do not match the restored dataset cursor")
    for replayed_sample in replayed:
        actual = _identity(replayed_sample)
        expected = _identity(fresh_by_index[replayed_sample.index])
        if actual.keys() != expected.keys() or any(not _equal(actual[key], expected[key]) for key in actual):
            raise ValueError(f"PTD replay input identity differs at sample index {replayed_sample.index}")
    return replayed, payload.get("metadata") or {}
