from types import SimpleNamespace

import pytest
import torch

from miles.ray.rollout.ptd_replay import load_ptd_replay_rollout
from miles.utils.types import Sample


def _sample(index, *, task="task", prompt=None):
    return Sample(
        index=index,
        group_index=index // 2,
        prompt=[1, index] if prompt is None else prompt,
        label="answer",
        metadata={"task_id": task, "_deferred_media_refs": {"image": [f"{index // 2}.png"]}},
    )


class Source:
    def __init__(self, samples, pending=0):
        self.samples = samples
        self.pending = pending
        self.calls = 0

    def get_buffer_length(self):
        return self.pending

    def get_samples(self, count):
        self.calls += 1
        assert count == 2
        return [self.samples[:2], self.samples[2:]]


def _args(path):
    return SimpleNamespace(
        ptd_exact_checkpoints=True,
        ptd_coef=0.05,
        rollout_global_dataset=True,
        load_debug_rollout_data=None,
        fully_async=False,
        partial_rollout=False,
        dynamic_sampling_filter_path=None,
        over_sampling_batch_size=2,
        rollout_batch_size=2,
        n_samples_per_prompt=2,
        ptd_replay_rollout_data=str(path),
        ptd_teacher_url=None,
        sglang_router_ip="10.0.0.2",
        sglang_router_port=18080,
    )


def _save(path, samples, rollout_id=1):
    torch.save({"rollout_id": rollout_id, "metadata": {"source": "baseline"},
                "samples": [sample.to_dict() for sample in samples]}, path)


def test_replay_preserves_saved_batch_and_advances_source(tmp_path):
    fresh = [_sample(index) for index in range(4, 8)]
    replayed = list(reversed([_sample(index) for index in range(4, 8)]))
    replayed[0].tokens = [10, 11, 12]
    replayed[0].ptd_teacher_context = {
        "url": "http://stale-router:10000/generate",
        "payload": {"input_ids": [10, 11, 12]},
    }
    path = tmp_path / "rollout-1.pt"
    _save(path, replayed)
    source = Source(fresh)

    loaded, metadata = load_ptd_replay_rollout(_args(path), source, 1)

    assert [sample.index for sample in loaded] == [sample.index for sample in replayed]
    assert loaded[0].tokens == [10, 11, 12]
    assert loaded[0].ptd_teacher_context["url"] == "http://10.0.0.2:18080/generate"
    assert loaded[0].ptd_teacher_context["payload"] == replayed[0].ptd_teacher_context["payload"]
    assert metadata == {"source": "baseline"}
    assert source.calls == 1


@pytest.mark.parametrize(
    ("change", "message"),
    [
        (lambda args, source: setattr(args, "partial_rollout", True), "synchronous"),
        (lambda args, source: setattr(args, "over_sampling_batch_size", 3), "over-sampling"),
        (lambda args, source: setattr(source, "pending", 1), "empty rollout buffer"),
    ],
)
def test_replay_rejects_unsupported_source_modes(tmp_path, change, message):
    samples = [_sample(index) for index in range(4, 8)]
    path = tmp_path / "rollout-1.pt"
    _save(path, samples)
    args, source = _args(path), Source(samples)
    change(args, source)
    with pytest.raises(ValueError, match=message):
        load_ptd_replay_rollout(args, source, 1)
    assert source.calls == 0


@pytest.mark.parametrize("mutation", ["rollout", "prompt", "task", "count"])
def test_replay_rejects_wrong_dump_or_cursor(tmp_path, mutation):
    saved = [_sample(index) for index in range(4, 8)]
    fresh = [_sample(index) for index in range(4, 8)]
    rollout_id = 0 if mutation == "rollout" else 1
    if mutation == "prompt":
        fresh[0].prompt = [999]
    if mutation == "task":
        fresh[0].metadata["task_id"] = "different"
    if mutation == "count":
        saved.pop()
    path = tmp_path / "rollout-1.pt"
    _save(path, saved, rollout_id=rollout_id)
    with pytest.raises(ValueError):
        load_ptd_replay_rollout(_args(path), Source(fresh), 1)
