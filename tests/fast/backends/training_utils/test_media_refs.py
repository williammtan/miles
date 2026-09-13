"""materialize_media_refs: PNG paths become the same tensors the processor builds, or the sample is rejected."""

from __future__ import annotations

from argparse import Namespace

import pytest
import torch
from PIL import Image

from miles.backends.training_utils import media_refs


class FakeImageProcessor:
    """One 2x2-merged patch per 32x32 block, like Qwen's processor at its factor."""

    merge_size = 2

    def __call__(self, images, return_tensors="pt"):
        grids = [torch.tensor([1, image.height // 16, image.width // 16]) for image in images]
        pixels = torch.cat([torch.full((int(g.prod()), 8), float(i), dtype=torch.float32) for i, g in enumerate(grids)])
        return {"pixel_values": pixels, "image_grid_thw": torch.stack(grids)}


class FakeTokenizer:
    def convert_tokens_to_ids(self, token):
        return {"<|image_pad|>": 7}[token]


class FakeProcessor:
    image_processor = FakeImageProcessor()
    tokenizer = FakeTokenizer()


@pytest.fixture
def pngs(tmp_path, monkeypatch):
    paths = []
    for name, size in (("a", (64, 32)), ("b", (32, 32))):
        path = tmp_path / f"{name}.png"
        Image.new("RGB", size, (10, 20, 30)).save(path)
        paths.append(str(path))
    monkeypatch.setattr(media_refs, "processor_for", lambda args: FakeProcessor())
    return paths


def test_refs_become_bf16_pixels_aligned_with_the_pads(pngs):
    a, b = pngs
    rollout_data = {
        "tokens": [[1] + [7] * 2 + [2] + [7] * 1, [1, 2, 3], [7] * 1],
        "media_refs": [{"image": [a, b]}, None, {"image": [b]}],
    }
    media_refs.materialize_media_refs(Namespace(hf_checkpoint="x"), rollout_data)
    assert "media_refs" not in rollout_data
    first, none, third = rollout_data["multimodal_train_inputs"]
    assert none is None
    assert first["pixel_values"].dtype == torch.bfloat16 and first["pixel_values"].shape == (12, 8)
    assert first["image_grid_thw"].tolist() == [[1, 2, 4], [1, 2, 2]]
    assert third["image_grid_thw"].tolist() == [[1, 2, 2]]


def test_pad_count_mismatch_rejects_the_shard(pngs):
    a, _ = pngs
    rollout_data = {"tokens": [[7] * 3], "media_refs": [{"image": [a]}]}
    with pytest.raises(ValueError, match="3 image-pad tokens"):
        media_refs.materialize_media_refs(Namespace(hf_checkpoint="x"), rollout_data)


def test_nothing_to_do_without_refs_or_with_tensors_present():
    rollout_data = {"tokens": [[7]]}
    media_refs.materialize_media_refs(Namespace(hf_checkpoint="x"), rollout_data)
    assert rollout_data == {"tokens": [[7]]}
    rollout_data = {"tokens": [[7]], "media_refs": [{"image": ["/nope.png"]}], "multimodal_train_inputs": [{"k": 1}]}
    media_refs.materialize_media_refs(Namespace(hf_checkpoint="x"), rollout_data)
    assert rollout_data["multimodal_train_inputs"] == [{"k": 1}]
