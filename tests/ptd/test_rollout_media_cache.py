"""Prepared media transport must retain pixels and survive concurrent publishers."""

from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import pytest
from PIL import Image

from miles.utils.rollout_media_cache import cache_prepared_image, prepared_image_paths


def test_identity_order_and_no_reencoding(tmp_path, monkeypatch):
    first = Image.new("RGB", (160, 160), (13, 71, 29))
    second = Image.new("RGB", (160, 160), (13, 71, 30))
    original = first.tobytes()
    paths = prepared_image_paths([first, second, first], tmp_path)
    assert paths[0] == paths[2] != paths[1]
    assert all(Path(path).is_absolute() for path in paths)
    with Image.open(paths[0]) as cached:
        assert cached.mode == "RGB"
        assert cached.tobytes() == original
    assert cache_prepared_image(first.resize((80, 320)), tmp_path) != paths[0]
    monkeypatch.setattr(Image.Image, "save", lambda *args, **kwargs: pytest.fail("cache hit re-encoded PNG"))
    assert prepared_image_paths([first, second, first], tmp_path) == paths
    assert first.tobytes() == original


def test_corruption_fails_closed(tmp_path):
    image = Image.new("RGB", (32, 32))
    path = Path(cache_prepared_image(image, tmp_path))
    path.chmod(0o644)
    path.write_bytes(b"broken PNG")
    with pytest.raises(ValueError, match="Invalid immutable"):
        cache_prepared_image(image, tmp_path)
    assert path.read_bytes() == b"broken PNG"


def test_wrong_pixels_fail_closed(tmp_path):
    image = Image.new("RGB", (32, 32))
    path = Path(cache_prepared_image(image, tmp_path))
    path.chmod(0o644)
    Image.new("RGB", (32, 32), "red").save(path)
    with pytest.raises(ValueError, match="Invalid immutable"):
        cache_prepared_image(image, tmp_path)


def test_concurrent_publication(tmp_path):
    with ThreadPoolExecutor(max_workers=12) as pool:
        paths = list(pool.map(lambda _: cache_prepared_image(Image.new("RGB", (320, 160), "blue"), tmp_path), range(36)))
    assert len(set(paths)) == 1
    with Image.open(paths[0]) as cached:
        assert cached.getpixel((0, 0)) == (0, 0, 255)
    assert list(tmp_path.rglob(".image-*")) == []


def test_rgb_conversion_does_not_mutate_input(tmp_path):
    image = Image.new("RGBA", (32, 32), (13, 71, 29, 100))
    before = image.tobytes()
    path = cache_prepared_image(image, tmp_path)
    assert image.mode == "RGBA" and image.tobytes() == before
    with Image.open(path) as cached:
        assert cached.mode == "RGB" and cached.getpixel((0, 0)) == (13, 71, 29)


def test_qwen_resize_boundary_preserves_training_tensors(tmp_path):
    import torch
    from qwen_vl_utils.vision_process import fetch_image
    from transformers import Qwen2VLImageProcessor

    # The failed page's 172x166 boundary: qwen_vl_utils prepares 160x160,
    # while direct HF processing of the raw page chooses a different grid.
    raw = Image.frombytes("RGB", (172, 166), bytes((i * 37) % 256 for i in range(172 * 166 * 3)))
    prepared = fetch_image({"image": raw}, image_patch_size=16)
    processor = Qwen2VLImageProcessor(patch_size=16, merge_size=2, size={"shortest_edge": 65536, "longest_edge": 16777216})
    training = processor(images=[prepared], return_tensors="pt")
    direct = processor(images=[raw], return_tensors="pt")
    path = cache_prepared_image(prepared, tmp_path)
    with Image.open(path) as cached:
        replay = processor(images=[cached.convert("RGB")], return_tensors="pt")
    assert prepared.size == (160, 160)
    assert training["image_grid_thw"].tolist() == [[1, 16, 16]]
    assert direct["image_grid_thw"].prod().item() // 4 == 72
    assert not torch.equal(direct["image_grid_thw"], training["image_grid_thw"])
    for key in ("image_grid_thw", "pixel_values"):
        assert torch.equal(training[key], replay[key])
