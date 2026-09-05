"""Run the patched deployed Qwen method with CPU image-processing doubles."""

import asyncio
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

from tools.patch_sglang_ptd import MM_MARKER, patch_mm_source


class _ProcessorOutput(SimpleNamespace):
    @staticmethod
    def build_padded_input_ids(ids, items):
        padded = list(ids)
        for item in items:
            start, end = item.offsets[0]
            padded[start:end + 1] = [9999] * (end - start + 1)
        return padded


def _processor(original, canonical, offsets):
    source = (Path(__file__).parent / "fixtures/qwen_mm_process.py.txt").read_text()
    namespace = {
        "torch": torch, "time": SimpleNamespace(perf_counter=lambda: 0),
        "_get_processor_video_config": lambda *args: None,
        "MultimodalProcessorOutput": _ProcessorOutput,
        "Modality": SimpleNamespace(IMAGE="image", VIDEO="video"),
        "logger": SimpleNamespace(debug=lambda *args: None),
    }
    exec(compile(patch_mm_source(source), "patched_qwen_mm_process.py", "exec"), namespace)
    processor = namespace["QwenVLImageProcessor"]()
    items = [SimpleNamespace(offsets=[span], is_image=lambda: True, feature=torch.tensor([i]))
             for i, span in enumerate(offsets)]
    old_ret = {"padded_input_ids": [-9] * len(canonical)}

    async def load(**kwargs):
        assert kwargs["prompt"] == original
        return SimpleNamespace(videos=[], input_ids=None)

    async def process(*args, **kwargs):
        return items, torch.tensor(canonical), old_ret

    rope_calls = []

    def rope(**kwargs):
        rope_calls.append(kwargs)
        return torch.arange(kwargs["input_len"]).expand(3, -1), torch.zeros((1, 1), dtype=torch.long)

    processor.legacy_load_mm_data = load
    processor.process_and_combine_mm_data_async = process
    processor._mark_dp_encoder_features_for_deferred_reconstruction = lambda _: None
    processor._compute_image_only_mrope_positions_from_offsets = rope
    processor._get_precomputed_mrope_from_output = lambda _: (torch.full((3, len(canonical)), -9), torch.tensor([[-9]]))
    processor._get_processor_output_value = lambda ret, key: ret.get(key)
    processor._get_grid_from_output_or_items = lambda *args: None
    processor.hf_config = SimpleNamespace(model_type="qwen3_5")
    processor.model_type = "qwen3_5"
    processor.mm_tokens = SimpleNamespace(image_token_id=90, video_token_id=91, audio_token_id=92)
    processor.vision_start_token_id = 88
    processor.vision_end_token_id = 89
    processor.video_config = None
    return processor, items, rope_calls


@pytest.mark.parametrize("original,canonical,old_spans,new_spans", [
    # Actual sampled pair from smoke4 sample9: equal-length retokenization drift.
    ([1, 90, 90, 2, 469, 26000, 3], [1, 90, 90, 2, 14944, 334, 3], [(1, 2)], [(1, 2)]),
    # A changed response length must also change padding and positional tensors.
    ([1, 90, 90, 2, 5, 6, 7], [1, 90, 90, 2, 8, 7], [(1, 2)], [(1, 2)]),
    # Noncanonical text before multiple images shifts each image's offsets.
    ([1, 2, 90, 90, 3, 4, 90, 90, 90, 5], [8, 90, 90, 9, 90, 90, 90, 5],
     [(1, 2), (4, 6)], [(2, 3), (6, 8)]),
])
def test_patched_qwen_preserves_exact_ids_offsets_padding_and_mrope(original, canonical, old_spans, new_spans):
    processor, old_items, rope_calls = _processor(original, canonical, old_spans)
    request = SimpleNamespace(input_ids=original, token_ids_logprob_positions=[[]], video_data=None, audio_data=None)
    result = asyncio.run(processor.process_mm_data_async(["same-image"], original, request))
    assert result.input_ids == original
    assert [item.offsets for item in result.mm_items] == [[span] for span in new_spans]
    assert [item.offsets for item in old_items] == [[span] for span in old_spans]
    for new, old in zip(result.mm_items, old_items, strict=True):
        assert new is not old and new.feature is old.feature
    expected_padded = [9999 if token == 90 else token for token in original]
    assert result.padded_input_ids == expected_padded
    assert len(rope_calls) == 1 and rope_calls[0]["input_len"] == len(original)
    torch.testing.assert_close(result.mrope_positions, torch.arange(len(original)).expand(3, -1))


@pytest.mark.parametrize("original,spans", [
    ([1, 90, 2], [(1, 2)]),  # Expanded token count mismatch.
    ([1, 90, 90, 2, 90, 90], [(1, 2)]),  # Too many image spans.
    ([1, 2, 3], [(1, 2)]),  # Missing image span.
])
def test_patched_qwen_rejects_inconsistent_media_without_retokenization_fallback(original, spans):
    processor, _, _ = _processor(original, [1, 90, 90, 2], spans)
    request = SimpleNamespace(input_ids=original, token_ids_logprob_positions=[[]], video_data=None, audio_data=None)
    with pytest.raises(ValueError, match="PTD expanded image token"):
        asyncio.run(processor.process_mm_data_async(["same-image"], original, request))


def test_non_ptd_qwen_keeps_existing_processing_behavior():
    original = [1, 90, 90, 2, 469, 26000]
    canonical = [1, 90, 90, 2, 14944, 334]
    processor, items, rope_calls = _processor(original, canonical, [(1, 2)])
    request = SimpleNamespace(input_ids=original, token_ids_logprob_positions=None, video_data=None, audio_data=None)
    result = asyncio.run(processor.process_mm_data_async(["same-image"], original, request))
    assert result.input_ids == canonical
    assert result.mm_items == items
    assert result.padded_input_ids == [-9] * len(canonical)
    assert rope_calls == []


def test_multimodal_patch_is_idempotent_and_requires_known_anchors():
    source = (Path(__file__).parent / "fixtures/qwen_mm_process.py.txt").read_text()
    patched = patch_mm_source(source)
    assert MM_MARKER in patched
    assert patch_mm_source(patched) == patched
    with pytest.raises(ValueError, match="expected one patch anchor"):
        patch_mm_source(source.replace("import math\n", ""))
