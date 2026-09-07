import pytest
import torch

from miles.backends.megatron_utils.vision_update_audit import (
    capture_vision_update,
    verify_vision_sync_checksums,
    verify_vision_update,
)


class TinyVLM(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.vision_model = torch.nn.Module()
        self.vision_model.encoder = torch.nn.Linear(4, 4, bias=False)
        self.vision_model.merger = torch.nn.Linear(4, 4, bias=False)


def _ready_model():
    model = TinyVLM()
    for param in model.parameters():
        param.main_grad = torch.ones_like(param)
    optimizer = torch.optim.SGD(model.parameters(), lr=0.1)
    return model, optimizer


def test_audit_requires_cache_disabled(monkeypatch):
    model, optimizer = _ready_model()
    monkeypatch.delenv("SGLANG_VLM_CACHE_SIZE_MB", raising=False)
    with pytest.raises(RuntimeError, match="cached vision embeddings"):
        capture_vision_update([model], optimizer)


def test_audit_observes_encoder_and_projection_updates(monkeypatch):
    model, optimizer = _ready_model()
    monkeypatch.setenv("SGLANG_VLM_CACHE_SIZE_MB", "0")
    snapshot = capture_vision_update([model], optimizer)
    for param in model.parameters():
        param.grad = param.main_grad
    optimizer.step()
    verify_vision_update(snapshot, rollout_id=0, step_id=0)


def test_audit_rejects_frozen_projection(monkeypatch):
    model, optimizer = _ready_model()
    monkeypatch.setenv("SGLANG_VLM_CACHE_SIZE_MB", "0")
    model.vision_model.merger.weight.requires_grad_(False)
    with pytest.raises(RuntimeError, match="frozen parameters"):
        capture_vision_update([model], optimizer)


def test_sync_audit_requires_changed_exact_vision_and_projection():
    first = {"model.visual.patch.weight": "a", "model.visual.merger.weight": "b"}
    second = {"model.visual.patch.weight": "c", "model.visual.merger.weight": "d"}
    body = {"success": True, "ranks": [{"checksums": second}]}
    verify_vision_sync_checksums(second, [body, body], previous=first, require_change=True)


def test_sync_audit_maps_hf_visual_wrapper_to_sglang_names():
    first = {
        "model.visual.patch.weight": "a",
        "model.visual.blocks.0.attn.qkv.weight": "b",
        "model.visual.merger.weight": "c",
    }
    second = {
        "model.visual.patch.weight": "d",
        "model.visual.blocks.0.attn.qkv.weight": "e",
        "model.visual.merger.weight": "f",
    }
    body = {
        "success": True,
        "ranks": [
            {
                "checksums": {
                    "visual.patch.weight": "d",
                    "visual.blocks.0.attn.qkv_proj.weight": "e",
                    "visual.merger.weight": "f",
                }
            }
        ],
    }
    verify_vision_sync_checksums(second, [body], previous=first, require_change=True)


@pytest.mark.parametrize("failure", ["stale", "missing", "mismatch"])
def test_sync_audit_rejects_invalid_rollout_vision(failure):
    first = {"model.visual.patch.weight": "a", "model.visual.merger.weight": "b"}
    second = {"model.visual.patch.weight": "c", "model.visual.merger.weight": "d"}
    actual = dict(second)
    previous = first
    if failure == "stale":
        second = first
    elif failure == "missing":
        actual.pop("model.visual.merger.weight")
    else:
        actual["model.visual.patch.weight"] = "wrong"
    body = {"success": True, "ranks": [{"checksums": actual}]}
    with pytest.raises(RuntimeError):
        verify_vision_sync_checksums(second, [body], previous=previous, require_change=True)
