"""Materialise path-carried images into multimodal training tensors on the trainer rank.

A rollout that hands the trainer image *file paths* (``train_data["media_refs"]``, one
``{"image": [path, ...]}`` per sample, taken from ``Sample.metadata["_deferred_media_refs"]``)
ships no pixel tensor through the object store. Each rank opens its shard's PNGs from the shared
filesystem and runs the same ``image_processor`` call the rollout engine ran on the same bytes,
so ``pixel_values`` / ``image_grid_thw`` match the ``<|image_pad|>`` tokens already in the prompt.
"""

import logging
import time
from concurrent.futures import ThreadPoolExecutor

import torch
from PIL import Image

from miles.utils.processing_utils import load_processor

logger = logging.getLogger(__name__)

IMAGE_PAD_TOKEN = "<|image_pad|>"
MAX_WORKERS = 16
PIXEL_DTYPE = torch.bfloat16

_processor_cache: dict[str, object] = {}


def processor_for(args):
    """One processor per rank; the vision config lives next to the checkpoint on the shared filesystem."""
    checkpoint = args.hf_checkpoint
    processor = _processor_cache.get(checkpoint)
    if processor is None:
        processor = load_processor(checkpoint, trust_remote_code=True)
        if processor is None:
            raise RuntimeError(f"media_refs in the rollout data but {checkpoint} has no image processor")
        _processor_cache[checkpoint] = processor
    return processor


def image_train_inputs(processor, paths: list[str]) -> dict[str, torch.Tensor]:
    """pixel_values (bf16) and image_grid_thw for the images at ``paths``, in order."""
    images = []
    for path in paths:
        with Image.open(path) as opened:
            images.append(opened.convert("RGB"))
    output = processor.image_processor(images=images, return_tensors="pt")
    # The vision patch embedding casts to bf16 anyway; halving here keeps the shard small.
    return {"pixel_values": output["pixel_values"].to(PIXEL_DTYPE), "image_grid_thw": output["image_grid_thw"]}


def expected_image_tokens(processor, train_inputs: dict[str, torch.Tensor] | None) -> int:
    if not train_inputs:
        return 0
    merge = getattr(processor.image_processor, "merge_size", 2)
    return int(train_inputs["image_grid_thw"].prod(dim=-1).sum().item()) // (merge * merge)


def materialize_media_refs(args, rollout_data: dict) -> None:
    """Replace ``rollout_data["media_refs"]`` with ``rollout_data["multimodal_train_inputs"]`` in place.

    Runs before the tokens move to the GPU (``tokens`` are still lists). A sample whose pad count
    disagrees with its pixels raises: training on it would misalign every image token.
    """
    refs = rollout_data.pop("media_refs", None)
    if refs is None or rollout_data.get("multimodal_train_inputs") is not None:
        return
    processor = processor_for(args)
    pad_id = processor.tokenizer.convert_tokens_to_ids(IMAGE_PAD_TOKEN)
    started = time.time()

    def one(ref):
        paths = list((ref or {}).get("image") or [])
        return image_train_inputs(processor, paths) if paths else None

    with ThreadPoolExecutor(max_workers=MAX_WORKERS) as pool:
        train_inputs = list(pool.map(one, refs))

    for index, (tokens, mm) in enumerate(zip(rollout_data["tokens"], train_inputs, strict=True)):
        pads = sum(1 for t in tokens if t == pad_id)
        expected = expected_image_tokens(processor, mm)
        if pads != expected:
            raise ValueError(
                f"sample {index}: {pads} image-pad tokens but the images at "
                f"{(refs[index] or {}).get('image')} produce {expected}; media refs and tokens disagree"
            )
    rollout_data["multimodal_train_inputs"] = train_inputs
    num_images = sum(len((ref or {}).get("image") or []) for ref in refs)
    logger.info(
        "materialized %d images for %d samples from media refs in %.1fs", num_images, len(refs), time.time() - started
    )
