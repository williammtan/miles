"""CPU preflight of every prepared image against the training processor."""

import argparse
import json
from pathlib import Path

import torch
from PIL import Image

from miles.utils.data import Dataset
from miles.utils.processing_utils import call_processor, load_processor, load_tokenizer, resolve_deferred_media
from miles.utils.rollout_media_cache import prepared_image_paths


def check_dataset(path, *, processor, tokenizer, cache_dir):
    dataset = Dataset(
        str(path), tokenizer, processor, None, prompt_key="problem",
        multimodal_keys={"image": "images"}, apply_chat_template=True,
    )
    report = {"path": str(path), "samples": 0, "images": 0, "failures": []}
    for index, sample in enumerate(dataset.samples):
        report["samples"] += 1
        try:
            refs = sample.metadata.get("_deferred_media_refs")
            media = resolve_deferred_media(refs, processor) if refs else sample.multimodal_inputs
            if not media or not media.get("images"):
                continue
            training = call_processor(processor, sample.prompt, media)
            paths = prepared_image_paths(media["images"], cache_dir)
            report["images"] += len(paths)
            replay_images = []
            for image_path in paths:
                with Image.open(image_path) as image:
                    replay_images.append(image.convert("RGB"))
            replay = processor.image_processor(images=replay_images, return_tensors="pt")
            for key in ("pixel_values", "image_grid_thw"):
                if not torch.equal(training[key], replay[key]):
                    raise ValueError(f"{key} differs between training and cached PNG")
            if paths != prepared_image_paths(replay_images, cache_dir):
                raise ValueError("Cached PNG identity is not stable")
        except Exception as exc:
            report["failures"].append({"sample": index, "error": str(exc)})
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", required=True)
    parser.add_argument("--train", type=Path, required=True)
    parser.add_argument("--eval", type=Path, required=True)
    parser.add_argument("--cache-dir", required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    tokenizer = load_tokenizer(args.model, trust_remote_code=True)
    processor = load_processor(args.model, trust_remote_code=True)
    if processor is None:
        raise ValueError("Model must provide an image processor")
    reports = [check_dataset(path, processor=processor, tokenizer=tokenizer, cache_dir=args.cache_dir)
               for path in (args.train, args.eval)]
    report = {"datasets": reports, "passed": all(item["samples"] and item["images"] and not item["failures"] for item in reports)}
    args.output.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))
    raise SystemExit(0 if report["passed"] else 1)


if __name__ == "__main__":
    main()
