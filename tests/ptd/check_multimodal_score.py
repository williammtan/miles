"""Live scoring check against the pinned Qwen/SGLang runtime; no training update."""

import argparse
import json
import uuid
from argparse import Namespace
from pathlib import Path

import requests
from PIL import Image
from transformers import AutoProcessor

from miles.rollout.ptd import extract_score_rows, score_teacher_missing_ids, teacher_score_context
from miles.utils.processing_utils import call_processor
from miles.utils.types import Sample


def post(url, payload):
    response = requests.post(url, json=payload, timeout=600)
    response.raise_for_status()
    return response.json()


def score_maps(result, context):
    return [{int(entry[1]): float(entry[0]) for entry in row}
            for row in extract_score_rows(result, context, "input_token_ids_logprobs")]


def max_difference(left, right):
    assert len(left) == len(right)
    assert all(set(a) == set(b) for a, b in zip(left, right, strict=True))
    return max(abs(value - b[token]) for a, b in zip(left, right, strict=True)
               for token, value in a.items())


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--url", default="http://127.0.0.1:18991")
    parser.add_argument("--model", default="/weka/models/Qwen/Qwen3.8-27B")
    parser.add_argument("--dataset", default="/weka/handshake/vqa/train_le24k.jsonl")
    parser.add_argument("--adapter", default="/weka/handshake/checkpoints/qwen3p8-27b-lora/iter_0000019/adapter")
    parser.add_argument("--read-only-existing-adapter", action="store_true",
                        help="Reuse an already loaded miles_lora adapter; never load/unload or change server state")
    parser.add_argument("--output", default="/tmp/ptd-multimodal-check.json")
    args = parser.parse_args()
    with open(args.dataset) as stream:
        row = json.loads(next(stream))
    processor = AutoProcessor.from_pretrained(args.model, trust_remote_code=True)
    image_paths = row["images"]
    images = [Image.open(path).convert("RGB") for path in image_paths]
    message = [{"role": "user", "content": [*({"type": "image"} for _ in images),
                 {"type": "text", "text": row["problem"].replace("<image>", "")}]}]
    prompt = processor.apply_chat_template(message, tokenize=False, add_generation_prompt=True)
    processor_output = call_processor(processor, prompt, {"images": images})
    prompt_ids = list(processor_output["input_ids"][0])
    generated = post(args.url + "/generate", {
        "input_ids": prompt_ids, "image_data": image_paths, "lora_path": None, "return_logprob": True,
        "sampling_params": {"max_new_tokens": 8, "temperature": 0, "skip_special_tokens": False},
    })
    tokens = [int(item[1]) for item in generated["meta_info"]["output_token_logprobs"]]
    assert len(tokens) == 8, generated
    sample = Sample(tokens=prompt_ids + tokens, response_length=len(tokens), status=Sample.Status.TRUNCATED,
                    metadata={"ptd_media_payload": {"image_data": image_paths}})
    config = json.loads((Path(args.model) / "config.json").read_text())
    vocab_size = config.get("text_config", config)["vocab_size"]
    score_args = Namespace(ptd_teacher_url=args.url + "/generate", ptd_vocab_size=vocab_size)
    context = teacher_score_context(score_args, sample, processor.tokenizer,
                                    "Read the row and column labels first; compare color intensity only within the specified groups. "
                                    f"Diagnostic identity: {uuid.uuid4().hex}")
    assert context["payload"]["image_data"] == image_paths
    assert context["payload"]["input_ids"][-len(sample.tokens):] == sample.tokens
    teacher = post(context["url"], {**context["payload"], "top_logprobs_num": 4})
    top = extract_score_rows(teacher, context, "input_top_logprobs")
    selected = [[int(entries[0][1]), 42 + index] for index, entries in enumerate(top)]
    sparse = score_teacher_missing_ids(context, selected, 600)
    # Small independent global-ID request is a reference for this 8-token smoke only.
    all_ids = sorted({token for ids in selected for token in ids} | {int(e[1]) for row in top for e in row})
    reference = post(context["url"], {**context["payload"], "token_ids_logprob": all_ids})
    ref_rows = extract_score_rows(reference, context, "input_token_ids_logprobs")
    reference_maps = [{int(entry[1]): float(entry[0]) for entry in row} for row in ref_rows]
    cold_warm_error = max(abs(float(entry[0]) - ref[int(entry[1])])
                          for row, ref in zip(top, reference_maps, strict=True) for entry in row)
    max_error = 0
    for actual, entries in zip(sparse, ref_rows, strict=True):
        expected = {int(entry[1]): float(entry[0]) for entry in entries}
        max_error = max(max_error, max(abs(value - expected[token]) for token, value in actual.items()))
    assert max_error < 2e-4, max_error
    # Explicit EOS and one-token continuation exercise the boundary independently of generation length.
    eos = processor.tokenizer.eos_token_id
    one = Sample(tokens=prompt_ids + [eos], response_length=1,
                 metadata={"ptd_media_payload": {"image_data": image_paths}})
    one_context = teacher_score_context(score_args, one, processor.tokenizer, "Check image labels.")
    one_result = score_teacher_missing_ids(one_context, [[eos, 42]], 600)
    assert set(one_result[0]) == {eos, 42}
    # Compare identical token IDs, not rank-ordered Top-K values. Repeated
    # pre-update requests expose serving/cache variability independently of LoRA.
    fixed_payload = {**context["payload"], "token_ids_logprob": all_ids}
    before = [score_maps(post(context["url"], fixed_payload), context) for _ in range(3)]
    # Loading an actual trained adapter must not alter the frozen teacher route.
    loaded = {"reused_existing_adapter": True}
    if not args.read_only_existing_adapter:
        loaded = post(args.url + "/load_lora_adapter", {"lora_name": "miles_lora", "lora_path": args.adapter})
    after = score_maps(post(context["url"], fixed_payload), context)
    base_error = max_difference(before[-1], after)
    repeat_error = max(max_difference(before[0], scores) for scores in before[1:])
    student = score_maps(post(context["url"], {**fixed_payload, "lora_path": "miles_lora"}), context)
    adapter_diff = max_difference(student, after)
    passed = max_error < 2e-4 and base_error < 2e-4 and repeat_error < 2e-4 and cold_warm_error < 2e-4 and adapter_diff > 1e-5
    artifact = {"prompt_tokens": len(prompt_ids), "response_tokens": tokens, "image_count": len(image_paths),
                      "sparse_max_logprob_error": max_error, "frozen_base_after_adapter_load_error": base_error,
                      "pre_update_repeat_error": repeat_error,
                      "initial_topk_vs_later_fixed_id_error": cold_warm_error,
                      "initial_topk": top,
                      "student_adapter_max_difference": adapter_diff, "one_token_eos": eos,
                      "adapter_load": loaded, "status": "passed" if passed else "failed",
                      "fixed_ids": all_ids, "base_before": before, "base_after": after, "student": student}
    Path(args.output).write_text(json.dumps(artifact, indent=2))
    print(json.dumps({k: v for k, v in artifact.items() if k not in {"base_before", "base_after", "student", "initial_topk"}}), flush=True)
    assert passed, "See fixed-ID scoring artifact for serving variability or adapter isolation failure"


if __name__ == "__main__":
    main()
