"""Live joint scoring check; cold/warm drift is measured, never mixed into q."""

import argparse
import hashlib
import json
import uuid
from argparse import Namespace
from pathlib import Path

import requests
from PIL import Image
from transformers import AutoProcessor

import miles.rollout.ptd as ptd
from miles.utils.processing_utils import call_processor
from miles.utils.types import Sample


def post(url, payload):
    response = requests.post(url, json=payload, timeout=600)
    response.raise_for_status()
    return response.json()


def sample_for_check(args, processor):
    if args.trace_file:
        row = json.loads(Path(args.trace_file).read_text())[args.row]
        media = row["metadata"]["_deferred_media_refs"]["image"]
        return Sample(tokens=row["tokens"], response_length=row["response_length"],
                      metadata={"ptd_media_payload": {"image_data": media}})
    with open(args.dataset) as stream:
        row = [json.loads(line) for line in stream][args.row]
    Image.MAX_IMAGE_PIXELS = None
    images = [Image.open(path).convert("RGB") for path in row["images"]]
    content = []
    for index, segment in enumerate(row["problem"].split("<image>")):
        if index:
            content.append({"type": "image"})
        if segment:
            content.append({"type": "text", "text": segment})
    prompt = processor.apply_chat_template([{"role": "user", "content": content}],
                                            tokenize=False, add_generation_prompt=True)
    ids = list(call_processor(processor, prompt, {"images": images})["input_ids"][0])
    result = post(args.url + "/generate", {
        "input_ids": ids, "image_data": row["images"], "lora_path": None, "return_logprob": True,
        "sampling_params": {"max_new_tokens": 8, "temperature": 0, "skip_special_tokens": False},
    })
    tokens = [int(item[1]) for item in result["meta_info"]["output_token_logprobs"]]
    assert len(tokens) == 8
    return Sample(tokens=ids + tokens, response_length=len(tokens),
                  metadata={"ptd_media_payload": {"image_data": row["images"]}})


def joint_check(context, selected):
    """Capture the real HTTP result to independently check the production helper."""
    captured = []
    original = ptd.request_scores

    def capture(url, payload, timeout):
        result = original(url, payload, timeout)
        captured.append(result)
        return result

    ptd.request_scores = capture
    try:
        ids, values = ptd.score_teacher_joint(context, selected, len(selected[0]), 600)
    finally:
        ptd.request_scores = original
    assert len(captured) == 1
    top = ptd.extract_score_rows(captured[0], context, "input_top_logprobs")
    cross = ptd.extract_score_rows(captured[0], context, "input_token_ids_logprobs")
    error, overlaps, maps = 0, 0, []
    for ii, vv, tt, cc in zip(ids, values, top, cross, strict=True):
        teacher = {int(e[1]): float(e[0]) for e in tt}
        student = {int(e[1]): float(e[0]) for e in cc}
        for token in teacher.keys() & student.keys():
            overlaps += 1
            error = max(error, abs(teacher[token] - student[token]))
        expected = {**student, **teacher}
        actual = {i: v for i, v in zip(ii, vv, strict=True) if i >= 0}
        assert actual == expected
        maps.append(student)
    assert error < 2e-4
    return maps, {"same_forward_overlap_error": error, "overlap_count": overlaps, "requests": 1}


def max_difference(left, right):
    return max(abs(value - b[token]) for a, b in zip(left, right, strict=True) for token, value in a.items())


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--url", required=True)
    parser.add_argument("--model", default="/weka/models/Qwen/Qwen3.8-27B")
    parser.add_argument("--dataset", default="/weka/handshake/vqa/train_le24k.jsonl")
    parser.add_argument("--trace-file")
    parser.add_argument("--row", type=int, default=0)
    parser.add_argument("--adapter", default="/weka/handshake/checkpoints/qwen3p8-27b-lora/iter_0000019/adapter")
    parser.add_argument("--read-only-existing-adapter", action="store_true")
    parser.add_argument("--output", default="/tmp/ptd-joint-check.json")
    args = parser.parse_args()
    processor = AutoProcessor.from_pretrained(args.model, trust_remote_code=True)
    sample = sample_for_check(args, processor)
    config = json.loads((Path(args.model) / "config.json").read_text())
    score_args = Namespace(ptd_teacher_url=args.url + "/generate",
                           ptd_vocab_size=config.get("text_config", config)["vocab_size"])
    hint = "Read row and column labels before comparing the indicated objects. Diagnostic " + uuid.uuid4().hex
    context = ptd.teacher_score_context(score_args, sample, processor.tokenizer, hint)
    assert context["payload"]["input_ids"][-len(sample.tokens):] == sample.tokens
    assert context["payload"]["image_data"] == sample.metadata["ptd_media_payload"]["image_data"]
    selected = [list(dict.fromkeys([token, 42 + i % 47, 100 + i % 53, 210, 211]))[:4]
                for i, token in enumerate(context["response_tokens"])]
    print("Checking one-forward teacher scores on original images and response IDs", flush=True)
    cold, cold_report = joint_check(context, selected)
    warm, warm_report = joint_check(context, selected)
    if not args.read_only_existing_adapter:
        post(args.url + "/load_lora_adapter", {"lora_name": "miles_lora", "lora_path": args.adapter})
    after, _ = joint_check(context, selected)
    student_context = {**context, "payload": {**context["payload"], "lora_path": "miles_lora"}}
    student, student_report = joint_check(student_context, selected)
    eos = processor.tokenizer.eos_token_id
    prefix = sample.tokens[:-sample.response_length]
    one = Sample(tokens=prefix + [eos], response_length=1, metadata=sample.metadata)
    one_context = ptd.teacher_score_context(score_args, one, processor.tokenizer, hint)
    _, eos_report = joint_check(one_context, [[eos, 42, 100, 210]])
    base_error, adapter_diff = max_difference(warm, after), max_difference(after, student)
    report = {"status": "passed" if base_error < 2e-4 and adapter_diff > 1e-5 else "failed",
              "joint_cold": cold_report, "joint_warm": warm_report, "joint_student": student_report,
              "joint_eos": eos_report, "cold_warm_logprob_difference": max_difference(cold, warm),
              "frozen_base_repeat_error": base_error, "student_adapter_difference": adapter_diff,
              "adapter_operation": "reuse_existing" if args.read_only_existing_adapter else "load_from_checkpoint",
              "prompt_tokens": len(prefix), "response_tokens": sample.response_length,
              "response_ids_sha256": hashlib.sha256(json.dumps(context["response_tokens"]).encode()).hexdigest(),
              "image_count": len(context["payload"]["image_data"])}
    Path(args.output).write_text(json.dumps(report, indent=2))
    print(json.dumps(report), flush=True)
    assert report["status"] == "passed", "Joint scoring passed but adapter isolation check failed"


if __name__ == "__main__":
    main()
