"""Bind synchronous PTD model checkpoints to their matching rollout cursor."""

import hashlib
import json
import os
import random
import re
import tempfile
from pathlib import Path

import numpy as np
import torch


VERSION = 1
DATA_KEYS = (
    "input_key", "label_key", "metadata_key", "tool_key", "multimodal_keys",
    "apply_chat_template", "apply_chat_template_kwargs", "rollout_seed", "rollout_shuffle",
    "rollout_batch_size", "over_sampling_batch_size", "n_samples_per_prompt", "rollout_max_prompt_len",
)
RUN_KEYS = (
    "lora_rank", "lora_alpha", "lora_dropout", "target_modules", "lr", "lr_decay_style",
    "lr_decay_iters", "lr_decay_samples", "num_rollout", "global_batch_size", "weight_decay",
    "adam_beta1", "adam_beta2", "ptd_coef", "ptd_top_k", "ptd_vocab_size", "ptd_hint_function_path",
    "ptd_score_mode", "ptd_score_concurrency",
    "advantage_estimator", "kl_coef", "kl_loss_coef", "entropy_coef", "calculate_per_token_loss",
    "eps_clip", "eps_clip_high", "rollout_temperature", "rollout_top_p", "rollout_max_response_len",
    "tensor_model_parallel_size", "context_parallel_size", "pipeline_model_parallel_size",
    "sequence_parallel", "qkv_format", "use_dynamic_batch_size", "max_tokens_per_gpu", "balance_data",
)


def file_digest(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as source:
        for block in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def atomic_write(path, writer):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    try:
        with os.fdopen(descriptor, "wb") as destination:
            writer(destination)
            destination.flush()
            os.fsync(destination.fileno())
        os.replace(temporary, path)
        directory = os.open(path.parent, os.O_RDONLY | os.O_DIRECTORY)
        try:
            os.fsync(directory)
        finally:
            os.close(directory)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)


def data_identity(args):
    # PTD recovery supports the local, immutable JSONL dataset used by this recipe.
    identity = {key: getattr(args, key, None) for key in DATA_KEYS}
    identity["dataset_sha256"] = file_digest(args.prompt_data)
    template = getattr(args, "chat_template_path", None)
    identity["chat_template_sha256"] = file_digest(template) if template else None
    return identity


def run_identity(args):
    identity = {key: getattr(args, key, None) for key in RUN_KEYS}
    identity["data"] = data_identity(args)
    model = Path(args.hf_checkpoint)
    identity["model_path"] = str(model.resolve())
    identity["model"] = {name: file_digest(model / name) for name in
                         ("config.json", "model.safetensors.index.json", "preprocessor_config.json")}
    return identity


def cursor_path(root, iteration):
    return Path(root) / "rollout" / f"global_dataset_state_dict_{iteration}.pt"


def save_cursor(source, iteration):
    state = {
        "version": VERSION, "iteration": iteration, "data_identity": data_identity(source.args),
        "dataset_size": len(source.dataset),
        "cursor": {key: getattr(source, key) for key in
                   ("sample_offset", "epoch_id", "sample_group_index", "sample_index", "metadata")},
        "buffer": getattr(source, "buffer", []),
        "rng": {"python": random.getstate(), "numpy": np.random.get_state(), "torch": torch.get_rng_state()},
    }
    atomic_write(cursor_path(source.args.save, iteration), lambda stream: torch.save(state, stream))


def load_cursor(source, root, iteration):
    state = torch.load(cursor_path(root, iteration), map_location="cpu", weights_only=False)
    if (state.get("version") != VERSION or state.get("iteration") != iteration
            or state.get("data_identity") != data_identity(source.args)
            or state.get("dataset_size") != len(source.dataset)):
        raise ValueError("PTD rollout cursor identity or iteration does not match the requested run")
    cursor = state["cursor"]
    counters = ("sample_offset", "epoch_id", "sample_group_index", "sample_index")
    if any(type(cursor.get(key)) is not int or cursor[key] < 0 for key in counters):
        raise ValueError("PTD rollout cursor contains invalid counters")
    if cursor["sample_offset"] > len(source.dataset):
        raise ValueError("PTD rollout cursor offset exceeds the dataset")
    if not isinstance(state["buffer"], list) or (state["buffer"] and not hasattr(source, "buffer")):
        raise ValueError("PTD rollout cursor contains an incompatible pending buffer")
    for key, value in cursor.items():
        if key in (*counters, "metadata"):
            setattr(source, key, value)
    if hasattr(source, "buffer"):
        source.buffer = state["buffer"]
    if source.args.rollout_shuffle:
        source.dataset.shuffle(source.epoch_id)
    random.setstate(state["rng"]["python"])
    np.random.set_state(state["rng"]["numpy"])
    torch.set_rng_state(state["rng"]["torch"])


def publish_checkpoint(args, iteration):
    if not getattr(args, "ptd_exact_checkpoints", False):
        return
    root = Path(args.save)
    adapter = root / f"iter_{iteration:07d}" / "adapter" / "checkpoint_complete.json"
    metadata = json.loads(adapter.read_text())
    if metadata.get("iteration") != iteration or metadata.get("training") is not True:
        raise ValueError("PTD model checkpoint is not a complete training checkpoint")
    cursor = cursor_path(root, iteration)
    record = {"version": VERSION, "iteration": iteration, "identity": run_identity(args),
              "adapter_manifest_sha256": file_digest(adapter), "cursor_sha256": file_digest(cursor)}
    destination = root / f"iter_{iteration:07d}" / "ptd_checkpoint_complete.json"
    if destination.exists():
        raise ValueError("Refusing to overwrite a committed PTD checkpoint")
    atomic_write(destination, lambda stream: stream.write(json.dumps(record, sort_keys=True).encode()))


def validate_resume(args):
    root_name = getattr(args, "rollout_resume_dir", None)
    if root_name is None:
        return None
    if (not getattr(args, "ptd_exact_checkpoints", False) or getattr(args, "ptd_coef", 0) <= 0
            or not getattr(args, "rollout_global_dataset", False)):
        raise ValueError("--rollout-resume-dir requires PTD exact checkpoints and its global dataset")
    root = Path(root_name).resolve()
    adapter = Path(args.lora_adapter_path or "").resolve()
    match = re.fullmatch(r"iter_(\d{7})", adapter.parent.name)
    if not match or adapter.name != "adapter" or adapter.parent.parent != root:
        raise ValueError("PTD resume requires the matching ROOT/iter_NNNNNNN/adapter path")
    iteration = int(match[1])
    record = json.loads((adapter.parent / "ptd_checkpoint_complete.json").read_text())
    if (record.get("version") != VERSION or record.get("iteration") != iteration
            or record.get("identity") != run_identity(args)
            or record.get("adapter_manifest_sha256") != file_digest(adapter / "checkpoint_complete.json")
            or record.get("cursor_sha256") != file_digest(cursor_path(root, iteration))):
        raise ValueError("PTD checkpoint configuration, model manifest or rollout cursor differs from its commit")
    expected = iteration + 1
    if args.start_rollout_id not in (None, expected) or expected > args.num_rollout:
        raise ValueError("PTD resume start/end does not match the saved training iteration")
    return expected
