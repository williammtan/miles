"""Exercise real data-source methods without importing serving dependencies."""

import ast
import copy
import json
import logging
import os
import random
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import torch

from miles.rollout import ptd_checkpoint as checkpoint


def source_class():
    tree = ast.parse((Path(__file__).parents[2] / "miles/rollout/data_source.py").read_text())
    nodes = [node for node in tree.body if getattr(node, "name", None) in
             {"RolloutDataSource", "RolloutDataSourceWithBuffer", "pop_first"}]
    namespace = dict(DataSource=object, Sample=SimpleNamespace, copy=copy, os=os, torch=torch,
                     logger=logging.getLogger(__name__), load_cursor=checkpoint.load_cursor,
                     save_cursor=checkpoint.save_cursor)
    exec(compile(ast.Module(body=nodes, type_ignores=[]), "data_source.py", "exec"), namespace)
    return namespace["RolloutDataSourceWithBuffer"], namespace["pop_first"]


class SmallDataset:
    def __init__(self):
        self.origin_samples = [SimpleNamespace(task_id=f"q{i}") for i in range(5)]
        self.samples = self.origin_samples

    def __len__(self):
        return len(self.samples)

    def shuffle(self, epoch):
        # Same seed+epoch permutation contract as Dataset.shuffle.
        random.seed(42 + epoch)
        indices = list(range(len(self)))
        random.shuffle(indices)
        self.samples = [self.origin_samples[i] for i in indices]


@pytest.fixture
def source(tmp_path):
    data = tmp_path / "train.jsonl"
    data.write_text('{"problem":"fixture"}\n')
    model = tmp_path / "model"
    model.mkdir()
    for name in ("config.json", "model.safetensors.index.json", "preprocessor_config.json"):
        (model / name).write_text("{}")
    cls, pop = source_class()
    value = cls.__new__(cls)
    value.args = SimpleNamespace(prompt_data=str(data), save=str(tmp_path / "run"),
                                 hf_checkpoint=str(model), ptd_coef=.05, ptd_exact_checkpoints=True,
                                 rollout_global_dataset=True, n_samples_per_prompt=2, rollout_shuffle=True,
                                 rollout_seed=42, rollout_batch_size=2, lora_rank=32, lora_alpha=64,
                                 num_rollout=3, start_rollout_id=None, rollout_resume_dir=None,
                                 lora_adapter_path=None)
    value.dataset = SmallDataset()
    value.dataset.shuffle(0)
    value.epoch_id = value.sample_offset = value.sample_index = value.sample_group_index = 0
    value.metadata = {"version": 1}
    value.buffer = []
    value.buffer_filter = pop
    return value


def compact(groups):
    return [[(s.task_id, s.index, s.group_index) for s in group] for group in groups]


def commit(source, iteration=0):
    source.save(iteration)
    adapter = Path(source.args.save) / f"iter_{iteration:07d}" / "adapter"
    adapter.mkdir(parents=True)
    (adapter / "checkpoint_complete.json").write_text(json.dumps({"version": 2, "iteration": iteration,
                                                                  "training": True, "ranks": []}))
    checkpoint.publish_checkpoint(source.args, iteration)
    source.args.lora_adapter_path = str(adapter)
    source.args.rollout_resume_dir = source.args.save


def test_cursor_buffer_shuffle_wraparound_and_rng_continue_exactly(source):
    groups = source.get_samples(4)
    source.add_samples(groups[-1:])
    source.save(0)
    expected_rng = random.random(), np.random.rand(), torch.rand(3)
    expected = compact(source.get_samples(3))  # Pending group, then dataset wraparound.
    expected_cursor = source.epoch_id, source.sample_offset, source.sample_index, source.sample_group_index
    source.args.rollout_resume_dir = source.args.save
    source.load(0)
    observed_rng = random.random(), np.random.rand(), torch.rand(3)
    assert expected_rng[:2] == observed_rng[:2] and torch.equal(expected_rng[2], observed_rng[2])
    assert compact(source.get_samples(3)) == expected
    assert (source.epoch_id, source.sample_offset, source.sample_index, source.sample_group_index) == expected_cursor


@pytest.mark.parametrize("change", ["dataset", "seed", "samples", "missing"])
def test_explicit_cursor_resume_rejects_incompatible_or_missing_data(source, change):
    source.save(0)
    source.args.rollout_resume_dir = source.args.save
    if change == "dataset":
        Path(source.args.prompt_data).write_text("changed\n")
    elif change == "seed":
        source.args.rollout_seed += 1
    elif change == "samples":
        source.args.n_samples_per_prompt += 1
    else:
        checkpoint.cursor_path(source.args.save, 0).unlink()
    with pytest.raises((ValueError, FileNotFoundError)):
        source.load(0)


def test_checkpoint_commit_and_final_iteration_resume(source):
    commit(source, 2)
    assert checkpoint.validate_resume(source.args) == 3
    source.args.start_rollout_id = 2
    with pytest.raises(ValueError, match="start/end"):
        checkpoint.validate_resume(source.args)


@pytest.mark.parametrize("change", ["cursor", "adapter", "alpha", "missing_commit", "wrong_adapter"])
def test_checkpoint_binding_rejects_partial_mixed_or_reconfigured_resume(source, change):
    commit(source)
    adapter = Path(source.args.lora_adapter_path)
    if change == "cursor":
        checkpoint.cursor_path(source.args.save, 0).write_bytes(b"corrupt")
    elif change == "adapter":
        (adapter / "checkpoint_complete.json").write_text("{}")
    elif change == "alpha":
        source.args.lora_alpha *= 2
    elif change == "missing_commit":
        (adapter.parent / "ptd_checkpoint_complete.json").unlink()
    else:
        source.args.lora_adapter_path = str(adapter.parent.parent / "iter_0000001/adapter")
    with pytest.raises((ValueError, FileNotFoundError)):
        checkpoint.validate_resume(source.args)


def test_cannot_publish_without_cursor_or_overwrite_completed_checkpoint(source):
    commit(source)
    with pytest.raises(ValueError, match="overwrite"):
        checkpoint.publish_checkpoint(source.args, 0)
    checkpoint.cursor_path(source.args.save, 0).unlink()
    with pytest.raises(FileNotFoundError):
        checkpoint.publish_checkpoint(source.args, 0)


def test_final_resume_eval_and_async_rejection_are_wired():
    root = Path(__file__).parents[2]
    sync = ast.parse((root / "train.py").read_text())
    checks = [node for node in ast.walk(sync) if isinstance(node, ast.If)
              and "args.start_rollout_id == args.num_rollout" in ast.unparse(node.test)]
    assert len(checks) == 1 and "rollout_id=args.num_rollout - 1" in ast.unparse(checks[0])
    asynchronous = ast.parse((root / "train_async.py").read_text())
    train = next(node for node in asynchronous.body if getattr(node, "name", None) == "train")
    assert isinstance(train.body[0], ast.If)
    assert "ptd_exact_checkpoints" in ast.unparse(train.body[0].test)
