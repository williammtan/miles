"""PTD transport fields must survive batching without becoming scalar metrics."""

import math
import importlib.util
import sys
from argparse import Namespace
from types import ModuleType, SimpleNamespace

import pytest
import torch

from miles.utils.types import Sample
from tests.fast.backends.training_utils.loss.loss_test_utils import make_parallel_state


@pytest.fixture
def logging_modules(monkeypatch):
    # The real logger imports DataIterator, whose dataset utilities also import
    # optional SGLang chat templates. These tests never render chat templates.
    if importlib.util.find_spec("sglang") is None:
        templates = ModuleType("miles.utils.chat_template_utils")
        monkeypatch.setitem(sys.modules, templates.__name__, templates)
    from miles.backends.training_utils import log_utils
    from miles.backends.training_utils.data import DataIterator
    from miles.ray.rollout.train_data_conversion import convert_samples_to_train_data, split_train_data_by_dp_raw

    return SimpleNamespace(log_utils=log_utils, DataIterator=DataIterator,
                           convert=convert_samples_to_train_data, split=split_train_data_by_dp_raw)


@pytest.mark.parametrize("contexts", [[None, None], [None, {}], [{}, {}]])
@pytest.mark.parametrize("legacy_rows", [0, 2])
def test_rollout_logging_excludes_ptd_metadata_and_preserves_metrics(monkeypatch, logging_modules, contexts, legacy_rows):
    log_utils = logging_modules.log_utils
    make_parallel_state()
    args = Namespace(qkv_format="thd", ci_test=False, log_multi_turn=False, log_correct_samples=False)
    captured = []

    def gather(metric_name, args, rollout_id, log_dict):
        captured.append(log_dict)

    monkeypatch.setattr(log_utils, "gather_log_data", gather)
    baseline = {
        "response_lengths": [3, 2], "total_lengths": [5, 4],
        "loss_masks": [torch.ones(3), torch.ones(2)], "rewards": [1.0, 0.0],
        "rollout_log_probs": [torch.tensor([-1.0, -2.0, -3.0]), torch.tensor([-4.0, -5.0])],
    }
    log_utils.log_rollout_data(0, args, baseline)
    with_ptd = {
        **baseline,
        "ptd_teacher_ids": [torch.zeros((legacy_rows, 100), dtype=torch.int32) for _ in contexts],
        "ptd_teacher_log_probs": [torch.zeros((legacy_rows, 100), dtype=torch.float32) for _ in contexts],
        "ptd_teacher_context": contexts,
        "ptd_normalizers": [[5000, 3000], [5000, 3000]],
    }
    log_utils.log_rollout_data(0, args, with_ptd)
    assert captured[1] == captured[0]
    assert not any(key.startswith("ptd_") for key in captured[1])
    assert all(math.isfinite(value) for pair in captured[1].values() for value in pair)


def test_context_and_empty_codec_tensors_survive_conversion_and_dp_batching(logging_modules):
    args = Namespace(ptd_coef=0.05, ptd_top_k=100, reward_key=None, advantage_estimator="grpo",
                     rewards_normalization=True, grpo_std_normalization=True, n_samples_per_prompt=2,
                     rollout_batch_size=1, use_dynamic_global_batch_size=False, balance_data=False)
    samples = [Sample(tokens=[1, 2, 3], response_length=2, reward=reward, index=index)
               for index, reward in enumerate([1.0, 0.0])]
    context = {"response_tokens": [2, 3], "payload": {"input_ids": [9, 1, 2, 3], "image_data": ["same.png"]}}
    samples[1].ptd_teacher_context = context
    samples[1].ptd_teacher_ids = torch.empty((0, 100), dtype=torch.int32)
    samples[1].ptd_teacher_log_probs = torch.empty((0, 100), dtype=torch.float32)
    data = logging_modules.convert(args, samples, {}, None, None)
    assert data["raw_reward"] == [1.0, 0.0]
    shards = logging_modules.split(args, data, dp_size=2)
    keys = ["ptd_teacher_ids", "ptd_teacher_log_probs", "ptd_teacher_context"]
    for rank, shard in enumerate(shards):
        batch = logging_modules.DataIterator(shard, micro_batch_size=1).get_next(keys)
        assert batch["ptd_teacher_context"] == [None if rank == 0 else context]
        assert batch["ptd_teacher_ids"][0].shape == batch["ptd_teacher_log_probs"][0].shape == (0, 100)
        assert batch["ptd_teacher_ids"][0].dtype == torch.int32
        assert batch["ptd_teacher_log_probs"][0].dtype == torch.float32
