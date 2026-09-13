"""Path-carried images: media_refs travel per sample to the trainer and disable rollout-side scheduling."""

from __future__ import annotations

from tests.fast.ray.rollout.conftest import make_args, make_sample

from miles.ray.rollout.train_data_conversion import (
    ROLLOUT_DATA_VALUE_SPEC,
    can_schedule_on_rollout_side,
    convert_samples_to_train_data,
    split_train_data_by_dp_raw,
)


def _samples():
    refs = [{"image": ["/media/a.png", "/media/b.png"]}, None, {"image": ["/media/c.png"]}, None]
    samples = []
    for index, ref in enumerate(refs):
        sample = make_sample(group_index=index // 2, index=index, reward=float(index % 2))
        if ref is not None:
            sample.metadata = {"_deferred_media_refs": ref}
        samples.append(sample)
    return samples, refs


def test_media_refs_are_lifted_per_sample_and_sharded():
    args = make_args(rewards_normalization=False)
    samples, refs = _samples()
    train_data = convert_samples_to_train_data(
        args, samples, metadata={}, custom_convert_samples_to_train_data_func=None, custom_reward_post_process_func=None
    )
    assert train_data["media_refs"] == refs
    assert "multimodal_train_inputs" not in train_data
    assert ROLLOUT_DATA_VALUE_SPEC["media_refs"].codec == "msgpack_ragged"

    shards = split_train_data_by_dp_raw(args, train_data, dp_size=2)
    seen = [ref for shard in shards for ref in shard["media_refs"]]
    assert sorted(seen, key=str) == sorted(refs, key=str)
    for shard in shards:
        assert len(shard["media_refs"]) == len(shard["tokens"])


def test_pixel_tensors_take_precedence_over_refs():
    args = make_args(rewards_normalization=False)
    samples, _ = _samples()
    samples[0].multimodal_train_inputs = {"pixel_values": [1]}
    train_data = convert_samples_to_train_data(
        args, samples, metadata={}, custom_convert_samples_to_train_data_func=None, custom_reward_post_process_func=None
    )
    assert "media_refs" not in train_data and "multimodal_train_inputs" in train_data


def test_media_refs_keep_the_schedule_on_the_trainer():
    args = make_args()
    config = {"dp_size": 1, "cp_size": 1, "vpp_size": 1, "microbatch_group_size_per_vp_stage": 1}
    data = {"rollout_ids": list(range(8)), "media_refs": [None] * 8}
    assert can_schedule_on_rollout_side(args, data, config) is False
    assert can_schedule_on_rollout_side(args, {"rollout_ids": list(range(8))}, config) is True
