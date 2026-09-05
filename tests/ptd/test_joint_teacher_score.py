"""Regressions for a coherent teacher distribution from one joint forward."""

import asyncio
import copy
import math
from argparse import Namespace
from types import SimpleNamespace

import pytest
import torch
import torch.distributed as dist
import torch.multiprocessing as mp

from miles.backends.training_utils.loss_hub import ptd as loss_ptd
from miles.backends.training_utils.parallel import set_parallel_state
from miles.rollout import ptd as rollout_ptd
from miles.utils.types import Sample


def _context():
    return {
        "url": "http://teacher/generate", "response_tokens": [2, 3], "vocab_size": 5,
        "payload": {"input_ids": [10, 11, 0, 1, 2, 3], "image_data": ["/same/image.png"],
                    "lora_path": None, "return_logprob": True, "logprob_start_len": 3,
                    "sampling_params": {"max_new_tokens": 0, "temperature": 0}},
    }


def _response(context, probabilities, student_ids, top_k):
    top_rows, cross_rows = [None], [None]
    for probs, ids in zip(probabilities, student_ids, strict=True):
        top_ids = sorted(range(len(probs)), key=probs.__getitem__, reverse=True)[:top_k]
        top_rows.append([[math.log(probs[i]), i] for i in top_ids])
        cross_rows.append([[math.log(probs[i]), i] for i in ids])
    return {"meta_info": {
        "input_token_logprobs": [[None, 1], *[[-2.0, token] for token in context["response_tokens"]]],
        "input_top_logprobs": top_rows, "input_token_ids_logprobs": cross_rows,
    }}


def _probabilities():
    return [[0.1, 0.4, 0.05, 0.25, 0.2], [0.25, 0.05, 0.1, 0.15, 0.45]]


def test_one_request_contains_both_fields_and_uses_only_joint_response(monkeypatch):
    context = _context()
    original = copy.deepcopy(context)
    student = [[0, 4], [1, 4]]
    calls = []

    def request(url, payload, timeout):
        calls.append((url, payload, timeout))
        assert payload["top_logprobs_num"] == 2
        assert payload["token_ids_logprob_positions"] == [[], *student]
        assert payload["input_ids"] == context["payload"]["input_ids"]
        assert payload["image_data"] == context["payload"]["image_data"]
        assert payload["lora_path"] is None
        return _response(context, _probabilities(), student, 2)

    monkeypatch.setattr(rollout_ptd, "request_scores", request)
    ids, log_probs = rollout_ptd.score_teacher_joint(context, student, 2, 12.0)
    assert len(calls) == 1
    assert ids == [[0, 1, 3, 4], [0, 1, 4, -1]]
    for row, values, probs in zip(ids, log_probs, _probabilities(), strict=True):
        assert len(row) == len(values) == 4
        for token, lp in zip(row, values, strict=True):
            assert lp == (math.log(probs[token]) if token >= 0 else -math.inf)
    assert context == original


@pytest.mark.parametrize("failure", ["overlap", "missing", "duplicate", "nan", "mass", "alignment", "row_count"])
def test_joint_response_rejects_incoherent_or_malformed_scores(monkeypatch, failure):
    context = _context()
    student = [[0, 4], [1, 4]]
    response = _response(context, _probabilities(), student, 2)
    meta = response["meta_info"]
    if failure == "overlap":
        meta["input_token_ids_logprobs"][2][1][0] -= 0.37457
    elif failure == "missing":
        meta["input_token_ids_logprobs"][1].pop()
    elif failure == "duplicate":
        meta["input_top_logprobs"][1][1][1] = meta["input_top_logprobs"][1][0][1]
    elif failure == "nan":
        meta["input_top_logprobs"][1][0][0] = math.nan
    elif failure == "mass":
        meta["input_token_ids_logprobs"][1][0][0] = math.log(0.5)
    elif failure == "alignment":
        meta["input_token_logprobs"][1][1] = 4
    else:
        meta["input_top_logprobs"].pop()
    monkeypatch.setattr(rollout_ptd, "request_scores", lambda *a, **kw: response)
    with pytest.raises(ValueError):
        rollout_ptd.score_teacher_joint(context, student, 2, 12.0)


def test_joint_overlap_accepts_only_small_rounding_error(monkeypatch):
    context = _context()
    student = [[0, 4], [1, 4]]
    response = _response(context, _probabilities(), student, 2)
    response["meta_info"]["input_token_ids_logprobs"][2][1][0] -= 1e-6
    monkeypatch.setattr(rollout_ptd, "request_scores", lambda *a, **kw: response)
    _, values = rollout_ptd.score_teacher_joint(context, student, 2, 12.0)
    assert values[1][2] == math.log(0.45)


def _dense_coarsening(student, teacher, ids):
    losses = []
    for p, q, row in zip(student.softmax(-1), teacher, ids, strict=True):
        selected = [token for token in row if token >= 0]
        rest = [token for token in range(p.numel()) if token not in selected]
        p = torch.cat((p[selected], p[rest].sum().view(1)))
        q = torch.cat((q[selected], q[rest].sum().view(1)))
        m = (p + q) / 2
        tiny = torch.finfo(p.dtype).tiny
        losses.append((p * (p.clamp_min(tiny).log() - m.clamp_min(tiny).log())).sum() / 2
                      + (q * (q.clamp_min(tiny).log() - m.clamp_min(tiny).log())).sum() / 2)
    return torch.stack(losses).sum()


def test_training_ignores_stale_probabilities_and_changed_teacher_topk(monkeypatch):
    state = SimpleNamespace(cp=SimpleNamespace(size=1), tp=SimpleNamespace(rank=0, size=1, group=None))
    set_parallel_state(state)
    args = Namespace(qkv_format="thd", true_on_policy_mode=False, ptd_top_k=2, ptd_vocab_size=5,
                     ptd_logits_chunk_size=1, ptd_score_timeout=12.0)
    context = _context()
    batch = {"ptd_teacher_context": [context], "response_lengths": [2], "total_lengths": [4],
             "unconcat_tokens": [torch.tensor([0, 1, 2, 3])], "loss_masks": [torch.ones(2)],
             # Deliberately wrong IDs and impossible mass from an earlier cold pass.
             "ptd_teacher_ids": [torch.tensor([[0, 2], [1, 2]])],
             "ptd_teacher_log_probs": [torch.full((2, 2), -0.001)]}
    logits = torch.tensor([[[0., 0., 0., 0., 0.], [2., -1., 0., 0.5, 1.],
                            [-1., 2., 0., 0.5, 1.], [0., 0., 0., 0., 0.]]], requires_grad=True)
    calls = []
    returned_ids = []

    def request(url, payload, timeout):
        calls.append(payload)
        student = payload["token_ids_logprob_positions"][1:]
        assert student == logits[0, 1:3].detach().topk(2).indices.tolist()
        assert payload["top_logprobs_num"] == 2
        returned_ids[:] = [sorted(set(ids) | set(sorted(range(5), key=probs.__getitem__, reverse=True)[:2]))
                           for probs, ids in zip(_probabilities(), student, strict=True)]
        return _response(context, _probabilities(), student, 2)

    monkeypatch.setattr(rollout_ptd, "request_scores", request)
    loss = loss_ptd.ptd_loss_sum(args, batch, logits)
    reference = _dense_coarsening(logits[0, 1:3], torch.tensor(_probabilities()), returned_ids)
    torch.testing.assert_close(loss, reference, atol=1e-7, rtol=1e-6)
    grad = torch.autograd.grad(loss, logits, retain_graph=True)[0]
    reference_grad = torch.autograd.grad(reference, logits)[0]
    torch.testing.assert_close(grad, reference_grad, atol=1e-7, rtol=1e-6)
    assert len(calls) == 1 and grad.norm() > 0
    del batch["ptd_teacher_ids"], batch["ptd_teacher_log_probs"]
    loss_without_legacy_fields = loss_ptd.ptd_loss_sum(args, batch, logits)
    torch.testing.assert_close(loss, loss_without_legacy_fields, atol=0, rtol=0)


def test_rollout_keeps_online_response_specific_hint_without_teacher_inference(monkeypatch):
    from miles.utils import misc

    args = Namespace(ptd_coef=0.05, reward_key=None, ptd_top_k=2, ptd_hint_function_path="online.hint",
                     ptd_hint_key="ptd_hint", ptd_teacher_url="http://teacher/generate", ptd_vocab_size=5)
    sample = Sample(tokens=[0, 1, 2, 3], response_length=2, response="the actual failed response", reward=0,
                    status=Sample.Status.COMPLETED, metadata={"grade_valid": True, "ptd_media_payload": {"image_data": ["same"]}})
    tokenizer = SimpleNamespace(apply_chat_template=lambda *a, **kw: [10, 11, 4], encode=lambda *a, **kw: [4])
    hints = []

    async def hint_fn(callback_args, callback_sample):
        assert callback_sample is sample and callback_sample.response == "the actual failed response"
        hints.append(callback_sample.tokens.copy())
        return "Check the object immediately left of the ruler."

    def forbidden_request(*args, **kwargs):
        pytest.fail("Rollout must not query the teacher model")

    monkeypatch.setattr(misc, "load_function", lambda path: hint_fn)
    monkeypatch.setattr(rollout_ptd, "request_scores", forbidden_request)
    asyncio.run(rollout_ptd.collect_teacher_targets(args, sample, tokenizer))
    assert hints == [[0, 1, 2, 3]]
    assert sample.ptd_teacher_ids.shape == sample.ptd_teacher_log_probs.shape == (0, 2)
    assert sample.ptd_teacher_context["response_tokens"] == [2, 3]
    assert sample.ptd_teacher_context["payload"]["input_ids"] == [10, 11, 0, 1, 2, 3]
    sample.validate()


def _tp_joint_worker(rank, rendezvous):
    dist.init_process_group("gloo", init_method=f"file://{rendezvous}", rank=rank, world_size=2)
    tp = SimpleNamespace(rank=rank, size=2, group=dist.group.WORLD)
    student = torch.tensor([[0, 4], [1, 4]])
    expected_ids = [[0, 1, 3, 4], [0, 1, 4, -1]]
    expected_probs = [[0.1, 0.4, 0.25, 0.2], [0.25, 0.05, 0.45, 0.0]]

    def joint(context, rows, top_k, timeout):
        assert dist.get_rank() == 0, "Only TP0 may contact the teacher"
        assert rows == student.tolist() and top_k == 2
        return expected_ids, [[math.log(value) if value > 0 else -math.inf for value in row] for row in expected_probs]

    loss_ptd.score_teacher_joint = joint
    ids, log_probs = loss_ptd._teacher_union_targets(_context(), student, timeout=12, tp=tp)
    assert ids.tolist() == expected_ids
    torch.testing.assert_close(log_probs.exp(), torch.tensor(expected_probs))

    def failing_joint(*args):
        assert dist.get_rank() == 0
        raise ValueError("inconsistent same-forward overlap")

    loss_ptd.score_teacher_joint = failing_joint
    with pytest.raises(RuntimeError, match="inconsistent same-forward overlap"):
        loss_ptd._teacher_union_targets(_context(), student, timeout=12, tp=tp)
    dist.destroy_process_group()


def test_tp_broadcasts_new_union_ids_probabilities_and_failure(tmp_path):
    mp.spawn(_tp_joint_worker, args=(str(tmp_path / "joint-tp"),), nprocs=2, join=True)
