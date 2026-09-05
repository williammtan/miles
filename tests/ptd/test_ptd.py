"""Run without the optional rollout fixtures: pytest --confcutdir=tests/ptd tests/ptd."""

import asyncio
import copy
from argparse import Namespace
from types import SimpleNamespace

import pytest
import torch

from miles.backends.training_utils.loss_hub import ptd
from miles.backends.training_utils.loss_hub.losses import policy_loss_function
from miles.backends.training_utils.loss_hub.ptd_math import sparse_vocab_parallel_jsd, support_union
from miles.backends.training_utils.parallel import set_parallel_state
from miles.rollout.ptd import collect_teacher_targets, extract_score_rows, is_tutor_target, teacher_score_context
from miles.utils.types import Sample
from tests.fast.backends.training_utils.loss.loss_test_utils import make_args, make_parallel_state


def dense_reference(logits, teacher, ids):
    p_full = logits.softmax(-1)
    q_full = teacher.detach().softmax(-1)
    losses = []
    for row, qrow, selected in zip(p_full, q_full, ids, strict=True):
        support = selected[selected >= 0]
        other = torch.ones(row.shape, dtype=torch.bool, device=row.device)
        other[support] = False
        p = torch.cat((row[support], row[other].sum().view(1)))
        q = torch.cat((qrow[support], qrow[other].sum().view(1)))
        m = (p + q) / 2
        tiny = torch.finfo(p.dtype).tiny
        losses.append((p * (p.clamp_min(tiny).log() - m.clamp_min(tiny).log())).sum() / 2
                      + (q * (q.clamp_min(tiny).log() - m.clamp_min(tiny).log())).sum() / 2)
    return torch.stack(losses)


@pytest.mark.parametrize("k", [1, 4, 11])
def test_sparse_values_gradients_match_dense_coarsening(k):
    torch.manual_seed(19)
    x = torch.randn(7, 11, dtype=torch.float64, requires_grad=True)
    teacher = torch.randn_like(x, requires_grad=True)
    ids = support_union(x.detach().topk(k).indices, teacher.detach().topk(k).indices)
    q = teacher.detach().log_softmax(-1).gather(-1, ids.clamp_min(0))
    sparse = sparse_vocab_parallel_jsd(x, ids, q, chunk_size=3)
    reference = dense_reference(x, teacher, ids)
    torch.testing.assert_close(sparse, reference, atol=2e-14, rtol=2e-13)
    sparse_grad = torch.autograd.grad(sparse.sum(), x, retain_graph=True)[0]
    dense_grad = torch.autograd.grad(reference.sum(), x)[0]
    torch.testing.assert_close(sparse_grad, dense_grad, atol=2e-14, rtol=2e-12)
    assert teacher.grad is None


def batch_and_args(targets):
    make_parallel_state()
    args = make_args(
        ptd_coef=0.5, ptd_top_k=2, ptd_vocab_size=7, ptd_logits_chunk_size=2, ptd_score_timeout=5,
        entropy_coef=0, observe_training_entropy=False, calculate_per_token_loss=True,
        skip_actor_forward_only=True, global_batch_size=len(targets),
    )
    torch.manual_seed(13)
    tokens = [torch.tensor([1, 2, 3, 4, 5]), torch.tensor([2, 3, 6, 1])][:len(targets)]
    response_lengths = [3, 1][:len(targets)]
    teacher = torch.tensor([1., -1., 2., 3., 0., 1.5, -3.]).log_softmax(-1)
    teacher_vals, teacher_ids = teacher.topk(2)
    masks = [torch.ones(n) for n in response_lengths]
    total_tokens = sum(response_lengths)
    selected_tokens = sum(n for n, target in zip(response_lengths, targets, strict=True) if target)
    batch = {
        "unconcat_tokens": tokens, "response_lengths": response_lengths, "total_lengths": list(map(len, tokens)),
        "loss_masks": masks, "advantages": [torch.zeros(n) for n in response_lengths],
        "ptd_teacher_context": [{"response_tokens": t[-n:].tolist()} if active else None
                                for t, n, active in zip(tokens, response_lengths, targets, strict=True)],
        "ptd_teacher_ids": [teacher_ids.expand(n, -1) if active else torch.empty(0, 2, dtype=torch.long)
                            for n, active in zip(response_lengths, targets, strict=True)],
        "ptd_teacher_log_probs": [teacher_vals.expand(n, -1) if active else torch.empty(0, 2)
                                  for n, active in zip(response_lengths, targets, strict=True)],
        "ptd_normalizers": [[total_tokens, selected_tokens]] * len(targets),
    }
    logits = torch.randn(1, sum(map(len, tokens)), 7, requires_grad=True)
    return args, batch, logits, teacher


def install_teacher(monkeypatch, teacher):
    calls = []

    def score(context, ids, timeout):
        calls.append(ids)
        return [{i: teacher[i].item() for i in row} for row in ids]

    monkeypatch.setattr(ptd, "score_teacher_missing_ids", score)
    return calls


def test_all_wrong_zero_advantages_have_nonzero_tutor_gradient(monkeypatch):
    args, batch, logits, teacher = batch_and_args([True, True])
    calls = install_teacher(monkeypatch, teacher)
    loss, metrics = policy_loss_function(args, batch, logits, torch.sum)
    loss.backward()
    assert metrics["pg_loss"] == 0
    assert loss > 0 and logits.grad.norm() > 0
    assert calls
    # Every prompt-only logit, final unused logit, and padding logit is untouched.
    assert torch.equal(logits.grad[0, [0, 4, 5, 6, 8]], torch.zeros(5, 7))


def test_correct_responses_have_exactly_zero_tutor_loss_gradient(monkeypatch):
    args, batch, logits, teacher = batch_and_args([False, False])
    calls = install_teacher(monkeypatch, teacher)
    loss, metrics = policy_loss_function(args, batch, logits, torch.sum)
    loss.backward()
    assert loss.item() == metrics["ptd_loss"].item() == 0
    assert torch.count_nonzero(logits.grad) == 0
    assert not calls


def test_first_correct_later_failed_and_selected_denominator(monkeypatch):
    args, batch, logits, teacher = batch_and_args([False, True])
    install_teacher(monkeypatch, teacher)
    loss, _ = policy_loss_function(args, batch, logits, torch.sum)
    # Final Megatron division by all 4 response tokens cancels the inserted factor.
    response = logits[0, 7:8]
    ids = support_union(response.detach().topk(2).indices, teacher.topk(2).indices.view(1, -1))
    expected = args.ptd_coef * dense_reference(response, teacher.expand_as(response), ids).mean()
    torch.testing.assert_close(loss / 4, expected)
    loss.backward()
    assert torch.count_nonzero(logits.grad[0, :7]) == 0
    assert logits.grad[0, 7].norm() > 0


def test_lambda_zero_exact_baseline_grpo(monkeypatch):
    args, batch, logits, teacher = batch_and_args([True, True])
    batch["advantages"] = [torch.tensor([1., -1., 0.5]), torch.tensor([-0.5])]
    args.ptd_coef = 0
    baseline_args = copy.copy(args)
    delattr(baseline_args, "ptd_coef")
    # No target fields are required and no teacher is contacted at lambda=0.
    batch = {key: val for key, val in batch.items() if not key.startswith("ptd_")}
    calls = install_teacher(monkeypatch, teacher)
    baseline_logits = logits.detach().clone().requires_grad_()
    baseline, baseline_metrics = policy_loss_function(baseline_args, batch, baseline_logits, torch.sum)
    loss, metrics = policy_loss_function(args, batch, logits, torch.sum)
    loss.backward()
    baseline.backward()
    assert loss.item() == baseline.item()
    assert metrics.keys() == baseline_metrics.keys()
    assert torch.equal(logits.grad, baseline_logits.grad)
    assert not calls


@pytest.mark.parametrize("status", [Sample.Status.COMPLETED, Sample.Status.TRUNCATED])
def test_same_image_and_token_alignment_including_eos_and_truncation(status):
    args = Namespace(ptd_teacher_url="http://teacher/generate")
    tokenizer = SimpleNamespace(apply_chat_template=lambda messages, **kw: [90, 91, 4], encode=lambda *a, **kw: [4])
    sample = Sample(tokens=[10, 20, 21, 30, 99], response_length=2, status=status,
                    metadata={"ptd_media_payload": {"image_data": ["/same/image.png", "base64_pixels"]}})
    context = teacher_score_context(args, sample, tokenizer, "Check the object left of the ruler.")
    payload = context["payload"]
    assert payload["image_data"] is sample.metadata["ptd_media_payload"]["image_data"]
    assert payload["input_ids"] == [90, 91, 10, 20, 21, 30, 99]
    assert payload["logprob_start_len"] == 4
    assert payload["lora_path"] is None
    score = {"meta_info": {"input_token_logprobs": [[None, 21], [-2, 30], [-3, 99]],
                           "input_top_logprobs": [None, [[-1, 30]], [[-1, 99]]]}}
    assert len(extract_score_rows(score, context, "input_top_logprobs")) == 2
    score["meta_info"]["input_token_logprobs"][-1][1] = 98
    with pytest.raises(ValueError, match="token IDs"):
        extract_score_rows(score, context, "input_top_logprobs")


@pytest.mark.parametrize("valid,reward,expected", [(True, 0, True), (True, 1, False), (False, 0, False), (None, 0, False)])
def test_grade_valid_is_separate_from_wrong_answer(valid, reward, expected):
    args = Namespace(ptd_coef=0.5, reward_key=None)
    sample = Sample(tokens=[1, 2], response_length=1, reward=reward, status=Sample.Status.COMPLETED,
                    metadata={"grade_valid": valid})
    assert is_tutor_target(args, sample) is expected


def test_step_normalizer_includes_all_microbatches():
    set_parallel_state(SimpleNamespace(effective_dp=SimpleNamespace(size=1)))
    data = {"tokens": [None] * 3, "loss_masks": [torch.ones(3), torch.ones(7), torch.ones(2)],
            "ptd_teacher_context": [None, {}, {}]}
    iterator = SimpleNamespace(micro_batch_indices=[[1], [0, 2]])
    ptd.attach_ptd_normalizers(Namespace(ptd_coef=0.5), data, [iterator], [2])
    assert data["ptd_normalizers"] == [[12, 9]] * 3


def test_unusable_hint_retains_reward_without_teacher_targets(monkeypatch):
    async def unusable(args, sample):
        sample.metadata["ptd_hint_status"] = "needs_review"
        return None

    monkeypatch.setattr("miles.utils.misc.load_function", lambda path: unusable)
    args = Namespace(ptd_coef=0.5, reward_key=None, ptd_hint_function_path="test.unusable")
    sample = Sample(tokens=[1, 2], response_length=1, reward=0.0, status=Sample.Status.COMPLETED,
                    metadata={"grade_valid": True})
    asyncio.run(collect_teacher_targets(args, sample, None))
    assert sample.reward == 0.0
    assert sample.ptd_teacher_context is None
    assert sample.ptd_teacher_ids is None
    assert sample.metadata["ptd_hint_status"] == "needs_review"


def test_correct_response_clears_stale_targets_without_generating_hint():
    args = Namespace(ptd_coef=0.5, reward_key=None)
    sample = Sample(tokens=[1, 2], response_length=1, reward=1.0, status=Sample.Status.COMPLETED,
                    metadata={"grade_valid": True})
    sample.ptd_teacher_context = {"stale": True}
    asyncio.run(collect_teacher_targets(args, sample, None))
    assert sample.ptd_teacher_context is None
