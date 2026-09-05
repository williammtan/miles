import asyncio
import io
import json
import urllib.error

import aiohttp
import pytest
import torch

from miles.backends.training_utils.loss_hub.ptd_math import sparse_vocab_parallel_jsd
from miles.rollout import ptd_scoring as scoring
from miles.rollout.ptd import score_teacher_missing_ids


@pytest.mark.parametrize("entries", [
    [[-2, 1], [-2, 1]], [[-2, -1]], [[-2, 5]], [[-2, 1.0]], [[-2, True]],
    [[float("nan"), 1]], [[float("inf"), 1]], [[0.01, 1]], [[float("-inf"), 1]],
    [[-0.1, 1], [-0.1, 2]], [[True, 1]],
])
def test_rejects_invalid_teacher_entries(entries):
    with pytest.raises(ValueError):
        scoring.validate_score_entries(entries, 5)


def test_exact_topk_and_sparse_membership():
    entries = [[-1, 1], [-2, 2]]
    assert scoring.validate_score_entries(entries, 5, count=2) == {1: -1., 2: -2.}
    with pytest.raises(ValueError, match="Top-K"):
        scoring.validate_score_entries(entries, 5, count=3)
    with pytest.raises(ValueError, match="omitted"):
        scoring.validate_score_entries(entries, 5, requested=[1, 3])


def test_sparse_duplicates_rejected_before_dict(monkeypatch):
    result = {"meta_info": {"input_token_logprobs": [None, [-2, 3]],
                            "input_token_ids_logprobs": [None, [[-2, 1], [-2, 1]]]}}
    monkeypatch.setattr(scoring.urllib.request, "urlopen", lambda *a, **kw: io.StringIO(json.dumps(result)))
    with pytest.raises(ValueError, match="duplicate"):
        score_teacher_missing_ids({"payload": {}, "url": "http://teacher", "vocab_size": 5,
                                   "response_tokens": [3]}, [[1]], 5)


@pytest.mark.parametrize("ids,logp", [([-2], [-1.]), ([5], [-1.]), ([1., 2.], [-2., -2.]),
                                      ([1, 1], [-2., -2.]), ([1, 2], [-0.1, -0.1]),
                                      ([1], [0.1]), ([1], [float("nan")])])
def test_math_rejects_invalid_sparse_support(ids, logp):
    with pytest.raises(ValueError):
        sparse_vocab_parallel_jsd(torch.zeros(1, 5), torch.tensor([ids]), torch.tensor([logp]))


def test_math_accepts_padding_and_roundoff():
    loss = sparse_vocab_parallel_jsd(torch.zeros(1, 5), torch.tensor([[-1, 1, 2]]),
                                    torch.tensor([[float("nan"), -1., -1.]]))
    assert torch.isfinite(loss).all()
    assert scoring.validate_score_entries([[-0.693146, 1], [-0.693146, 2]], 5)


@pytest.mark.parametrize("asynchronous", [False, True])
@pytest.mark.parametrize("failure,retries", [(429, True), (503, True), (400, False),
                                            ("transport", True), ("abort", True), ("malformed", False)])
def test_transport_retry_policy(monkeypatch, asynchronous, failure, retries):
    calls = []
    good = {"meta_info": {}}
    payload = {"input_ids": [1, 2], "lora_path": None}

    def answer():
        if len(calls) > 1:
            return good
        if isinstance(failure, int):
            if asynchronous:
                raise aiohttp.ClientResponseError(None, (), status=failure, message="secret")
            raise urllib.error.HTTPError("http://secret", failure, "secret", {}, None)
        if failure == "transport":
            raise ConnectionError("secret")
        if failure == "abort":
            return {"meta_info": {"finish_reason": {"type": "abort", "message": "secret"}}}
        return []

    if asynchronous:
        class Response:
            async def __aenter__(self):
                return self

            async def __aexit__(self, *args):
                pass

            def raise_for_status(self):
                self.value = answer()

            async def json(self):
                return self.value

        class Session:
            async def __aenter__(self):
                return self

            async def __aexit__(self, *args):
                pass

            def post(self, url, *, json, timeout):
                calls.append((url, json, timeout.total))
                return Response()

        async def no_sleep(delay):
            pass

        monkeypatch.setattr(scoring.aiohttp, "ClientSession", Session)
        monkeypatch.setattr(scoring.asyncio, "sleep", no_sleep)
        invoke = lambda: asyncio.run(scoring.request_scores_async("http://teacher", payload, 5))
    else:
        def urlopen(request, *, timeout):
            calls.append((request.full_url, json.loads(request.data), timeout))
            return io.StringIO(json.dumps(answer()))

        monkeypatch.setattr(scoring.urllib.request, "urlopen", urlopen)
        monkeypatch.setattr(scoring.time, "sleep", lambda delay: None)
        invoke = lambda: scoring.request_scores("http://teacher", payload, 5)
    if retries:
        assert invoke() == good
        assert len(calls) == 2
        assert calls[0][:2] == calls[1][:2]
        assert 0 < calls[1][2] <= calls[0][2] <= 5
    else:
        with pytest.raises((ValueError, RuntimeError)) as error:
            invoke()
        assert "secret" not in str(error.value)
        assert len(calls) == 1


def test_retries_bounded_by_attempt_count_and_deadline(monkeypatch):
    calls = []

    def fail(*args, **kwargs):
        calls.append(kwargs["timeout"])
        raise TimeoutError("secret")

    monkeypatch.setattr(scoring.urllib.request, "urlopen", fail)
    monkeypatch.setattr(scoring.time, "sleep", lambda delay: None)
    with pytest.raises(RuntimeError):
        scoring.request_scores("http://teacher", {}, 5)
    assert len(calls) == 3
    calls.clear()
    times = iter([0, 0, 6])
    monkeypatch.setattr(scoring.time, "monotonic", lambda: next(times))
    with pytest.raises(RuntimeError):
        scoring.request_scores("http://teacher", {}, 5)
    assert len(calls) == 1
