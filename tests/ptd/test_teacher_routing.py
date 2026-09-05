"""Exercise PTD's actual joint scorer through the production Python proxy."""

import asyncio
import copy
import json
import math
from argparse import Namespace

import httpx
import pytest

from miles.rollout import ptd
from miles.router.router import MilesRouter


def _router_args(**overrides):
    return Namespace(**{
        "miles_router_max_connections": 8,
        "miles_router_timeout": 10,
        "rollout_health_check_interval": 30,
        "miles_router_health_check_failure_threshold": 3,
        **overrides,
    })


@pytest.mark.parametrize("coef,url,miles_router", [
    (0, None, False), (0.05, "http://teacher/generate", False), (0.05, None, True),
])
def test_supported_teacher_routes_and_disabled_ptd(coef, url, miles_router):
    ptd.validate_teacher_route(Namespace(ptd_coef=coef, ptd_teacher_url=url, use_miles_router=miles_router))


def test_reject_default_rust_teacher_route_before_rollout():
    with pytest.raises(ValueError, match="--use-miles-router.*token_ids_logprob_positions and cache_salt"):
        ptd.validate_teacher_route(Namespace(ptd_coef=0.05, ptd_teacher_url=None, use_miles_router=False))


@pytest.mark.parametrize("disable", [True, False, None])
def test_python_router_honors_existing_disable_health_check_flag(disable):
    async def exercise():
        options = {} if disable is None else {"router_disable_health_check": disable}
        router = MilesRouter(_router_args(**options))
        before = asyncio.all_tasks()
        await router._start_background_health_check()
        spawned = asyncio.all_tasks() - before
        try:
            assert len(spawned) == (0 if disable else 1)
        finally:
            for task in spawned:
                task.cancel()
            await asyncio.gather(*spawned, return_exceptions=True)
            await router.client.aclose()

    asyncio.run(exercise())


def test_joint_teacher_rpc_preserves_sparse_ids_cache_salt_media_and_base_route(monkeypatch):
    context = {
        "url": "http://router/generate", "vocab_size": 5, "response_tokens": [2, 3],
        "payload": {
            "input_ids": [4, 1, 2, 3], "sampling_params": {"temperature": 0, "max_new_tokens": 0},
            "return_logprob": True, "logprob_start_len": 1, "lora_path": None,
            "image_data": ["data:image/png;base64,c2FtZS1pbWFnZQ=="],
        },
    }
    original = copy.deepcopy(context)
    student = [[1, 2], [1, 2]]
    received = []
    q = [0.4, 0.3, 0.15, 0.1, 0.05]

    async def roundtrip(url, payload, timeout):
        router = MilesRouter(_router_args(router_disable_health_check=True))
        router.worker_request_counts["http://worker"] = 0
        encoded = json.dumps(payload).encode()

        def worker(request):
            assert request.url == "http://worker/generate"
            assert request.content == encoded
            body = json.loads(request.content)
            received.append(body)
            meta = {
                "input_token_logprobs": [[None, 1], [math.log(q[2]), 2], [math.log(q[3]), 3]],
                "input_top_logprobs": [None] + [[[math.log(q[i]), i] for i in [0, 1]]] * 2,
                "input_token_ids_logprobs": [None] + [
                    [[math.log(q[i]), i] for i in row] for row in body["token_ids_logprob_positions"][1:]
                ],
            }
            return httpx.Response(200, json={"meta_info": meta})

        await router.client.aclose()
        async with httpx.AsyncClient(transport=httpx.MockTransport(worker)) as outgoing:
            router.client = outgoing
            async with httpx.AsyncClient(transport=httpx.ASGITransport(app=router.app)) as incoming:
                response = await incoming.post(url, content=encoded, headers={"Content-Type": "application/json"},
                                               timeout=timeout)
                response.raise_for_status()
                assert router.worker_request_counts == {"http://worker": 0}
                return response.json()

    monkeypatch.setattr(ptd, "request_scores", lambda *args: asyncio.run(roundtrip(*args)))
    ids, values = ptd.score_teacher_joint(context, student, 2, 10)
    assert ids == [[0, 1, 2, -1], [0, 1, 2, -1]]
    assert values == [[math.log(q[i]) for i in [0, 1, 2]] + [-math.inf]] * 2
    assert len(received) == 1
    assert received[0]["token_ids_logprob_positions"] == [[], *student]
    assert len(received[0]["cache_salt"]) == 32
    assert received[0]["top_logprobs_num"] == 2
    for key, value in original["payload"].items():
        assert received[0][key] == value
    assert context == original
