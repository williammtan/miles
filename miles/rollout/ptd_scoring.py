"""Validation and deadline-bounded transport for privileged teacher scores."""

import asyncio
import http.client
import json
import math
import time
import urllib.error
import urllib.request
from numbers import Real

import aiohttp


_MASS_TOLERANCE = 1e-5
_MAX_ATTEMPTS = 3


class _ServerAbort(Exception):
    pass


def validate_score_entries(entries, vocab_size, *, count=None, requested=None):
    """Validate before constructing a map so duplicate token IDs cannot disappear."""
    if not isinstance(vocab_size, int) or vocab_size <= 0:
        raise ValueError("PTD scoring requires a positive vocabulary size")
    if not isinstance(entries, list) or (count is not None and len(entries) != count):
        raise ValueError("PTD teacher did not return the requested Top-K width")
    result = {}
    for entry in entries:
        if not isinstance(entry, (list, tuple)) or len(entry) < 2:
            raise ValueError("PTD teacher returned a malformed score entry")
        lp, token = entry[:2]
        if type(token) is not int or not 0 <= token < vocab_size or token in result:
            raise ValueError("PTD teacher returned invalid or duplicate token IDs")
        if isinstance(lp, bool) or not isinstance(lp, Real) or not math.isfinite(lp) or lp > 0:
            raise ValueError("PTD teacher returned an invalid log probability")
        result[token] = float(lp)
    if math.fsum(math.exp(lp) for lp in result.values()) > 1 + _MASS_TOLERANCE:
        raise ValueError("PTD teacher probability mass exceeds one")
    if requested is not None and (len(requested) != len(set(requested)) or set(requested) != set(result)):
        raise ValueError("PTD teacher sparse scorer omitted or added IDs")
    return result


def _check_response(result):
    if not isinstance(result, dict) or not isinstance(result.get("meta_info"), dict):
        raise ValueError("PTD teacher returned malformed score metadata")
    reason = result["meta_info"].get("finish_reason")
    if isinstance(reason, dict) and reason.get("type") == "abort":
        raise _ServerAbort()
    return result


def _retryable(error):
    if isinstance(error, (aiohttp.ClientResponseError, urllib.error.HTTPError)):
        status = error.status if isinstance(error, aiohttp.ClientResponseError) else error.code
        return status == 429 or 500 <= status < 600
    return isinstance(error, (_ServerAbort, aiohttp.ClientConnectionError, aiohttp.ClientPayloadError,
                              urllib.error.URLError, http.client.HTTPException, TimeoutError, ConnectionError))


def _retry_delay(error, attempt, deadline):
    remaining = deadline - time.monotonic()
    if not _retryable(error) or attempt + 1 >= _MAX_ATTEMPTS or remaining <= 0:
        raise RuntimeError("PTD teacher scoring failed within its retry deadline") from None
    return min(0.25 * 2**attempt, remaining)


async def request_scores_async(url, payload, timeout):
    """Retry only transient failures, with one total deadline for identical requests."""
    deadline = time.monotonic() + timeout
    async with aiohttp.ClientSession() as session:
        for attempt in range(_MAX_ATTEMPTS):
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                raise RuntimeError("PTD teacher scoring exceeded its deadline")
            try:
                async with session.post(url, json=payload, timeout=aiohttp.ClientTimeout(total=remaining)) as response:
                    response.raise_for_status()
                    result = await response.json()
                return _check_response(result)
            except (ValueError, aiohttp.ContentTypeError):
                raise ValueError("PTD teacher returned malformed score JSON") from None
            except Exception as error:
                await asyncio.sleep(_retry_delay(error, attempt, deadline))


def request_scores(url, payload, timeout):
    """Synchronous equivalent used inside the training loss."""
    deadline = time.monotonic() + timeout
    request = urllib.request.Request(
        url, data=json.dumps(payload).encode(), headers={"Content-Type": "application/json"}, method="POST",
    )
    for attempt in range(_MAX_ATTEMPTS):
        remaining = deadline - time.monotonic()
        if remaining <= 0:
            raise RuntimeError("PTD teacher scoring exceeded its deadline")
        try:
            with urllib.request.urlopen(request, timeout=remaining) as response:
                result = json.load(response)
            return _check_response(result)
        except ValueError:
            raise ValueError("PTD teacher returned malformed score JSON") from None
        except Exception as error:
            time.sleep(_retry_delay(error, attempt, deadline))
