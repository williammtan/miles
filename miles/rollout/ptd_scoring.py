"""Validation and deadline-bounded transport for privileged teacher scores."""

import asyncio
import hashlib
import http.client
import json
import math
import time
import urllib.error
import urllib.request
import uuid
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


def _request_summary(payload):
    """Identify a failed scoring context without exposing its text, media or URL."""
    ids = payload.get("input_ids", [])
    context_hash = hashlib.sha256(json.dumps(ids, separators=(",", ":")).encode()).hexdigest()[:20]
    positions = payload.get("token_ids_logprob_positions", [])
    return f"context={context_hash}, input_tokens={len(ids)}, scoring_rows={len(positions)}"


def _retry_delay(error, attempt, deadline, request_summary=""):
    remaining = deadline - time.monotonic()
    if not _retryable(error) or attempt + 1 >= _MAX_ATTEMPTS or remaining <= 0:
        category = ("nonretryable_error" if not _retryable(error) else
                    "deadline_exhausted" if remaining <= 0 else "attempts_exhausted")
        status = (error.status if isinstance(error, aiohttp.ClientResponseError) else
                  error.code if isinstance(error, urllib.error.HTTPError) else None)
        # Exception messages/URLs and response bodies may contain private payloads.
        raise RuntimeError(
            f"PTD teacher scoring failed: {category}, error_type={type(error).__name__}, "
            f"http_status={status}, attempts={attempt + 1}/{_MAX_ATTEMPTS}, {request_summary}"
        ) from None
    return min(0.25 * 2**attempt, remaining)


async def request_scores_async(url, payload, timeout):
    """Retry transient failures within one deadline, refreshing salted teacher caches."""
    deadline = time.monotonic() + timeout
    summary = _request_summary(payload)
    async with aiohttp.ClientSession() as session:
        for attempt in range(_MAX_ATTEMPTS):
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                raise RuntimeError("PTD teacher scoring exceeded its deadline")
            attempt_payload = _attempt_payload(payload, attempt)
            try:
                async with session.post(url, json=attempt_payload, timeout=aiohttp.ClientTimeout(total=remaining)) as response:
                    response.raise_for_status()
                    result = await response.json()
                return _check_response(result)
            except (ValueError, aiohttp.ContentTypeError):
                raise ValueError("PTD teacher returned malformed score JSON") from None
            except Exception as error:
                await asyncio.sleep(_retry_delay(error, attempt, deadline, summary))


def request_scores(url, payload, timeout):
    """Synchronous equivalent used inside the training loss."""
    deadline = time.monotonic() + timeout
    summary = _request_summary(payload)
    for attempt in range(_MAX_ATTEMPTS):
        remaining = deadline - time.monotonic()
        if remaining <= 0:
            raise RuntimeError("PTD teacher scoring exceeded its deadline")
        attempt_payload = _attempt_payload(payload, attempt)
        request = urllib.request.Request(
            url, data=json.dumps(attempt_payload).encode(), headers={"Content-Type": "application/json"}, method="POST",
        )
        try:
            with urllib.request.urlopen(request, timeout=remaining) as response:
                result = json.load(response)
            return _check_response(result)
        except ValueError:
            raise ValueError("PTD teacher returned malformed score JSON") from None
        except Exception as error:
            time.sleep(_retry_delay(error, attempt, deadline, summary))


def _attempt_payload(payload, attempt):
    """A timed-out attempt may still cache hybrid state; salted retries isolate it."""
    if attempt > 0 and "cache_salt" in payload:
        return {**payload, "cache_salt": uuid.uuid4().hex}
    return payload
