import json
import logging
import os
import re
import time
import uuid

from fastapi import Request
from fastapi.responses import JSONResponse, StreamingResponse
from starlette.responses import Response

from miles.rollout.session.linear_trajectory import SessionRegistry
from miles.rollout.session.session_errors import (
    SessionError,
    SessionNotFoundError,
    TokenizationError,
    UpstreamResponseError,
)
from miles.rollout.session.session_types import GetSessionResponse, SessionRecord
from miles.utils.chat_template_utils import get_tito_tokenizer
from miles.utils.processing_utils import load_tokenizer

logger = logging.getLogger(__name__)

# --- Session hardening ported from the mercor 122B run's deployed miles tree ---
# (weka:/osmosis-qwen36-harbor/miles): Qwen text tool-call promotion, forced
# sampling overrides, and a graceful context-budget stop. See that tree's
# sessions.py for the original; only the pieces needed for agentic mercor
# rollouts are ported here.

_TOOL_CALL_RE = re.compile(
    r"<tool_call>\s*<function=([A-Za-z0-9_.:-]+)>\s*(.*?)\s*</function>\s*</tool_call>",
    re.DOTALL,
)
_TOOL_PARAM_RE = re.compile(
    r"<parameter=([A-Za-z0-9_.:-]+)>\s*(.*?)\s*</parameter>",
    re.DOTALL,
)


def _parse_tool_arg(value: str):
    value = value.strip()
    try:
        return json.loads(value)
    except json.JSONDecodeError:
        return value


def _extract_qwen_text_tool_calls(content: str) -> tuple[str, list[dict] | None]:
    if not isinstance(content, str) or "<tool_call>" not in content:
        return content, None

    tool_calls = []
    spans = []
    for match in _TOOL_CALL_RE.finditer(content):
        name = match.group(1).strip()
        body = match.group(2)
        args = {
            param_match.group(1).strip(): _parse_tool_arg(param_match.group(2))
            for param_match in _TOOL_PARAM_RE.finditer(body)
        }
        tool_calls.append(
            {
                "id": f"call_{uuid.uuid4().hex[:24]}",
                "type": "function",
                "function": {
                    "name": name,
                    "arguments": json.dumps(args, ensure_ascii=False),
                },
            }
        )
        spans.append(match.span())

    if not tool_calls:
        return content, None

    pieces = []
    pos = 0
    for start, end in spans:
        pieces.append(content[pos:start])
        pos = end
    pieces.append(content[pos:])
    return "".join(pieces).rstrip(), tool_calls


def _normalize_qwen_text_tool_calls(response: dict, *, enabled: bool = True) -> None:
    """Promote Qwen chat-template tool markup to OpenAI tool_calls."""
    if not enabled:
        return
    choices = response.get("choices") or []
    if not choices:
        return
    message = choices[0].get("message") or {}
    if message.get("tool_calls"):
        return
    content = message.get("content")
    if not isinstance(content, str):
        return

    stripped_content, tool_calls = _extract_qwen_text_tool_calls(content)
    if not tool_calls:
        return
    message["content"] = stripped_content
    message["tool_calls"] = tool_calls
    choices[0]["finish_reason"] = "tool_calls"


def _json_result(result: dict, response: dict) -> dict:
    headers = dict(result["headers"])
    headers["content-type"] = "application/json"
    return {
        **result,
        "response_body": json.dumps(response, ensure_ascii=False).encode(),
        "headers": headers,
    }


def _int_env(name: str, default: int = 0) -> int:
    value = os.environ.get(name)
    if not value:
        return default
    try:
        parsed = int(value)
    except ValueError:
        logger.warning("[session-server] Ignoring invalid %s=%r", name, value)
        return default
    return max(parsed, 0)


def _float_env(name: str) -> float | None:
    value = os.environ.get(name)
    if not value:
        return None
    try:
        return float(value)
    except ValueError:
        logger.warning("[session-server] Ignoring invalid %s=%r", name, value)
        return None


def _apply_sampling_overrides(request_body: dict) -> None:
    temperature = _float_env("MILES_SESSION_FORCE_TEMPERATURE")
    if temperature is not None:
        request_body["temperature"] = temperature
    top_p = _float_env("MILES_SESSION_FORCE_TOP_P")
    if top_p is not None:
        request_body["top_p"] = top_p


def _stream_event(payload: dict) -> bytes:
    return f"data: {json.dumps(payload, separators=(',', ':'))}\n\n".encode()


def _build_streaming_chat_response(response: dict, result: dict) -> StreamingResponse:
    """Convert a non-stream OpenAI response into a minimal SSE stream."""
    choice = response.get("choices", [{}])[0]
    message = choice.get("message") or {}
    index = choice.get("index", 0)

    base = {
        "id": response.get("id", "chatcmpl-miles-session"),
        "object": "chat.completion.chunk",
        "created": response.get("created", int(time.time())),
        "model": response.get("model", "model"),
    }
    if "system_fingerprint" in response:
        base["system_fingerprint"] = response["system_fingerprint"]

    def events():
        first = {
            **base,
            "choices": [
                {
                    "index": index,
                    "delta": {"role": message.get("role", "assistant")},
                    "finish_reason": None,
                }
            ],
        }
        yield _stream_event(first)

        delta = {}
        content = message.get("content")
        if content:
            delta["content"] = content
        if message.get("tool_calls") is not None:
            delta["tool_calls"] = message["tool_calls"]
        if message.get("function_call") is not None:
            delta["function_call"] = message["function_call"]

        if delta:
            yield _stream_event(
                {
                    **base,
                    "choices": [
                        {
                            "index": index,
                            "delta": delta,
                            "finish_reason": None,
                        }
                    ],
                }
            )

        yield _stream_event(
            {
                **base,
                "choices": [
                    {
                        "index": index,
                        "delta": {},
                        "finish_reason": choice.get("finish_reason"),
                    }
                ],
            }
        )
        yield b"data: [DONE]\n\n"

    headers = {
        k: v
        for k, v in result["headers"].items()
        if k.lower() not in ("content-length", "transfer-encoding", "content-encoding", "content-type")
    }
    return StreamingResponse(
        events(),
        status_code=result["status_code"],
        headers=headers,
        media_type="text/event-stream",
    )


def _build_context_limit_response(
    *,
    client_requested_stream: bool,
    model: str,
    prompt_tokens: int,
    max_prompt_tokens: int,
) -> Response:
    response = {
        "id": f"chatcmpl-miles-session-limit-{uuid.uuid4().hex[:12]}",
        "object": "chat.completion",
        "created": int(time.time()),
        "model": model or "model",
        "choices": [
            {
                "index": 0,
                "message": {
                    "role": "assistant",
                    "content": (
                        "Stopping here because the session context budget was reached. "
                        "Use the current repository state as the final answer."
                    ),
                },
                "finish_reason": "stop",
            }
        ],
        "usage": {
            "prompt_tokens": prompt_tokens,
            "completion_tokens": 0,
            "total_tokens": prompt_tokens,
        },
    }
    result = {"status_code": 200, "headers": {}, "response_body": json.dumps(response).encode()}
    if client_requested_stream:
        return _build_streaming_chat_response(response, result)
    return JSONResponse(status_code=200, content=response)


def setup_session_routes(app, backend, args):
    hf_checkpoint = getattr(args, "hf_checkpoint", None)
    if not hf_checkpoint:
        logger.info("[session] Skipping session routes (hf_checkpoint not set).")
        return

    session_server_instance_id = getattr(args, "session_server_instance_id", None)
    max_prompt_tokens = _int_env("MILES_SESSION_MAX_PROMPT_TOKENS", 0)
    if max_prompt_tokens:
        logger.info("[session-server] Max prompt-token cutoff enabled: %d", max_prompt_tokens)

    tokenizer = load_tokenizer(
        hf_checkpoint, chat_template_path=getattr(args, "chat_template_path", None), trust_remote_code=True
    )

    tito_tokenizer = get_tito_tokenizer(
        tokenizer,
        tokenizer_type=getattr(args, "tito_model", "default"),
        chat_template_kwargs=getattr(args, "apply_chat_template_kwargs", None),
        allowed_append_roles=getattr(args, "tito_allowed_append_roles", None),
    )

    registry = SessionRegistry(args, tokenizer, tito_tokenizer=tito_tokenizer)

    @app.get("/health")
    async def health():
        body = {"status": "ok"}
        if session_server_instance_id is not None:
            body["session_server_instance_id"] = session_server_instance_id
        return body

    # --- DEBUG: track in-flight chat_completions ---
    _inflight_chat = {"count": 0}

    @app.middleware("http")
    async def debug_request_logger(request: Request, call_next):
        client = request.client
        client_info = f"{client.host}:{client.port}" if client else "unknown"
        logger.info(
            f"[session-server] REQUEST ARRIVED: {request.method} {request.url.path} from={client_info} inflight_chat={_inflight_chat['count']}"
        )
        t0 = time.time()
        response = await call_next(request)
        elapsed = time.time() - t0
        logger.info(
            f"[session-server] REQUEST DONE: {request.method} {request.url.path} status={response.status_code} elapsed={elapsed:.3f}s from={client_info}"
        )
        return response

    @app.exception_handler(SessionError)
    async def session_error_handler(request: Request, exc: SessionError):
        return JSONResponse(status_code=exc.status_code, content={"error": str(exc)})

    @app.post("/sessions")
    async def create_session():
        session_id = registry.create_session()
        return {"session_id": session_id}

    @app.get("/sessions/{session_id}")
    async def get_session(session_id: str):
        session = registry.get_session(session_id)
        metadata = {}
        try:
            mismatch = registry.compute_session_mismatch(session)
        except TokenizationError:
            logger.exception("Failed to compute tito_session_mismatch for session %s", session_id)
            mismatch = None
        if mismatch is not None:
            metadata["tito_session_mismatch"] = mismatch
        metadata["accumulated_token_ids"] = session.token_ids
        metadata["max_trim_tokens"] = registry.tito_tokenizer.max_trim_tokens
        return GetSessionResponse(
            session_id=session_id,
            records=session.records,
            metadata=metadata,
        )

    @app.delete("/sessions/{session_id}")
    async def delete_session(session_id: str):
        session = registry.get_session(session_id)
        if session.closing:
            raise SessionNotFoundError(f"session not found: session_id={session_id}")
        session.closing = True
        logger.debug(
            f"[session-server] DELETE waiting for lock: session={session_id} lock_locked={session.lock.locked()}"
        )
        await session.lock.acquire()
        logger.debug(f"[session-server] DELETE acquired lock: session={session_id}")
        try:
            registry.remove_session(session_id)
        finally:
            session.lock.release()
        return Response(status_code=204)

    @app.post("/sessions/{session_id}/v1/chat/completions")
    async def chat_completions(request: Request, session_id: str):
        """Proxy a chat completion through SGLang with TITO token tracking.

        Flow: prepare pretokenized input_ids (lock held briefly) → inject
        SGLang flags → proxy to backend (NO lock) → validate response →
        update trajectory checkpoint (lock held briefly) → append session record.

        The lock is NOT held during the slow proxy call to avoid blocking
        DELETE/other operations when the agent disconnects mid-request.
        """
        _inflight_chat["count"] += 1
        try:
            session = registry.get_session(session_id)
            if session.closing:
                raise SessionNotFoundError(f"session not found: session_id={session_id}")

            # --- Phase 1: prepare request (lock held briefly) ---
            async with session.lock:
                # Double-check: session may have been marked closing while waiting for lock.
                if session.closing:
                    raise SessionNotFoundError(f"session not found: session_id={session_id}")

                body = await request.body()
                request_body = json.loads(body) if body else {}

                # The proxy path is non-streaming; if the agent asked for SSE we
                # proxy non-stream and convert the final response below.
                client_requested_stream = bool(request_body.get("stream"))
                request_body["stream"] = False
                _apply_sampling_overrides(request_body)

                # TITO token tracking requires Miles-owned input_ids plus SGLang
                # output-token metadata:
                #   logprobs=True     → populates meta_info.output_token_logprobs
                #   return_meta_info  → wraps the above in choice.meta_info
                # Both flags are hardcoded (not set default) to prevent agent-side
                # overrides from breaking the token accumulation invariants.
                request_body["logprobs"] = True
                request_body["return_meta_info"] = True
                if getattr(args, "use_rollout_routing_replay", False):
                    request_body["return_routed_experts"] = True
                if getattr(args, "use_rollout_indexer_replay", False):
                    request_body["return_indexer_topk"] = True
                # Must be False so stop-token text is trimmed from assistant
                # message content; token IDs are still taken from logprobs below.
                request_body["no_stop_trim"] = False

                request_messages = request_body.get("messages", [])
                prompt_token_ids = session.prepare_pretokenized(
                    request_messages,
                    tools=request_body.get("tools"),
                    tito_tokenizer=registry.tito_tokenizer,
                )
                request_body["input_ids"] = prompt_token_ids
                logger.debug(
                    "Using TITO input_ids: %d tokens",
                    len(prompt_token_ids),
                )

                # Graceful context-budget stop: tell the agent to wrap up instead
                # of letting the engine 400 on an over-long prompt (which agents
                # burn retries on). The turn is NOT recorded — the session keeps
                # only the trainable turns that actually ran.
                if max_prompt_tokens and len(prompt_token_ids) > max_prompt_tokens:
                    logger.info(
                        "[session-server] Returning context-limit stop: session=%s "
                        "prompt_tokens=%d max_prompt_tokens=%d",
                        session_id,
                        len(prompt_token_ids),
                        max_prompt_tokens,
                    )
                    return _build_context_limit_response(
                        client_requested_stream=client_requested_stream,
                        model=request_body.get("model", "model"),
                        prompt_tokens=len(prompt_token_ids),
                        max_prompt_tokens=max_prompt_tokens,
                    )

                body = json.dumps(request_body).encode()
                expected_num_assistant = session.num_assistant
            # --- lock released here ---

            # --- Phase 2: proxy to SGLang (NO lock held) ---
            result = await backend.do_proxy(request, "v1/chat/completions", body=body)

            # If SGLang returned a non-200 error (e.g. 400 for context too long),
            # pass it through to the agent without recording — the agent can retry
            # or handle the error.
            if result["status_code"] != 200:
                return backend.build_proxy_response(result)

            response = json.loads(result["response_body"])

            # If the engine's tool parser did not fire and the model emitted
            # Qwen-style <tool_call> text, promote it to OpenAI tool_calls so
            # tool-calling agents (litellm) keep working. Mutates message in
            # place so the recorded session state stays consistent.
            _normalize_qwen_text_tool_calls(response, enabled=bool(request_body.get("tools")))
            result = _json_result(result, response)

            def _client_response():
                if client_requested_stream:
                    return _build_streaming_chat_response(response, result)
                return backend.build_proxy_response(result)

            choice = response.get("choices", [{}])[0]

            meta_info = choice.get("meta_info")
            if not isinstance(meta_info, dict) or "output_token_logprobs" not in meta_info:
                raise UpstreamResponseError(
                    "meta_info and output_token_logprobs must be in choice (requires logprobs=True)"
                )
            assistant_message = choice.get("message", {})
            if assistant_message.get("content") is None:
                raise UpstreamResponseError(
                    "assistant message content is None, when tool call parser failed SGLang should still return "
                    "an empty content rather than None. Please check your modified SGLang version."
                )

            output_token_logprobs = meta_info["output_token_logprobs"]
            completion_tokens = meta_info["completion_tokens"]

            actual_output_logprobs_len = len(output_token_logprobs)
            if actual_output_logprobs_len != completion_tokens:
                raise UpstreamResponseError(
                    "invalid chat completion response: "
                    f"len(output_token_logprobs)={actual_output_logprobs_len} "
                    f"!= completion_tokens={completion_tokens}. "
                    f"Please check whether you use the correct SGLang branch which has fix the tokenizer batch decode issue."
                )

            completion_token_ids = [t[1] for t in output_token_logprobs]

            # --- Phase 3: update state (lock held briefly) ---
            async with session.lock:
                if session.closing:
                    logger.warning(f"Session {session_id} closed during proxy, skipping state update")
                    return _client_response()

                if session.num_assistant != expected_num_assistant:
                    logger.warning(
                        f"Session {session_id} state changed during proxy "
                        f"(expected num_assistant={expected_num_assistant}, "
                        f"got {session.num_assistant}), skipping state update"
                    )
                    return _client_response()

                session.update_pretokenized_state(
                    request_messages,
                    assistant_message,
                    prompt_token_ids=prompt_token_ids,
                    completion_token_ids=completion_token_ids,
                    max_trim_tokens=registry.tito_tokenizer.max_trim_tokens,
                )

                record = SessionRecord(
                    timestamp=time.time(),
                    method=request.method,
                    path="/v1/chat/completions",
                    status_code=result["status_code"],
                    request=request_body,
                    response=response,
                )
                session.append_record(record)
            # --- lock released here ---

            return _client_response()
        finally:
            _inflight_chat["count"] -= 1

    @app.api_route("/sessions/{session_id}/{path:path}", methods=["GET", "POST", "PUT", "DELETE", "PATCH"])
    async def session_proxy(request: Request, session_id: str, path: str):
        result = await backend.do_proxy(request, path)
        return backend.build_proxy_response(result)
