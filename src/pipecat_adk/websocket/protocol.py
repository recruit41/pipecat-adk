"""Wire protocol for the Pipecat <-> external LLM WebSocket bridge.

This module is the single source of truth for the bridge protocol.  The
TypeScript implementation in ``js-bridge/src/protocol.ts`` mirrors these
constants and shapes; keep the two in sync.

Transport
---------
Messages are UTF-8 JSON text frames over a WebSocket connection.  Every
message is an envelope with a ``v`` (protocol version) and a ``type`` field.

Direction
---------
- Python -> JS: ``session.start``, ``turn.run``, ``turn.cancel``,
  ``turn.completed``, ``session.end``, ``ping``.
- JS -> Python: ``session.ready``, ``frame``, ``turn.usage``, ``state.delta``,
  ``error``, ``pong``.

A turn
------
``WebSocketLLMService`` sends one ``turn.run`` per ``VqlContextFrame``.  The
JS component answers with a stream of ``frame`` messages that always begins
with ``VqlLLMFullResponseStartFrame`` and ends with
``VqlLLMFullResponseEndFrame`` — the end frame is the stream terminator.
``turn.cancel`` aborts an in-flight turn (pipeline interruption).
"""

from __future__ import annotations

import json
from typing import Any, Optional

from pipecat.frames.frames import (
    ErrorFrame,
    Frame,
    FunctionCallFromLLM,
)
from pipecat.processors.frame_processor import FrameDirection

from ..frames import (
    VqlFunctionCallInProgressFrame,
    VqlFunctionCallResultFrame,
    VqlFunctionCallsStartedFrame,
    VqlLLMFullResponseEndFrame,
    VqlLLMFullResponseStartFrame,
    VqlLLMTextFrame,
)

PROTOCOL_VERSION = 1

# ---------------------------------------------------------------------------
# Message types
# ---------------------------------------------------------------------------

# Python -> JS
MSG_SESSION_START = "session.start"
MSG_TURN_RUN = "turn.run"
MSG_TURN_CANCEL = "turn.cancel"
MSG_TURN_COMPLETED = "turn.completed"
MSG_SESSION_END = "session.end"
MSG_PING = "ping"

# JS -> Python
MSG_SESSION_READY = "session.ready"
MSG_FRAME = "frame"
MSG_TURN_USAGE = "turn.usage"
MSG_STATE_DELTA = "state.delta"
MSG_ERROR = "error"
MSG_PONG = "pong"

# ---------------------------------------------------------------------------
# Frame names — the ``frame`` field of an MSG_FRAME envelope (JS -> Python)
# ---------------------------------------------------------------------------

FRAME_RESPONSE_START = "VqlLLMFullResponseStartFrame"
FRAME_TEXT = "VqlLLMTextFrame"
FRAME_FUNCTION_CALLS_STARTED = "VqlFunctionCallsStartedFrame"
FRAME_FUNCTION_CALL_IN_PROGRESS = "VqlFunctionCallInProgressFrame"
FRAME_FUNCTION_CALL_RESULT = "VqlFunctionCallResultFrame"
FRAME_RESPONSE_END = "VqlLLMFullResponseEndFrame"

# Pipeline direction(s) each output frame is pushed in.  Function-call frames
# go both ways (upstream so STTMuteFilter can mute the mic, downstream so the
# UI can render a "thinking" state) — this mirrors AdkLLMService exactly.
_BOTH = (FrameDirection.UPSTREAM, FrameDirection.DOWNSTREAM)
_DOWN = (FrameDirection.DOWNSTREAM,)

FRAME_DIRECTIONS: dict[str, tuple[FrameDirection, ...]] = {
    FRAME_RESPONSE_START: _DOWN,
    FRAME_TEXT: _DOWN,
    FRAME_FUNCTION_CALLS_STARTED: _BOTH,
    FRAME_FUNCTION_CALL_IN_PROGRESS: _BOTH,
    FRAME_FUNCTION_CALL_RESULT: _BOTH,
    FRAME_RESPONSE_END: _DOWN,
}


class ProtocolError(Exception):
    """Raised when a message cannot be parsed or violates the protocol."""


# ---------------------------------------------------------------------------
# Encoding — Python -> JS
# ---------------------------------------------------------------------------


def _dumps(obj: dict[str, Any]) -> str:
    return json.dumps(obj, separators=(",", ":"), ensure_ascii=False)


def encode_session_start(
    session: Optional[dict[str, Any]] = None,
    metadata: Optional[dict[str, Any]] = None,
) -> str:
    """Handshake sent by the Python side as soon as the socket opens."""
    return _dumps(
        {
            "v": PROTOCOL_VERSION,
            "type": MSG_SESSION_START,
            "session": session or {},
            "metadata": metadata or {},
        }
    )


def encode_turn_run(
    turn_id: str,
    payload: dict[str, Any],
    state_delta: Optional[dict[str, Any]] = None,
) -> str:
    """Request the JS component to generate a response for ``turn_id``."""
    return _dumps(
        {
            "v": PROTOCOL_VERSION,
            "type": MSG_TURN_RUN,
            "turn_id": turn_id,
            "payload": payload,
            "state_delta": state_delta,
        }
    )


def encode_turn_cancel(turn_id: str) -> str:
    """Abort an in-flight turn (the pipeline was interrupted)."""
    return _dumps({"v": PROTOCOL_VERSION, "type": MSG_TURN_CANCEL, "turn_id": turn_id})


def encode_turn_completed(turn_id: str, text: str, interrupted: bool) -> str:
    """Report what the user actually heard once a turn finishes.

    The JS component uses this to write its own ``[HEARD]`` provenance, the
    same way AdkLLMService writes a ``[HEARD]`` event into the ADK session.
    """
    return _dumps(
        {
            "v": PROTOCOL_VERSION,
            "type": MSG_TURN_COMPLETED,
            "turn_id": turn_id,
            "text": text,
            "interrupted": interrupted,
        }
    )


def encode_session_end() -> str:
    """Signal a graceful shutdown before closing the socket."""
    return _dumps({"v": PROTOCOL_VERSION, "type": MSG_SESSION_END})


def encode_ping(ts: float) -> str:
    """Application-level heartbeat — detects a hung JS event loop."""
    return _dumps({"v": PROTOCOL_VERSION, "type": MSG_PING, "ts": ts})


# ---------------------------------------------------------------------------
# Decoding — JS -> Python
# ---------------------------------------------------------------------------


def decode_message(raw: str | bytes) -> dict[str, Any]:
    """Parse and validate a raw WebSocket message into an envelope dict.

    Raises:
        ProtocolError: if the payload is not JSON, not an object, has the
            wrong protocol version, or is missing a ``type``.
    """
    if isinstance(raw, bytes):
        try:
            raw = raw.decode("utf-8")
        except UnicodeDecodeError as e:
            raise ProtocolError(f"message is not valid UTF-8: {e}") from e

    try:
        envelope = json.loads(raw)
    except json.JSONDecodeError as e:
        raise ProtocolError(f"message is not valid JSON: {e}") from e

    if not isinstance(envelope, dict):
        raise ProtocolError(f"message must be a JSON object, got {type(envelope).__name__}")

    version = envelope.get("v")
    if version != PROTOCOL_VERSION:
        raise ProtocolError(
            f"unsupported protocol version {version!r} (expected {PROTOCOL_VERSION})"
        )

    if not envelope.get("type"):
        raise ProtocolError("message is missing 'type'")

    return envelope


def build_output_frame(envelope: dict[str, Any]) -> tuple[Frame, tuple[FrameDirection, ...]]:
    """Reconstruct a Vql frame (and its push direction) from an MSG_FRAME envelope.

    Returns:
        A ``(frame, directions)`` pair.  ``directions`` is the tuple of
        FrameDirection values the frame must be pushed in.

    Raises:
        ProtocolError: if the frame name is unknown or required fields are missing.
    """
    name = envelope.get("frame")
    if name not in FRAME_DIRECTIONS:
        raise ProtocolError(f"unknown output frame {name!r}")

    turn_id = envelope.get("turn_id")
    if not turn_id:
        raise ProtocolError(f"frame {name!r} is missing 'turn_id'")

    data = envelope.get("data") or {}
    invocation_id = data.get("invocation_id", "")
    directions = FRAME_DIRECTIONS[name]

    if name == FRAME_RESPONSE_START:
        return (
            VqlLLMFullResponseStartFrame(turn_id=turn_id, invocation_id=invocation_id),
            directions,
        )

    if name == FRAME_RESPONSE_END:
        return (
            VqlLLMFullResponseEndFrame(turn_id=turn_id, invocation_id=invocation_id),
            directions,
        )

    if name == FRAME_TEXT:
        text = data.get("text")
        if text is None:
            raise ProtocolError("VqlLLMTextFrame is missing 'text'")
        return (
            VqlLLMTextFrame(text=text, turn_id=turn_id, invocation_id=invocation_id),
            directions,
        )

    if name == FRAME_FUNCTION_CALLS_STARTED:
        raw_calls = data.get("function_calls") or []
        calls = [_build_function_call_from_llm(c) for c in raw_calls]
        return (
            VqlFunctionCallsStartedFrame(
                function_calls=calls, turn_id=turn_id, invocation_id=invocation_id
            ),
            directions,
        )

    if name == FRAME_FUNCTION_CALL_IN_PROGRESS:
        tool_call_id, function_name = _require_call_ids(data, name)
        return (
            VqlFunctionCallInProgressFrame(
                function_name=function_name,
                tool_call_id=tool_call_id,
                arguments=data.get("arguments"),
                turn_id=turn_id,
                invocation_id=invocation_id,
            ),
            directions,
        )

    # FRAME_FUNCTION_CALL_RESULT
    tool_call_id, function_name = _require_call_ids(data, name)
    return (
        VqlFunctionCallResultFrame(
            function_name=function_name,
            tool_call_id=tool_call_id,
            arguments=None,
            result=data.get("result") or {},
            turn_id=turn_id,
            invocation_id=invocation_id,
        ),
        directions,
    )


def build_error_frame(envelope: dict[str, Any]) -> ErrorFrame:
    """Build an ErrorFrame from an MSG_ERROR envelope."""
    data = envelope.get("data") or {}
    message = data.get("message") or "remote LLM bridge error"
    return ErrorFrame(error=str(message), fatal=bool(data.get("fatal", False)))


def _require_call_ids(data: dict[str, Any], frame_name: str) -> tuple[str, str]:
    tool_call_id = data.get("tool_call_id")
    function_name = data.get("function_name")
    if not tool_call_id or not function_name:
        raise ProtocolError(
            f"{frame_name} requires 'tool_call_id' and 'function_name'"
        )
    return tool_call_id, function_name


def _build_function_call_from_llm(call: dict[str, Any]) -> FunctionCallFromLLM:
    tool_call_id = call.get("tool_call_id")
    function_name = call.get("function_name")
    if not tool_call_id or not function_name:
        raise ProtocolError(
            "function_calls entry requires 'tool_call_id' and 'function_name'"
        )
    return FunctionCallFromLLM(
        tool_call_id=tool_call_id,
        function_name=function_name,
        arguments=call.get("arguments") or {},
        context=None,
    )
