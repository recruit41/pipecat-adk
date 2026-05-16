"""Configurable in-process WebSocket server speaking the bridge protocol.

This is a Python stand-in for the JavaScript LLM component.  Unlike the real
Bun bridge it is *deliberately* able to misbehave — stay silent, return
errors, send malformed frames, drop the connection — so the hardening of
``WebSocketBridgeClient`` (timeouts, reconnection, malformed-message
tolerance) can be exercised deterministically without a subprocess.

For protocol round-trip tests against the *real* implementation, use the Bun
bridge in ``js-bridge/`` instead (see ``test_websocket_bridge.py``).
"""

from __future__ import annotations

import asyncio
import json
from typing import Any, Optional

import websockets

# A turn spec is a list of ops; each op is {"text": str} or
# {"call": {"name": str, "args": dict, "result": Any}}.  A bare string is
# shorthand for a single text op.
TurnSpec = Any


def _frame(name: str, turn_id: str, data: dict[str, Any]) -> str:
    return json.dumps(
        {"v": 1, "type": "frame", "frame": name, "turn_id": turn_id, "data": data}
    )


def _normalise_turn(spec: TurnSpec) -> list[dict[str, Any]]:
    if isinstance(spec, str):
        return [{"text": spec}]
    return [({"text": op} if isinstance(op, str) else op) for op in spec]


class FakeBridgeServer:
    """A scriptable, optionally-misbehaving bridge server for tests.

    Use as an async context manager::

        async with FakeBridgeServer(script=["Hello!"]) as server:
            ... connect WebSocketLLMService to server.uri ...
    """

    def __init__(
        self,
        *,
        script: Optional[list[TurnSpec]] = None,
        respond_to_turns: bool = True,
        turn_error: Optional[str] = None,
        drop_first_connection: bool = False,
        send_malformed_before_response: bool = False,
        chunk_delay: float = 0.0,
        chunks_per_text: int = 2,
    ) -> None:
        self._script = script or []
        self._respond_to_turns = respond_to_turns
        self._turn_error = turn_error
        self._drop_first_connection = drop_first_connection
        self._send_malformed = send_malformed_before_response
        self._chunk_delay = chunk_delay
        self._chunks_per_text = chunks_per_text

        # Observable state for assertions.
        self.received: list[dict[str, Any]] = []
        self.connections = 0
        self.cancelled_turns: list[str] = []
        self.completed_turns: list[dict[str, Any]] = []

        self._server: Optional[websockets.Server] = None
        self._host = "127.0.0.1"
        self._port: Optional[int] = None
        self._turn_index = 0
        self._cancelled: set[str] = set()
        self._turn_tasks: set[asyncio.Task] = set()

    @property
    def uri(self) -> str:
        assert self._port is not None, "server not started"
        return f"ws://{self._host}:{self._port}"

    async def __aenter__(self) -> "FakeBridgeServer":
        self._server = await websockets.serve(self._handle, self._host, 0)
        self._port = self._server.sockets[0].getsockname()[1]
        return self

    async def __aexit__(self, *exc: object) -> None:
        for task in list(self._turn_tasks):
            task.cancel()
        if self._server is not None:
            self._server.close()
            await self._server.wait_closed()

    async def _handle(self, ws: websockets.ServerConnection) -> None:
        self.connections += 1
        this_connection = self.connections
        try:
            async for raw in ws:
                try:
                    envelope = json.loads(raw)
                except json.JSONDecodeError:
                    continue
                self.received.append(envelope)
                await self._dispatch(ws, envelope, this_connection)
        except websockets.ConnectionClosed:
            pass

    async def _dispatch(
        self,
        ws: websockets.ServerConnection,
        envelope: dict[str, Any],
        connection_number: int,
    ) -> None:
        msg_type = envelope.get("type")

        if msg_type == "session.start":
            await ws.send(json.dumps({"v": 1, "type": "session.ready", "info": {}}))
            if self._drop_first_connection and connection_number == 1:
                await ws.close()
            return

        if msg_type == "ping":
            await ws.send(json.dumps({"v": 1, "type": "pong", "ts": envelope.get("ts")}))
            return

        if msg_type == "turn.cancel":
            turn_id = envelope.get("turn_id", "")
            self.cancelled_turns.append(turn_id)
            self._cancelled.add(turn_id)
            return

        if msg_type == "turn.completed":
            self.completed_turns.append(envelope)
            return

        if msg_type == "turn.run":
            task = asyncio.create_task(self._run_turn(ws, envelope))
            self._turn_tasks.add(task)
            task.add_done_callback(self._turn_tasks.discard)
            return

    async def _run_turn(
        self, ws: websockets.ServerConnection, envelope: dict[str, Any]
    ) -> None:
        turn_id = envelope.get("turn_id", "")
        try:
            if not self._respond_to_turns:
                return  # stay silent — exercises the client's turn idle timeout

            if self._turn_error is not None:
                await ws.send(
                    json.dumps(
                        {
                            "v": 1,
                            "type": "error",
                            "turn_id": turn_id,
                            "data": {"message": self._turn_error},
                        }
                    )
                )
                return

            if self._send_malformed:
                await ws.send("this is not json")
                await ws.send(json.dumps({"garbage": True}))

            invocation_id = f"inv-fake-{turn_id}"
            await ws.send(
                _frame("VqlLLMFullResponseStartFrame", turn_id, {"invocation_id": invocation_id})
            )

            spec = self._script[self._turn_index] if self._turn_index < len(self._script) else ""
            self._turn_index += 1
            await self._emit_turn(ws, turn_id, invocation_id, _normalise_turn(spec))

            if turn_id in self._cancelled:
                return
            await ws.send(
                _frame("VqlLLMFullResponseEndFrame", turn_id, {"invocation_id": invocation_id})
            )
        except websockets.ConnectionClosed:
            pass

    async def _emit_turn(
        self,
        ws: websockets.ServerConnection,
        turn_id: str,
        invocation_id: str,
        ops: list[dict[str, Any]],
    ) -> None:
        for op in ops:
            if turn_id in self._cancelled:
                return
            if "text" in op:
                for chunk in self._chunk(op["text"]):
                    if turn_id in self._cancelled:
                        return
                    if self._chunk_delay:
                        await asyncio.sleep(self._chunk_delay)
                    if turn_id in self._cancelled:
                        return
                    await ws.send(
                        _frame(
                            "VqlLLMTextFrame",
                            turn_id,
                            {"invocation_id": invocation_id, "text": chunk},
                        )
                    )
            elif "call" in op:
                call = op["call"]
                fc = {
                    "tool_call_id": f"{invocation_id}-{call['name']}",
                    "function_name": call["name"],
                    "arguments": call.get("args", {}),
                }
                await ws.send(
                    _frame(
                        "VqlFunctionCallsStartedFrame",
                        turn_id,
                        {"invocation_id": invocation_id, "function_calls": [fc]},
                    )
                )
                await ws.send(
                    _frame(
                        "VqlFunctionCallInProgressFrame",
                        turn_id,
                        {"invocation_id": invocation_id, **fc},
                    )
                )
                await ws.send(
                    _frame(
                        "VqlFunctionCallResultFrame",
                        turn_id,
                        {
                            "invocation_id": invocation_id,
                            "tool_call_id": fc["tool_call_id"],
                            "function_name": fc["function_name"],
                            "result": call.get("result", {"status": "ok"}),
                        },
                    )
                )

    def _chunk(self, text: str) -> list[str]:
        n = self._chunks_per_text
        if n <= 1 or len(text) <= n:
            return [text]
        size = -(-len(text) // n)  # ceil division
        return [text[i : i + size] for i in range(0, len(text), size)]
