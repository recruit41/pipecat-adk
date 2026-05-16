"""Hardened WebSocket client for the external LLM bridge.

``WebSocketBridgeClient`` owns the persistent connection to the JavaScript /
Bun LLM component.  It builds on Pipecat's :class:`WebsocketService`, so it
inherits automatic reconnection with exponential backoff and rapid-failure
detection, and adds:

- a connect timeout and per-turn idle timeout,
- protocol-level keepalive (``ping_interval`` / ``ping_timeout``) plus an
  application-level heartbeat that detects a hung JS event loop,
- a turn-scoped receive stream so ``run_turn`` reads as a plain async
  iterator while a single background task drains the socket,
- fail-fast turn termination when the connection drops mid-turn (in-flight
  turns are failed immediately instead of waiting for the idle timeout).

The client speaks the protocol defined in :mod:`.protocol`.
"""

from __future__ import annotations

import asyncio
import time
from collections.abc import AsyncIterator, Awaitable, Callable
from typing import Any, Optional

from loguru import logger
from pipecat.frames.frames import ErrorFrame
from pipecat.services.websocket_service import WebsocketService
from websockets.asyncio.client import connect as ws_connect
from websockets.protocol import State

from . import protocol

# A coroutine factory matching FrameProcessor.create_task.
TaskFactory = Callable[..., asyncio.Task]
TaskCanceller = Callable[[asyncio.Task], Awaitable[None]]
ErrorReporter = Callable[[ErrorFrame], Awaitable[None]]


class WebSocketBridgeError(Exception):
    """Raised when the bridge connection is unusable or a turn fails."""


class _StreamClosed:
    """Sentinel pushed into a turn queue when the connection drops mid-turn."""

    __slots__ = ("message",)

    def __init__(self, message: str) -> None:
        self.message = message


class WebSocketBridgeClient(WebsocketService):
    """Persistent, self-healing WebSocket connection to the external LLM process."""

    def __init__(
        self,
        uri: str,
        *,
        task_factory: TaskFactory,
        task_canceller: TaskCanceller,
        report_error: ErrorReporter,
        session: Optional[dict[str, Any]] = None,
        metadata: Optional[dict[str, Any]] = None,
        connect_timeout: float = 10.0,
        turn_timeout: float = 60.0,
        heartbeat_interval: float = 20.0,
        heartbeat_timeout: float = 10.0,
        reconnect_on_error: bool = True,
        additional_headers: Optional[dict[str, str]] = None,
    ) -> None:
        """Initialize the bridge client.

        Args:
            uri: WebSocket URI of the JS LLM component (e.g. ``ws://localhost:8787``).
            task_factory: Creates background tasks (``FrameProcessor.create_task``).
            task_canceller: Cancels background tasks (``FrameProcessor.cancel_task``).
            report_error: Async callback used to surface connection errors as ErrorFrames.
            session: Optional session identity sent in the handshake.
            metadata: Optional arbitrary metadata sent in the handshake.
            connect_timeout: Seconds to wait for the WebSocket handshake.
            turn_timeout: Idle timeout — max seconds between two messages of a
                turn before it is abandoned. Resets on every received message,
                so long streaming responses are fine.
            heartbeat_interval: Seconds between application-level pings.
            heartbeat_timeout: Seconds to wait for a pong before counting a miss.
            reconnect_on_error: Whether to auto-reconnect after connection errors.
            additional_headers: Extra HTTP headers for the WebSocket handshake.
        """
        super().__init__(reconnect_on_error=reconnect_on_error)
        self._uri = uri
        self._task_factory = task_factory
        self._task_canceller = task_canceller
        self._report_error = report_error
        self._session = session or {}
        self._metadata = metadata or {}
        self._connect_timeout = connect_timeout
        self._turn_timeout = turn_timeout
        self._heartbeat_interval = heartbeat_interval
        self._heartbeat_timeout = heartbeat_timeout
        self._additional_headers = additional_headers

        # One queue per in-flight turn; the receive loop routes messages here.
        self._turn_streams: dict[str, asyncio.Queue] = {}
        self._receive_task: Optional[asyncio.Task] = None
        self._heartbeat_task: Optional[asyncio.Task] = None

        # Heartbeat bookkeeping.
        self._last_ping_ts: float = 0.0
        self._last_pong_ts: float = 0.0
        self._missed_heartbeats: int = 0

    def __str__(self) -> str:
        return f"WebSocketBridgeClient({self._uri})"

    @property
    def is_connected(self) -> bool:
        """True when the underlying socket is open."""
        return self._websocket is not None and self._websocket.state is State.OPEN

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    async def connect(self) -> None:
        """Open the connection and start the receive and heartbeat tasks."""
        await self._connect()  # resets WebsocketService disconnecting flags
        await self._connect_websocket()
        self._receive_task = self._task_factory(
            self._receive_task_handler(self._report_error),
            name="ws-bridge-receive",
        )
        self._heartbeat_task = self._task_factory(
            self._heartbeat_loop(), name="ws-bridge-heartbeat"
        )

    async def disconnect(self) -> None:
        """Gracefully close the connection and stop background tasks."""
        await self._disconnect()  # sets WebsocketService disconnecting flag

        if self.is_connected:
            try:
                await self._send(protocol.encode_session_end())
            except Exception as e:
                logger.debug(f"{self} could not send session.end: {e}")

        for task in (self._heartbeat_task, self._receive_task):
            if task is not None:
                try:
                    await self._task_canceller(task)
                except Exception as e:
                    logger.debug(f"{self} error cancelling task: {e}")
        self._heartbeat_task = None
        self._receive_task = None

        await self._disconnect_websocket()
        self._fail_active_turns("LLM bridge disconnected")

    # WebsocketService abstract methods -------------------------------------

    async def _connect_websocket(self) -> None:
        """Open the socket and send the handshake. Re-invoked on every reconnect."""
        logger.debug(f"{self} connecting")
        self._websocket = await ws_connect(
            self._uri,
            open_timeout=self._connect_timeout,
            # Protocol-level keepalive: detects a dead TCP connection.
            ping_interval=self._heartbeat_interval,
            ping_timeout=self._heartbeat_timeout,
            additional_headers=self._additional_headers,
            max_size=None,
        )
        await self._send(
            protocol.encode_session_start(session=self._session, metadata=self._metadata)
        )
        logger.info(f"{self} connected")

    async def _disconnect_websocket(self) -> None:
        if self._websocket is not None:
            try:
                await self._websocket.close()
            except Exception as e:
                logger.debug(f"{self} error closing socket: {e}")
            self._websocket = None

    async def _receive_messages(self) -> None:
        """Drain the socket; route every message to its turn queue or handler.

        The ``finally`` block fails in-flight turns the moment the socket
        closes, so ``run_turn`` does not block until the idle timeout when the
        bridge goes away.
        """
        ws = self._websocket
        if ws is None:
            return
        try:
            async for raw in ws:
                self._dispatch(raw)
        finally:
            self._fail_active_turns("LLM bridge connection lost")

    # ------------------------------------------------------------------
    # Turn API
    # ------------------------------------------------------------------

    async def run_turn(
        self,
        turn_id: str,
        payload: dict[str, Any],
        state_delta: Optional[dict[str, Any]] = None,
    ) -> AsyncIterator[dict[str, Any]]:
        """Request a response for ``turn_id`` and yield protocol envelopes.

        Yields every ``frame`` / ``turn.usage`` / ``state.delta`` envelope for
        the turn, in order, ending with the ``VqlLLMFullResponseEndFrame``
        envelope (the stream terminator).

        Raises:
            WebSocketBridgeError: if the bridge is not connected or the turn fails.
            asyncio.TimeoutError: if no message arrives within ``turn_timeout``.
        """
        if not self.is_connected:
            raise WebSocketBridgeError(f"{self} is not connected")
        if turn_id in self._turn_streams:
            raise WebSocketBridgeError(f"turn {turn_id!r} is already running")

        queue: asyncio.Queue = asyncio.Queue()
        self._turn_streams[turn_id] = queue
        try:
            await self._send(protocol.encode_turn_run(turn_id, payload, state_delta))
            while True:
                try:
                    item = await asyncio.wait_for(queue.get(), timeout=self._turn_timeout)
                except asyncio.TimeoutError:
                    raise asyncio.TimeoutError(
                        f"no message from LLM bridge for turn {turn_id!r} "
                        f"within {self._turn_timeout}s"
                    )

                if isinstance(item, _StreamClosed):
                    raise WebSocketBridgeError(item.message)

                yield item

                if (
                    item.get("type") == protocol.MSG_FRAME
                    and item.get("frame") == protocol.FRAME_RESPONSE_END
                ):
                    return
        finally:
            self._turn_streams.pop(turn_id, None)

    def request_cancel(self, turn_id: str) -> None:
        """Tell the bridge to abort an in-flight turn.

        Fire-and-forget: this is called from an interruption path where the
        caller's task is itself being cancelled, so the send must run in an
        independent task.
        """
        self._task_factory(
            self._safe_send(protocol.encode_turn_cancel(turn_id)),
            name="ws-bridge-cancel",
        )

    async def send_turn_completed(self, turn_id: str, text: str, interrupted: bool) -> None:
        """Report what the user actually heard for ``turn_id`` to the bridge."""
        await self._safe_send(protocol.encode_turn_completed(turn_id, text, interrupted))

    # ------------------------------------------------------------------
    # Internals
    # ------------------------------------------------------------------

    async def _send(self, message: str) -> None:
        ws = self._websocket
        if ws is None or ws.state is not State.OPEN:
            raise WebSocketBridgeError(f"{self} socket is not open")
        await ws.send(message)

    async def _safe_send(self, message: str) -> None:
        try:
            await self._send(message)
        except Exception as e:
            logger.warning(f"{self} send failed: {e}")

    def _dispatch(self, raw: str | bytes) -> None:
        """Parse one raw message and route it to the right place."""
        try:
            envelope = protocol.decode_message(raw)
        except protocol.ProtocolError as e:
            logger.warning(f"{self} dropping malformed message: {e}")
            return

        msg_type = envelope["type"]

        if msg_type == protocol.MSG_PONG:
            self._last_pong_ts = time.monotonic()
            self._missed_heartbeats = 0
            return

        if msg_type == protocol.MSG_SESSION_READY:
            logger.info(f"{self} bridge reported ready: {envelope.get('info', {})}")
            return

        turn_id = envelope.get("turn_id")
        if turn_id and turn_id in self._turn_streams:
            self._turn_streams[turn_id].put_nowait(envelope)
            return

        if msg_type == protocol.MSG_ERROR:
            # Connection-scoped error (no active turn): surface it directly.
            frame = protocol.build_error_frame(envelope)
            self._task_factory(
                self._report_error(frame), name="ws-bridge-error"
            )
            return

        logger.debug(
            f"{self} message for unknown/inactive turn {turn_id!r} dropped "
            f"(type={msg_type})"
        )

    def _fail_active_turns(self, message: str) -> None:
        for queue in list(self._turn_streams.values()):
            queue.put_nowait(_StreamClosed(message))

    async def _heartbeat_loop(self) -> None:
        """Application-level heartbeat.

        Protocol pings already detect a dead socket; this additionally detects
        a JS process whose event loop is wedged (it still answers protocol
        pings at the library layer but never processes our ``ping`` message).
        After two consecutive misses the socket is closed so the receive task's
        reconnect logic takes over.
        """
        while True:
            await asyncio.sleep(self._heartbeat_interval)
            if not self.is_connected:
                continue

            self._last_ping_ts = time.monotonic()
            await self._safe_send(protocol.encode_ping(self._last_ping_ts))

            await asyncio.sleep(self._heartbeat_timeout)
            if self._last_pong_ts >= self._last_ping_ts:
                continue

            self._missed_heartbeats += 1
            logger.warning(
                f"{self} missed application heartbeat "
                f"({self._missed_heartbeats} consecutive)"
            )
            if self._missed_heartbeats >= 2 and self._websocket is not None:
                logger.error(f"{self} heartbeat lost — forcing reconnect")
                self._missed_heartbeats = 0
                try:
                    await self._websocket.close()
                except Exception as e:
                    logger.debug(f"{self} error closing wedged socket: {e}")
