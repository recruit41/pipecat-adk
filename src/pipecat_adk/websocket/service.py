"""WebSocket-backed LLM service: delegate LLM processing to an external process.

``WebSocketLLMService`` is a drop-in replacement for ``AdkLLMService`` with
the *exact same black-box behaviour* on the Pipecat side, but instead of
driving a Google ADK runner in-process it forwards each turn to an external
LLM component (a JavaScript / Bun process) over a persistent WebSocket and
streams the response frames back into the pipeline.

Black-box equivalence with AdkLLMService
----------------------------------------
Inbound frames (``process_frame``):
- ``VqlContextFrame(turn_id, text)``  -> run a turn against the remote LLM.
- ``VqlTurnCompletedFrame``  (UPSTREAM) -> consumed here; forwarded to the
  remote process so it can record its own ``[HEARD]`` provenance (the remote
  equivalent of AdkLLMService writing a ``[HEARD]`` event to the ADK session).
- everything else -> forwarded unchanged in the same direction.

Outbound frames (``push_frame``), produced from the remote stream:
- ``VqlLLMFullResponseStartFrame`` / ``VqlLLMTextFrame`` /
  ``VqlLLMFullResponseEndFrame`` -> downstream.
- ``VqlFunctionCallsStartedFrame`` / ``VqlFunctionCallInProgressFrame`` /
  ``VqlFunctionCallResultFrame`` -> both upstream and downstream.
- a matched start/end pair is always emitted, even if the remote process
  fails before producing anything.

Interruption
------------
``VqlContextFrame`` is a plain ``Frame``, so it is processed in Pipecat's
cancellable ``__process_frame_task``.  When ``VqlInterruptionFrame`` arrives,
that task is cancelled, ``CancelledError`` propagates into ``_run_remote``,
and the service tells the remote process to abort the turn (``turn.cancel``).
This mirrors how AdkLLMService's ADK runner is cancelled mid-stream.
"""

from __future__ import annotations

import asyncio
from typing import Any, Optional
from uuid import uuid4

from loguru import logger
from pipecat.frames.frames import (
    CancelFrame,
    EndFrame,
    ErrorFrame,
    Frame,
    StartFrame,
)
from pipecat.metrics.metrics import LLMTokenUsage
from pipecat.processors.aggregators.llm_context import LLMContext
from pipecat.processors.frame_processor import FrameDirection
from pipecat.services.ai_service import AIService
from pipecat.services.llm_service import LLMService

from ..aggregators import (
    VqlAssistantContextAggregator,
    VqlContextAggregatorPair,
    VqlUserContextAggregator,
)
from ..frames import (
    VqlContextFrame,
    VqlLLMFullResponseEndFrame,
    VqlLLMFullResponseStartFrame,
    VqlTurnCompletedFrame,
)
from ..types import SessionParams
from . import protocol
from .client import WebSocketBridgeClient, WebSocketBridgeError


class WebSocketLLMService(LLMService):
    """LLM service that delegates response generation to an external process.

    Example::

        llm = WebSocketLLMService(uri="ws://localhost:8787")
        aggregators = llm.create_context_aggregator()
        pipeline = Pipeline([
            transport.input(),
            stt,
            aggregators.user(),
            llm,
            tts,                # wrapped with VqlTTSMixin
            transport.output(),
            aggregators.assistant(),
        ])

    Extension points for subclasses:
    - ``_build_user_payload(turn_id, text)``: customise the ``turn.run`` payload.
    - ``_on_state_delta(state_delta)``: forward remote state deltas to clients.
    - ``_run_remote(turn_id, text, state_delta)``: inject a programmatic turn.
    """

    def __init__(
        self,
        uri: str,
        *,
        session_params: Optional[SessionParams] = None,
        metadata: Optional[dict[str, Any]] = None,
        connect_timeout: float = 10.0,
        turn_timeout: float = 60.0,
        heartbeat_interval: float = 20.0,
        heartbeat_timeout: float = 10.0,
        reconnect: bool = True,
        additional_headers: Optional[dict[str, str]] = None,
        **kwargs: Any,
    ) -> None:
        """Initialize the WebSocket LLM service.

        Args:
            uri: WebSocket URI of the external LLM component.
            session_params: Optional session identity sent in the handshake.
            metadata: Optional arbitrary metadata sent in the handshake.
            connect_timeout: Seconds to wait for the WebSocket handshake.
            turn_timeout: Idle timeout (seconds between messages of a turn)
                before the turn is abandoned with a timeout error.
            heartbeat_interval: Seconds between application-level heartbeats.
            heartbeat_timeout: Seconds to wait for a heartbeat reply.
            reconnect: Whether to auto-reconnect after connection errors.
            additional_headers: Extra HTTP headers for the WebSocket handshake.
            **kwargs: Additional keyword arguments passed to LLMService.
        """
        super().__init__(**kwargs)
        self.session_params = session_params
        self._uri = uri

        self._client = WebSocketBridgeClient(
            uri,
            task_factory=self.create_task,
            task_canceller=self.cancel_task,
            report_error=self._report_error,
            session=session_params.model_dump() if session_params else None,
            metadata=metadata,
            connect_timeout=connect_timeout,
            turn_timeout=turn_timeout,
            heartbeat_interval=heartbeat_interval,
            heartbeat_timeout=heartbeat_timeout,
            reconnect_on_error=reconnect,
            additional_headers=additional_headers,
        )

    def can_generate_metrics(self) -> bool:
        return True

    def create_context_aggregator(
        self,
        *,
        user_params: Optional[Any] = None,
        assistant_params: Optional[Any] = None,
    ) -> VqlContextAggregatorPair:
        """Create matching user and assistant aggregators for this service.

        The Vql aggregators are LLM-backend agnostic — they are identical to
        the pair AdkLLMService produces.
        """
        user = VqlUserContextAggregator(params=user_params)
        assistant = VqlAssistantContextAggregator(params=assistant_params)
        return VqlContextAggregatorPair(_user=user, _assistant=assistant)

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    async def start(self, frame: StartFrame) -> None:
        await super().start(frame)
        try:
            await self._client.connect()
        except Exception as e:
            logger.error(f"{self} failed to connect to LLM bridge: {e}")
            await self.push_error(
                error_msg=f"Could not connect to LLM bridge at {self._uri}: {e}",
                exception=e,
            )

    async def stop(self, frame: EndFrame) -> None:
        await super().stop(frame)
        await self._client.disconnect()

    async def cancel(self, frame: CancelFrame) -> None:
        await super().cancel(frame)
        await self._client.disconnect()

    async def _process_context(self, context: LLMContext) -> None:
        """No-op: context arrives via VqlContextFrame, not LLMContext."""
        pass

    async def push_frame(
        self, frame: Frame, direction: FrameDirection = FrameDirection.DOWNSTREAM
    ) -> None:
        # Bypass LLMService's skip_tts injection, exactly like AdkLLMService.
        await AIService.push_frame(self, frame, direction)

    async def process_frame(self, frame: Frame, direction: FrameDirection) -> None:
        await super().process_frame(frame, direction)

        if isinstance(frame, VqlTurnCompletedFrame) and direction == FrameDirection.UPSTREAM:
            # Consumed here — forward to the remote process and do not propagate.
            await self._client.send_turn_completed(
                frame.turn_id, frame.text, frame.interrupted
            )

        elif isinstance(frame, VqlContextFrame):
            await self._run_remote(frame.turn_id, frame.text)

        elif not isinstance(frame, VqlTurnCompletedFrame):
            await self.push_frame(frame, direction)

    # ------------------------------------------------------------------
    # Remote turn execution
    # ------------------------------------------------------------------

    async def _run_remote(
        self,
        turn_id: str,
        text: str,
        state_delta: Optional[dict[str, Any]] = None,
    ) -> None:
        """Run one turn against the remote LLM and stream the response.

        Guarantees a matched ``VqlLLMFullResponseStartFrame`` /
        ``VqlLLMFullResponseEndFrame`` pair, just like AdkLLMService.
        """
        await self.start_ttfb_metrics()
        await self.start_processing_metrics()

        usage = LLMTokenUsage(
            prompt_tokens=0,
            completion_tokens=0,
            total_tokens=0,
            cache_read_input_tokens=0,
            reasoning_tokens=0,
        )
        start_frame_pushed = False
        ttfb_stopped = False
        invocation_id = ""

        try:
            payload = await self._build_user_payload(turn_id, text)
            async for envelope in self._client.run_turn(turn_id, payload, state_delta):
                if not ttfb_stopped:
                    await self.stop_ttfb_metrics()
                    ttfb_stopped = True

                msg_type = envelope.get("type")
                data = envelope.get("data") or {}
                if data.get("invocation_id"):
                    invocation_id = data["invocation_id"]

                if msg_type == protocol.MSG_TURN_USAGE:
                    usage = self._build_usage(data)
                    continue

                if msg_type == protocol.MSG_STATE_DELTA:
                    await self._on_state_delta(data)
                    continue

                if msg_type == protocol.MSG_ERROR:
                    raise WebSocketBridgeError(
                        data.get("message") or "remote LLM bridge error"
                    )

                if msg_type != protocol.MSG_FRAME:
                    continue

                pframe, directions = protocol.build_output_frame(envelope)

                if isinstance(pframe, VqlLLMFullResponseEndFrame):
                    # Stream terminator — the matched end frame is emitted in
                    # the finally block, so do not push this one.
                    continue

                if isinstance(pframe, VqlLLMFullResponseStartFrame):
                    start_frame_pushed = True
                elif not start_frame_pushed:
                    # The remote sent content before a start frame — emit a
                    # synthetic one so downstream always sees start-then-content.
                    await self.push_frame(
                        VqlLLMFullResponseStartFrame(
                            turn_id=turn_id, invocation_id=invocation_id
                        )
                    )
                    start_frame_pushed = True

                for d in directions:
                    await self.push_frame(pframe, d)

        except asyncio.CancelledError:
            # Pipeline interruption: tell the remote process to abort the turn.
            self._client.request_cancel(turn_id)
            raise
        except asyncio.TimeoutError as e:
            logger.warning(f"{self} LLM bridge timeout: {e}")
            await self._call_event_handler("on_completion_timeout")
            await self.push_error(error_msg="LLM bridge timed out", exception=e)
        except WebSocketBridgeError as e:
            logger.error(f"{self} LLM bridge error: {e}")
            await self.push_error(error_msg=f"LLM bridge error: {e}", exception=e)
        except protocol.ProtocolError as e:
            logger.error(f"{self} LLM bridge protocol error: {e}")
            await self.push_error(
                error_msg=f"LLM bridge protocol error: {e}", exception=e
            )
        except Exception as e:
            logger.exception(f"{self} unexpected LLM bridge error: {e}")
            await self.push_error(error_msg=f"LLM bridge error: {e}", exception=e)
        finally:
            if not start_frame_pushed:
                await self.push_frame(
                    VqlLLMFullResponseStartFrame(
                        turn_id=turn_id, invocation_id=invocation_id
                    )
                )
            await self.start_llm_usage_metrics(usage)
            await self.stop_processing_metrics()
            await self.push_frame(
                VqlLLMFullResponseEndFrame(turn_id=turn_id, invocation_id=invocation_id)
            )

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _build_usage(data: dict[str, Any]) -> LLMTokenUsage:
        return LLMTokenUsage(
            prompt_tokens=int(data.get("prompt_tokens", 0) or 0),
            completion_tokens=int(data.get("completion_tokens", 0) or 0),
            total_tokens=int(data.get("total_tokens", 0) or 0),
            cache_read_input_tokens=int(data.get("cache_read_input_tokens", 0) or 0),
            reasoning_tokens=int(data.get("reasoning_tokens", 0) or 0),
        )

    async def _report_error(self, error: ErrorFrame) -> None:
        """Bridge connection-error callback — surface the ErrorFrame upstream."""
        await self.push_error_frame(error)

    # ------------------------------------------------------------------
    # Extension points
    # ------------------------------------------------------------------

    async def _build_user_payload(self, turn_id: str, text: str) -> dict[str, Any]:
        """Build the ``turn.run`` payload sent to the remote LLM.

        Override to wrap the text, attach extra context, etc.  The default
        payload is ``{"text": text}``.
        """
        return {"text": text}

    async def _on_state_delta(self, state_delta: dict[str, Any]) -> None:
        """Called for every ``state.delta`` message from the remote process.

        Override to forward state updates to clients (e.g. RTVI state-sync).
        """
        pass

    async def run_injected_turn(
        self,
        text: str,
        state_delta: Optional[dict[str, Any]] = None,
    ) -> None:
        """Inject a programmatic turn (system events, idle prompts, etc.).

        A synthetic turn_id is generated.  Equivalent to AdkLLMService's
        ``_persist_and_run``.
        """
        await self._run_remote(f"injected_{uuid4().hex}", text, state_delta)
