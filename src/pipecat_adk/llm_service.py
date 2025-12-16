"""LLM service for ADK integration with Pipecat.

This module provides the main LLM service that uses Google ADK agents
instead of direct LLM calls, enabling agentic workflows in Pipecat pipelines.

Key design: The service receives AdkContextFrame (with invocation_id) from
the user aggregator. The user message has already been saved to the ADK
session, so we just need to invoke run_async with the invocation_id.
"""

import time
from typing import Any, Optional

from google.adk.agents.run_config import RunConfig, StreamingMode
from google.adk.apps.app import App
from google.adk.events.event import Event
from google.adk.events.event_actions import EventActions
from google.adk.runners import Runner
from google.adk.sessions.base_session_service import BaseSessionService
from google.genai.types import FunctionCall, FunctionResponse
from loguru import logger
from pipecat.frames.frames import (
    Frame,
    FunctionCallFromLLM,
    FunctionCallInProgressFrame,
    FunctionCallResultFrame,
    FunctionCallsStartedFrame,
    LLMFullResponseEndFrame,
    LLMFullResponseStartFrame,
    LLMTextFrame,
)
from pipecat.metrics.metrics import LLMTokenUsage
from pipecat.processors.aggregators.llm_context import LLMContext
from pipecat.processors.aggregators.llm_response import (
    LLMUserAggregatorParams,
    LLMAssistantAggregatorParams,
)
from pipecat.processors.aggregators.openai_llm_context import OpenAILLMContext
from pipecat.processors.frame_processor import FrameDirection
from pipecat.services.llm_service import LLMService

from .context_aggregators import (
    AdkAssistantContextAggregator,
    AdkContextAggregatorPair,
    AdkUserContextAggregator,
)
from .frames import AdkAppendEventFrame, AdkContextFrame, AdkInvokeAgentFrame, AdkStateDeltaFrame
from .types import SessionParams


class AdkBasedLLMService(LLMService):
    """LLM service that uses Google ADK agents for conversation.

    This service replaces direct LLM calls with ADK agent invocations,
    enabling sophisticated agent workflows including:
    - Multi-agent orchestration
    - Session-based conversation management
    - Tool execution with state management
    - Interruption handling

    Key differences from standard LLM services:
    - Receives AdkContextFrame (with invocation_id) instead of OpenAILLMContextFrame
    - User messages are already saved to ADK session by the aggregator
    - Uses runner.run_async(invocation_id=...) to resume invocations
    """

    def __init__(
        self,
        session_service: BaseSessionService,
        session_params: SessionParams,
        app: App,
        **kwargs: Any,
    ) -> None:
        """Initialize the ADK-based LLM service.

        Args:
            session_service: The session service for managing ADK sessions
            session_params: Parameters identifying the session (app_name, user_id, session_id)
            app: The ADK App containing the root agent and plugins.
                 IMPORTANT: The app MUST have resumability_config.is_resumable=True
                 and SHOULD include InterruptionHandlerPlugin in its plugins list.
            **kwargs: Additional keyword arguments passed to LLMService

        Raises:
            ValueError: If app doesn't have resumability enabled
        """
        # Validate resumability is enabled
        resumability_config = getattr(app, 'resumability_config', None)
        if not resumability_config or not resumability_config.is_resumable:
            raise ValueError(
                "App must have resumability_config.is_resumable=True. "
                "Example: App(..., resumability_config=ResumabilityConfig(is_resumable=True))"
            )

        super().__init__(**kwargs)

        self.session_service = session_service
        self.session_params = session_params

        # Create ADK runner with the provided app
        # The app should contain InterruptionHandlerPlugin in its plugins list
        self.runner = Runner(
            app=app,
            session_service=self.session_service,
        )
        logger.debug(f"Created runner with app '{app.name}' and {len(app.plugins)} plugin(s)")

    async def run_inference(
        self, _context: LLMContext | OpenAILLMContext
    ) -> Optional[str]:
        """Not supported for ADK-based service.

        ADK manages conversation through sessions, not out-of-band inference.

        Raises:
            NotImplementedError: Always, as ADK doesn't support this pattern
        """
        raise NotImplementedError(
            "ADK-based LLM service does not support out-of-band inference. "
            "Use the pipeline with AdkContextFrame instead."
        )

    def create_context_aggregator(
        self,
        _context: Optional[OpenAILLMContext] = None,  # Ignored - ADK manages context
        *,
        user_params: LLMUserAggregatorParams = LLMUserAggregatorParams(),
        assistant_params: LLMAssistantAggregatorParams = LLMAssistantAggregatorParams(),
    ) -> AdkContextAggregatorPair:
        """Create ADK-aware context aggregators.

        Creates user and assistant aggregators that handle the
        bidirectional communication between Pipecat and ADK.

        Note: The context parameter is ignored because ADK manages
        conversation history in its own session store.

        Args:
            context: Ignored - ADK manages context
            user_params: Parameters for user aggregator
            assistant_params: Parameters for assistant aggregator

        Returns:
            Pair of user and assistant context aggregators
        """
        user_aggregator = AdkUserContextAggregator(
            session_service=self.session_service,
            session_params=self.session_params,
            params=user_params,
        )
        assistant_aggregator = AdkAssistantContextAggregator(
            session_service=self.session_service,
            session_params=self.session_params,
            runner=self.runner,
            params=assistant_params,
        )
        return AdkContextAggregatorPair(
            _user=user_aggregator, _assistant=assistant_aggregator
        )

    async def process_frame(self, frame: Frame, direction: FrameDirection) -> None:
        """Process frames including ADK control frames.

        Handles:
        - AdkContextFrame: Invokes ADK with the invocation_id
        - AdkAppendEventFrame: Appends event to session without LLM invocation
        - AdkInvokeAgentFrame: Programmatic agent invocation

        All other frames are forwarded through the pipeline.
        """
        # Always call super first to handle system frames (StartFrame, etc.)
        await super().process_frame(frame, direction)

        # Handle ADK context frame - this is the main entry point
        if isinstance(frame, AdkContextFrame):
            await self._run_adk(invocation_id=frame.invocation_id)
        # Handle append event frame - persist without LLM call
        elif isinstance(frame, AdkAppendEventFrame):
            await self._handle_append_event(frame)
        # Handle invoke agent frame - programmatic invocation
        elif isinstance(frame, AdkInvokeAgentFrame):
            await self._handle_invoke_agent(frame)
        else:
            # Forward all other frames through the pipeline
            await self.push_frame(frame, direction)

    async def _handle_append_event(self, frame: AdkAppendEventFrame) -> None:
        """Append event to ADK session and emit state delta if present."""
        session = await self.session_service.get_session(
            app_name=self.session_params.app_name,
            user_id=self.session_params.user_id,
            session_id=self.session_params.session_id
        )
        if session is None:
            raise RuntimeError(
                f"ADK session not found: app={self.session_params.app_name}, "
                f"user={self.session_params.user_id}, "
                f"session={self.session_params.session_id}"
            )

        await self.session_service.append_event(session, frame.event)

        # If event has state_delta, emit downstream for transport
        if frame.event.actions and frame.event.actions.state_delta:
            await self.push_frame(AdkStateDeltaFrame(
                state_delta=frame.event.actions.state_delta,
                source=frame.event.author
            ))

    async def _handle_invoke_agent(self, frame: AdkInvokeAgentFrame) -> None:
        """Handle programmatic agent invocation.

        Saves the content to session first, then invokes ADK.
        This ensures the message is persisted even if interrupted.
        """
        session = await self.session_service.get_session(
            app_name=self.session_params.app_name,
            user_id=self.session_params.user_id,
            session_id=self.session_params.session_id,
        )
        if session is None:
            raise RuntimeError(
                f"ADK session not found: app={self.session_params.app_name}, "
                f"user={self.session_params.user_id}, "
                f"session={self.session_params.session_id}"
            )

        # Create event and save to session first
        # Include state_delta in the event's actions to persist it to session state
        invocation_id = Event.new_id()
        event_kwargs = {
            "invocation_id": invocation_id,
            "author": "user",
            "content": frame.new_content,
            "timestamp": time.time(),
        }
        if frame.state_delta:
            event_kwargs["actions"] = EventActions(state_delta=frame.state_delta)
        event = Event(**event_kwargs)
        await self.session_service.append_event(session, event)

        # Now invoke ADK with the invocation_id
        await self._run_adk(
            invocation_id=invocation_id,
            state_delta=frame.state_delta
        )

    async def _run_adk(
        self,
        invocation_id: str,
        state_delta: Optional[dict[str, Any]] = None
    ) -> None:
        """Invoke ADK agent with the given invocation_id.

        The user message has already been saved to the session by the
        aggregator or _handle_invoke_agent. We use run_async with
        invocation_id to resume the invocation.

        Args:
            invocation_id: The invocation ID referencing the saved user message
            state_delta: Optional state changes to apply
        """
        await self.push_frame(LLMFullResponseStartFrame())

        # If state_delta provided, emit it downstream for transport
        if state_delta:
            await self.push_frame(AdkStateDeltaFrame(
                state_delta=state_delta,
                source="user"
            ))

        # Initialize token usage counters
        prompt_tokens = 0
        completion_tokens = 0
        total_tokens = 0
        cache_read_input_tokens = 0
        reasoning_tokens = 0

        try:
            async for event in self.runner.run_async(
                user_id=self.session_params.user_id,
                session_id=self.session_params.session_id,
                invocation_id=invocation_id,
                new_message=None,  # Already saved in session
                state_delta=state_delta,
                run_config=RunConfig(streaming_mode=StreamingMode.SSE),
            ):
                # Stop TTFB metrics after first chunk
                await self.stop_ttfb_metrics()

                # Track token usage from ADK events
                # ADK may send usage_metadata in multiple events with varying behavior:
                # - Sometimes a single event, sometimes multiple events
                # - Token counts may be cumulative (growing) or may change between events
                # We use assignment (not accumulation) because the final event always contains
                # the authoritative, billable token usage for the entire response.
                if event.usage_metadata:
                    prompt_tokens = event.usage_metadata.prompt_token_count or 0
                    completion_tokens = event.usage_metadata.candidates_token_count or 0
                    total_tokens = event.usage_metadata.total_token_count or 0
                    cache_read_input_tokens = event.usage_metadata.cached_content_token_count or 0
                    reasoning_tokens = event.usage_metadata.thoughts_token_count or 0

                # Convert event to frames and push
                await self._push_frames_from_event(event)

        except Exception as e:
            logger.exception(f"{self} exception: {e}")
        finally:
            # Emit token usage metrics to keep Pipecat informed
            await self.start_llm_usage_metrics(
                LLMTokenUsage(
                    prompt_tokens=prompt_tokens,
                    completion_tokens=completion_tokens,
                    total_tokens=total_tokens,
                    cache_read_input_tokens=cache_read_input_tokens,
                    reasoning_tokens=reasoning_tokens,
                )
            )
            await self.push_frame(LLMFullResponseEndFrame())

    async def _push_frames_from_event(self, event: Event) -> None:
        """Convert ADK events to Pipecat frames.

        Specifically:
            - LLMTextFrame for textual content (only partial events)
            - Function call frames to inform Pipecat of function call lifecycle
            - AdkStateDeltaFrame for state deltas from tools
        """
        # Emit state delta frame when present (e.g., from tool results)
        if event.actions and event.actions.state_delta:
            await self.push_frame(AdkStateDeltaFrame(
                state_delta=event.actions.state_delta,
                source=event.author
            ))

        if not event.content or not event.content.parts:
            return

        if event.partial:
            # Since we are in streaming mode, we receive partial text events
            # Only partial events are pushed to Pipecat as LLMTextFrame
            # The final response is discarded because it is a repetition
            text = "".join([part.text for part in event.content.parts if part.text])
            if text:
                await self.push_frame(LLMTextFrame(text=text))

        # Handle function calls - keep Pipecat informed of lifecycle
        for part in event.content.parts:
            if part.function_call:
                await self._handle_function_call(part.function_call)
            elif part.function_response:
                await self._handle_function_response(part.function_response)

    async def _handle_function_call(self, func_call: FunctionCall) -> None:
        """Push function call frames to inform Pipecat the function call has started."""
        # Google ADK FunctionCall must have id and name
        assert func_call.id is not None, "Function call must have an ID"
        assert func_call.name is not None, "Function call must have a name"

        # Create function call frame
        func_call_from_llm = FunctionCallFromLLM(
            tool_call_id=func_call.id,
            function_name=func_call.name,
            arguments=func_call.args or {},
            context=None,
        )

        # Notify pipeline: function calls started
        function_start_frame = FunctionCallsStartedFrame(
            function_calls=[func_call_from_llm]
        )
        await self.push_frame(function_start_frame, FrameDirection.UPSTREAM)
        await self.push_frame(function_start_frame, FrameDirection.DOWNSTREAM)

        # Notify pipeline: function in progress
        function_inprogress_frame = FunctionCallInProgressFrame(
            tool_call_id=func_call.id,
            function_name=func_call.name,
            arguments=func_call.args,
        )
        await self.push_frame(function_inprogress_frame, FrameDirection.UPSTREAM)
        await self.push_frame(function_inprogress_frame, FrameDirection.DOWNSTREAM)

    async def _handle_function_response(self, func_response: FunctionResponse) -> None:
        """Push function response frames to inform Pipecat of completion."""
        assert func_response.id is not None, "Function response must have an ID"
        assert func_response.name is not None, "Function response must have a name"
        result_frame = FunctionCallResultFrame(
            tool_call_id=func_response.id,
            function_name=func_response.name,
            arguments=None,
            result=func_response.response or {},
        )
        await self.push_frame(result_frame, FrameDirection.UPSTREAM)
        await self.push_frame(result_frame, FrameDirection.DOWNSTREAM)
