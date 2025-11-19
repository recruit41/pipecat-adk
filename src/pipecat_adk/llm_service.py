"""LLM service for ADK integration with Pipecat.

This module provides the main LLM service that uses Google ADK agents
instead of direct LLM calls, enabling agentic workflows in Pipecat pipelines.
"""

from typing import Any, Optional

from google.adk.agents.run_config import RunConfig, StreamingMode
from google.adk.apps.app import App
from google.adk.events.event import Event
from google.adk.runners import Runner
from google.adk.sessions.base_session_service import BaseSessionService
from google.genai.types import Content, FunctionCall, FunctionResponse
from loguru import logger
from pipecat.frames.frames import (
    FunctionCallFromLLM,
    FunctionCallInProgressFrame,
    FunctionCallResultFrame,
    FunctionCallsStartedFrame,
    LLMFullResponseEndFrame,
    LLMFullResponseStartFrame,
    LLMTextFrame,
)
from pipecat.metrics.metrics import LLMTokenUsage
from pipecat.processors.frame_processor import FrameDirection
from pipecat.services.google.llm import (
    GoogleContextAggregatorPair,
    GoogleLLMContext,
    GoogleLLMService,
)
from pipecat.utils.tracing.service_decorators import traced_llm

from .context_aggregators import (
    AdkAssistantContextAggregator,
    AdkUserContextAggregator,
)
from .types import SessionParams


class AdkBasedLLMService(GoogleLLMService):
    def __init__(
        self,
        session_service: BaseSessionService,
        session_params: SessionParams,
        app: App,
        *args: Any,
        **kwargs: Any,
    ) -> None:
        """Initialize the ADK-based LLM service.

        Args:
            session_service: The session service for managing ADK sessions
            session_params: Parameters identifying the session (app_name, user_id, session_id)
            app: The ADK App containing the root agent and plugins.
                 IMPORTANT: The app MUST include InterruptionHandlerPlugin in its plugins
                 list for proper interruption handling.
            *args: Additional arguments passed to GoogleLLMService
            **kwargs: Additional keyword arguments passed to GoogleLLMService
        """
        super().__init__(*args, api_key="does-not-matter", **kwargs)

        self.session_service = session_service
        self.session_params = session_params

        # Create ADK runner with the provided app
        # The app should contain InterruptionHandlerPlugin in its plugins list
        self.runner = Runner(
            app=app,
            session_service=self.session_service,
        )
        logger.debug(f"Created runner with app '{app.name}' and {len(app.plugins)} plugin(s)")

    def create_context_aggregator(
        self, *args: Any, **kwargs: Any
    ) -> GoogleContextAggregatorPair:
        """Create ADK-aware context aggregators.

        Creates user and assistant aggregators that handle the
        bidirectional communication between Pipecat and ADK.

        Returns:
            Pair of user and assistant context aggregators
        """
        user_aggregator = AdkUserContextAggregator()
        assistant_aggregator = AdkAssistantContextAggregator(
            session_service=self.session_service,
            session_params=self.session_params,
            runner=self.runner
        )
        return GoogleContextAggregatorPair(
            _user=user_aggregator, _assistant=assistant_aggregator
        )

    @traced_llm
    async def _process_context(self, context: GoogleLLMContext) -> None:
        messages = context.get_messages()
        if not messages:
            return

        # Get the last message (most recent user input)
        new_message = messages[-1]
        await self._run_adk(new_message)  # type: ignore

    async def _run_adk(
        self, new_message: Content, state_delta: Optional[dict[str, Any]] = None
    ) -> None:
        await self.push_frame(LLMFullResponseStartFrame())

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
                new_message=new_message,
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
            - LLMTextFrame for textual content. Only partial text events. Final text is ignored because it is a repetition.
            - Function call frames to inform Pipecat of function call lifecycle.
        """
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
        """Push function call frames to inform Pipecat the function call has started.
        """
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
        """Push function response frames to inform Pipecat of completion.
        """
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
