"""LLM service for ADK integration with Pipecat.

This module provides the main LLM service that uses Google ADK agents
instead of direct LLM calls, enabling agentic workflows in Pipecat pipelines.
"""

from typing import Any, Optional

from google.adk.agents import Agent
from google.adk.agents.run_config import RunConfig, StreamingMode
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
from .plugin import InterruptionHandlerPlugin
from .types import SessionParams


class AdkBasedLLMService(GoogleLLMService):
    def __init__(
        self,
        session_service: BaseSessionService,
        session_params: SessionParams,
        agent: Agent,
        *args: Any,
        **kwargs: Any,
    ) -> None:

        super().__init__(*args, api_key="does-not-matter", **kwargs)

        self.session_service = session_service
        self.session_params = session_params

        # Create interruption handler plugin
        # This plugin filters interrupted responses before each LLM call
        interruption_plugin = InterruptionHandlerPlugin()

        # Create ADK runner with the agent and plugins
        # The plugin's before_model_callback will be invoked automatically
        self.runner = Runner(
            app_name=self.session_params.app_name,
            agent=agent,
            session_service=self.session_service,
            plugins=[interruption_plugin],
        )
        logger.debug("Registered InterruptionHandlerPlugin with runner")

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

                # Convert event to frames and push
                await self._push_frames_from_event(event)

        except Exception as e:
            logger.exception(f"{self} exception: {e}")
        finally:
            await self.push_frame(LLMFullResponseEndFrame())


    async def _push_frames_from_event(self, event: Event) -> None:
        """Convert ADK events to Pipecat frames.

        Creates standard LLMTextFrame for text content and pushes
        function call frames upstream/downstream to keep Pipecat
        processors informed of the function call lifecycle.

        This is critical for:
        - STTMuteFilter: Mutes microphone during function execution
        - UserIdleProcessor: Pauses idle detection during function calls
        - Other processors that need function call awareness

        Args:
            event: ADK event to convert
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
        """Push function call frames to inform Pipecat of execution.

        Creates and pushes FunctionCallFromLLM and FunctionCallInProgressFrame
        both UPSTREAM and DOWNSTREAM so all processors in the pipeline are
        aware of the function call lifecycle.

        Upstream processors like STTMuteFilter use this to mute the microphone.
        Downstream processors can use this for tracking or logging.

        Args:
            func_call: The function call from ADK event
        """
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

        Creates and pushes FunctionCallResultFrame both UPSTREAM and
        DOWNSTREAM to notify all processors that the function call
        has completed.

        Args:
            func_response: The function response from ADK event
        """
        result_frame = FunctionCallResultFrame(
            tool_call_id=func_response.id or "",
            function_name=func_response.name or "",
            arguments=None,
            result=func_response.response or {},
        )
        await self.push_frame(result_frame, FrameDirection.UPSTREAM)
        await self.push_frame(result_frame, FrameDirection.DOWNSTREAM)
