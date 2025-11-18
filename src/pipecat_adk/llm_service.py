"""LLM service for ADK integration with Pipecat.

This module provides the main LLM service that uses Google ADK agents
instead of direct LLM calls, enabling agentic workflows in Pipecat pipelines.
"""

from typing import Any, Optional

from google.adk.agents import Agent
from google.adk.agents.run_config import RunConfig, StreamingMode
from google.adk.runners import Runner
from google.genai.types import Content
from loguru import logger
from pipecat.frames.frames import (
    EndTaskFrame,
    Frame,
    FunctionCallFromLLM,
    FunctionCallInProgressFrame,
    FunctionCallResultFrame,
    FunctionCallsStartedFrame,
    LLMFullResponseEndFrame,
    LLMFullResponseStartFrame,
    LLMTextFrame,
)
from pipecat.processors.frame_processor import FrameDirection
from pipecat.processors.frameworks.rtvi import RTVIServerMessageFrame
from pipecat.services.ai_service import AIService
from pipecat.services.google import (
    GoogleContextAggregatorPair,
    GoogleLLMContext,
    GoogleLLMService,
)
from pipecat.utils.tracing.service_decorators import traced_llm

from .context_aggregators import (
    AdkAssistantContextAggregator,
    AdkUserContextAggregator,
)
from .request_processors import InterruptionAwareRequestProcessor
from .types import SessionParams


class AdkBasedLLMService(GoogleLLMService):
    """LLM service that uses Google ADK agents instead of direct LLM calls.

    This service integrates ADK agents with Pipecat pipelines, handling:
    - Agent invocation via ADK runner
    - Event to frame conversion
    - Function call lifecycle notification to Pipecat processors
    - Interruption handling with synthetic events
    - Extension points for RTVI and custom frames

    The service uses an "accountant's approach" for interruptions:
    1. Let ADK commit events immediately (natural flow)
    2. On interruption, add synthetic event with spoken portion
    3. Request processor filters conversation before each LLM call
    4. LLM only sees what user actually heard

    Extension Points:
        - Override `process_frame()` to handle custom frames
        - Override `handle_rtvi_frame()` to handle RTVI messages
        - Override `get_custom_context_parts()` in user aggregator

    Example:
        class MyLLMService(AdkBasedLLMService):
            async def process_frame(self, frame, direction):
                if isinstance(frame, MyCustomFrame):
                    await self.handle_custom_frame(frame)
                else:
                    await super().process_frame(frame, direction)

            async def handle_rtvi_frame(self, frame, direction):
                if frame.data.get('type') == 'device-state-changed':
                    await self._handle_device_state(frame.data)
    """

    def __init__(
        self,
        session_service,
        session_params: SessionParams,
        agent: Agent,
        *args,
        **kwargs,
    ):
        """Initialize the ADK-based LLM service.

        Args:
            session_service: ADK session service (any implementation)
            session_params: Session identification parameters
            agent: ADK agent to invoke for conversation
            *args: Additional arguments for GoogleLLMService
            **kwargs: Additional keyword arguments for GoogleLLMService
        """
        super().__init__(*args, **kwargs)

        self.session_service = session_service
        self.session_params = session_params

        # Create ADK runner with the agent
        self.runner = Runner(
            app_name=self.session_params.app_name,
            agent=agent,
            session_service=self.session_service,
        )

        # Register interruption-aware request processor
        # This ensures interruptions are handled transparently
        self._register_interruption_processor()

    def _register_interruption_processor(self):
        """Register InterruptionAwareRequestProcessor with the agent.

        This processor runs before every LLM request, filtering
        interrupted responses from the conversation history.

        The processor is added to the agent's flow so it applies
        automatically to all LLM requests.
        """
        processor = InterruptionAwareRequestProcessor()

        # Add to agent's flow request processors
        # The agent's flow manages the LLM request lifecycle
        if hasattr(self.runner.agent, "_llm_flow"):
            flow = self.runner.agent._llm_flow
            if hasattr(flow, "request_processors"):
                # Ensure request_processors is a list
                if not isinstance(flow.request_processors, list):
                    flow.request_processors = []
                flow.request_processors.append(processor)
                logger.debug("Registered InterruptionAwareRequestProcessor with agent")
            else:
                logger.warning("Agent flow has no request_processors attribute")
        else:
            logger.warning("Agent has no _llm_flow attribute")

    def create_context_aggregator(
        self, *args, **kwargs
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
        )
        return GoogleContextAggregatorPair(
            _user=user_aggregator, _assistant=assistant_aggregator
        )

    async def process_frame(self, frame: Frame, direction: FrameDirection):
        """Process frames - extension point for custom frame handling.

        The base implementation handles RTVI frames via the extension
        point. Override this method to handle custom application frames.

        Example:
            async def process_frame(self, frame, direction):
                if isinstance(frame, MyCustomFrame):
                    await self.handle_custom_frame(frame)
                else:
                    await super().process_frame(frame, direction)

        Args:
            frame: The frame to process
            direction: Direction of frame flow
        """
        # Handle RTVI frames via extension point
        if isinstance(frame, RTVIServerMessageFrame):
            await self.handle_rtvi_frame(frame, direction)
        else:
            await super().process_frame(frame, direction)

    async def handle_rtvi_frame(
        self, frame: RTVIServerMessageFrame, direction: FrameDirection
    ):
        """Extension point for RTVI frame handling.

        Override this method to handle RTVI protocol messages from
        the frontend. Common use cases include device state changes,
        quiz events, coding challenge events, etc.

        Example:
            async def handle_rtvi_frame(self, frame, direction):
                message_type = frame.data.get('type')

                if message_type == 'device-state-changed':
                    # Add event to ADK session
                    session = await self.session_service.get_session(...)
                    event = Event(
                        author="user",
                        content=Content(role="user", parts=[
                            Part(text=f"<system>Device: {frame.data}</system>")
                        ])
                    )
                    await self.session_service.append_event(session, event)

                elif message_type == 'quiz-answer-submitted':
                    await self._handle_quiz_answer(frame.data)

        Args:
            frame: The RTVI server message frame
            direction: Direction of frame flow
        """
        pass  # No-op by default - override to handle RTVI messages

    @traced_llm
    async def _process_context(self, context: GoogleLLMContext):
        """Invoke ADK agent with user message.

        Called by Pipecat when there is a new user message to process.
        Extracts the last message from the context and passes it to ADK.

        Args:
            context: The LLM context containing messages
        """
        messages = context.get_messages()
        if not messages:
            return

        # Get the last message (most recent user input)
        new_message = messages[-1]
        await self._run_adk(new_message)

    async def _run_adk(
        self, new_message: Content, state_delta: Optional[dict[str, Any]] = None
    ):
        """Run ADK agent and stream events as frames.

        Invokes the ADK agent with the user message and streams the
        response events as Pipecat frames.

        Args:
            new_message: The user message to process
            state_delta: Optional state updates to apply
        """
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

    async def push_frame(
        self, frame: Frame, direction: FrameDirection = FrameDirection.DOWNSTREAM
    ):
        """Push frames without LLMService's skip_tts logic.

        The base LLMService has skip_tts logic that we don't want.
        We explicitly control which frames go to TTS.

        Args:
            frame: The frame to push
            direction: Direction to push the frame
        """
        await AIService.push_frame(self, frame, direction)

    async def _push_frames_from_event(self, event) -> None:
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

        # Convert text to standard LLMTextFrame (no custom frames needed)
        text = "".join([part.text for part in event.content.parts if part.text])
        if text:
            await self.push_frame(LLMTextFrame(text=text))

        # Handle function calls - keep Pipecat informed of lifecycle
        for part in event.content.parts:
            if part.function_call:
                await self._handle_function_call(part.function_call)
            elif part.function_response:
                await self._handle_function_response(part.function_response)

    async def _handle_function_call(self, func_call):
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

    async def _handle_function_response(self, func_response):
        """Push function response frames to inform Pipecat of completion.

        Creates and pushes FunctionCallResultFrame both UPSTREAM and
        DOWNSTREAM to notify all processors that the function call
        has completed.

        Args:
            func_response: The function response from ADK event
        """
        result_frame = FunctionCallResultFrame(
            tool_call_id=func_response.id,
            function_name=func_response.name,
            arguments=None,
            result=func_response.response or {},
        )
        await self.push_frame(result_frame, FrameDirection.UPSTREAM)
        await self.push_frame(result_frame, FrameDirection.DOWNSTREAM)
