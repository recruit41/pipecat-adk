"""Context aggregators for ADK integration with Pipecat.

This module provides context aggregators that bridge Pipecat's pipeline
architecture with Google ADK's session-based conversation management.
"""

import time
from typing import Optional

from google.adk.events.event import Event
from google.adk.runners import Runner
from google.genai.types import Content, Part
from loguru import logger
from pipecat.frames.frames import (
    FunctionCallCancelFrame,
    FunctionCallInProgressFrame,
    FunctionCallResultFrame,
    StartInterruptionFrame,
)

from pipecat.services.google import (
    GoogleLLMContext,
    GoogleUserContextAggregator,
    GoogleAssistantContextAggregator
)

from .types import SessionParams


class AdkUserContextAggregator(GoogleUserContextAggregator):
    def __init__(self, context: Optional[GoogleLLMContext] = None):
        super().__init__(context=context or GoogleLLMContext())

    async def handle_aggregation(self, aggregation: str):
        content = await self._aggregation_to_content(aggregation)

        # Only keep the current user input in context
        self._context.set_messages([])
        self._context.add_message(content) # type: ignore

    async def _aggregation_to_content(self, aggregation: str) -> Content:
        parts = [Part(text=aggregation)]
        return Content(role="user", parts=parts)

class AdkAssistantContextAggregator(GoogleAssistantContextAggregator):
    """Handles assistant output from ADK agent.

    This aggregator uses Pipecat's built-in text aggregation to track what
    was actually spoken to the user. On interruption, it adds a synthetic
    event to the ADK session indicating what portion was heard.

    The synthetic event approach allows ADK to maintain an accurate
    conversation history that reflects reality (what the user actually heard).
    """

    def __init__(
        self,
        session_service,
        session_params: SessionParams,
        runner: Runner
    ):
        """Initialize the assistant context aggregator.

        Args:
            session_service: ADK session service for event persistence
            session_params: Session identification parameters
            context: Optional GoogleLLMContext. If not provided, creates a new one.
        """
        super().__init__(GoogleLLMContext())
        self.session_service = session_service
        self.session_params = session_params
        self.runner = runner

    async def handle_aggregation(self, aggregation: str):
        # Google ADK already added the LLM response to the session,
        # so we don't need to do anything here.
        pass  # No-op: ADK manages assistant messages internally

    async def _handle_interruptions(self, frame: StartInterruptionFrame):
        await super()._handle_interruptions(frame)
        spoken_text = self._aggregation
        if spoken_text:
            await self._add_interruption_event(spoken_text)

    async def _add_interruption_event(self, spoken_text: str):
        """Add synthetic event to ADK session indicating interruption.

        This event will be processed by InterruptionHandlerPlugin's
        before_model_callback before the next LLM request to filter the
        conversation history.

        Args:
            spoken_text: The portion of the response that was actually spoken
        """
    
        session = await self.session_service.get_session(
            app_name=self.session_params.app_name,
            user_id=self.session_params.user_id,
            session_id=self.session_params.session_id,
        )

        # Determine which agent last responded
        # We need this to set the correct author on the event
        # TODO: Using private API _find_agent_to_run - consider finding public alternative
        last_run_agent = self.runner._find_agent_to_run(session, self.runner.agent)

        # Create synthetic event indicating interruption
        # Wrap in <interruption> tags for reliable marker detection
        interruption_text = f"<interruption>{spoken_text}</interruption>"
        interruption_event = Event(
            invocation_id=Event.new_id(),
            author=last_run_agent.name,
            content=Content(
                role="user",
                parts=[Part(text=interruption_text)],
            ),
            timestamp=time.time(),
        )

        # Commit to ADK session
        await self.session_service.append_event(session, interruption_event)

        logger.info(
            f"Added interruption event to session {self.session_params.session_id}"
        )


    # ADK manages function call lifecycle internally, so we don't want
    # these frames to pollute Pipecat's context. 
    # 
    # BUT we still push them upstream/downstream to inform other
    # processors (STTMuteFilter, UserIdleProcessor).
    async def handle_function_call_in_progress(self, frame: FunctionCallInProgressFrame):
        pass  # Intentionally skip super() - ADK manages function calls

    async def handle_function_call_result(self, frame: FunctionCallResultFrame):
        pass  # Intentionally skip super() - ADK manages function calls

    async def handle_function_call_cancel(self, frame: FunctionCallCancelFrame):
        pass  # Intentionally skip super() - ADK manages function calls
