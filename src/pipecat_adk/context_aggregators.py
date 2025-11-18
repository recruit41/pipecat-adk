"""Context aggregators for ADK integration with Pipecat.

This module provides context aggregators that bridge Pipecat's pipeline
architecture with Google ADK's session-based conversation management.
"""

import time
from typing import List, Optional

from google.adk.events.event import Event
from google.adk.events.event_actions import EventActions
from google.genai.types import Content, Part
from loguru import logger
from pipecat.frames.frames import (
    FunctionCallCancelFrame,
    FunctionCallInProgressFrame,
    FunctionCallResultFrame,
    StartInterruptionFrame,
)
from pipecat.processors.aggregators.llm_response import (
    LLMAssistantContextAggregator,
)
from pipecat.services.google import (
    GoogleLLMContext,
    GoogleUserContextAggregator,
)

from .types import SessionParams


class AdkUserContextAggregator(GoogleUserContextAggregator):
    """Packages user input for ADK agent processing.

    This aggregator clears Pipecat's context (since ADK manages the conversation
    history) and wraps user speech in <candidate> tags for ADK processing.

    Extension Point:
        Override `get_custom_context_parts()` to add custom context like
        warnings, metadata, time tracking, etc.

    Example:
        class MyUserAggregator(AdkUserContextAggregator):
            async def get_custom_context_parts(self):
                parts = []
                if warning := await self.check_device_status():
                    parts.append(Part(text=f"<system>{warning}</system>"))
                return parts
    """

    def __init__(self, context: Optional[GoogleLLMContext] = None):
        """Initialize the user context aggregator.

        Args:
            context: Optional GoogleLLMContext. If not provided, creates a new one.
        """
        super().__init__(context=context or GoogleLLMContext())

    async def handle_aggregation(self, aggregation: str):
        """Package user input for ADK agent.

        Clears Pipecat's context (ADK manages conversation) and wraps user
        speech in <candidate> tags. Calls extension point for custom context.

        Args:
            aggregation: The aggregated user speech from STT
        """
        # Clear Pipecat context - ADK manages conversation history
        self._context.set_messages([])

        # Wrap user input in <candidate> tags
        parts = [Part(text=f"<candidate>{aggregation}</candidate>")]

        # Extension point: add custom context (warnings, metadata, etc.)
        custom_parts = await self.get_custom_context_parts()
        parts.extend(custom_parts)

        # Add to context as single user message
        self._context.add_message(Content(role="user", parts=parts))

    async def get_custom_context_parts(self) -> List[Part]:
        """Extension point for adding custom context.

        Override this method to add custom context parts like warnings,
        metadata, time tracking, device status, etc.

        Returns:
            List of Part objects to add to the user message

        Example:
            async def get_custom_context_parts(self):
                parts = []

                # Add device warning
                if not self.camera_enabled:
                    parts.append(Part(
                        text="<system>Camera is disabled. Ask user to enable it.</system>"
                    ))

                # Add time tracking
                elapsed = time.time() - self.start_time
                parts.append(Part(
                    text=f"<system>Elapsed time: {elapsed:.0f} seconds</system>"
                ))

                return parts
        """
        return []


class AdkAssistantContextAggregator(LLMAssistantContextAggregator):
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
        context: Optional[GoogleLLMContext] = None,
    ):
        """Initialize the assistant context aggregator.

        Args:
            session_service: ADK session service for event persistence
            session_params: Session identification parameters
            context: Optional GoogleLLMContext. If not provided, creates a new one.
        """
        super().__init__(context=context or GoogleLLMContext())
        self.session_service = session_service
        self.session_params = session_params

    async def _handle_interruptions(self, frame: StartInterruptionFrame):
        """Handle user interruption.

        Called by Pipecat when the user interrupts the bot's speech.
        Adds a synthetic event to the ADK session with the portion that
        was actually heard by the user.

        The InterruptionAwareRequestProcessor will filter this during
        the next LLM request to ensure only the heard portion is included
        in the conversation history.

        Args:
            frame: The interruption frame from Pipecat
        """
        # Call parent to aggregate text that was actually spoken
        # This uses Pipecat's built-in tracking of TTSTextFrames
        await super()._handle_interruptions(frame)

        # Get the aggregated text (what user actually heard before interrupting)
        spoken_text = self._aggregation

        if spoken_text:
            logger.debug(
                f"Handling interruption - spoken text: '{spoken_text[:50]}...'"
            )
            await self._add_interruption_event(spoken_text)
        else:
            logger.debug("Handling interruption - no text was spoken yet")

    async def _add_interruption_event(self, spoken_text: str):
        """Add synthetic event to ADK session indicating interruption.

        This event will be processed by InterruptionAwareRequestProcessor
        before the next LLM request to filter the conversation history.

        Args:
            spoken_text: The portion of the response that was actually spoken
        """
        try:
            session = await self.session_service.get_session(
                app_name=self.session_params.app_name,
                user_id=self.session_params.user_id,
                session_id=self.session_params.session_id,
            )

            # Create synthetic event indicating interruption
            interruption_event = Event(
                invocation_id=Event.new_id(),
                author="system",
                content=Content(
                    role="user",
                    parts=[
                        Part(
                            text=f"<system>Previous agent response was interrupted. "
                            f"Only the following portion was heard by the user: "
                            f"'{spoken_text}'</system>"
                        )
                    ],
                ),
                actions=EventActions(skip_summarization=True),
                timestamp=time.time(),
            )

            # Commit to ADK session
            await self.session_service.append_event(session, interruption_event)

            logger.info(
                f"Added interruption event to session {self.session_params.session_id}"
            )

        except Exception as e:
            logger.error(
                f"Failed to add interruption event: {e}",
                exc_info=True,
            )

    # Block function call frames from entering Pipecat context
    # ADK manages function call lifecycle internally, so we don't want
    # these frames to pollute Pipecat's context. However, they are still
    # pushed upstream/downstream by the LLM service to inform other
    # processors (STTMuteFilter, UserIdleProcessor) of the lifecycle.

    async def handle_function_call_in_progress(self, frame: FunctionCallInProgressFrame):
        """Block function call frames from entering context.

        ADK manages function calls internally. We intentionally skip
        the parent implementation to prevent these frames from being
        added to Pipecat's context.

        Args:
            frame: The function call in progress frame
        """
        pass  # Intentionally skip super() - ADK manages function calls

    async def handle_function_call_result(self, frame: FunctionCallResultFrame):
        """Block function response frames from entering context.

        ADK manages function calls internally. We intentionally skip
        the parent implementation to prevent these frames from being
        added to Pipecat's context.

        Args:
            frame: The function call result frame
        """
        pass  # Intentionally skip super() - ADK manages function calls

    async def handle_function_call_cancel(self, frame: FunctionCallCancelFrame):
        """Block function cancel frames from entering context.

        ADK manages function calls internally. We intentionally skip
        the parent implementation to prevent these frames from being
        added to Pipecat's context.

        Args:
            frame: The function call cancel frame
        """
        pass  # Intentionally skip super() - ADK manages function calls
