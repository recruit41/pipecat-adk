"""Context aggregators for ADK integration with Pipecat.

This module provides context aggregators that bridge Pipecat's pipeline
architecture with Google ADK's session-based conversation management.

Key design decision: ADK owns the conversation history, not Pipecat.
The aggregators save messages directly to ADK sessions and pass only
an invocation_id reference to the LLM service.
"""

import time
from dataclasses import dataclass
from typing import Optional

from google.adk.events.event import Event
from google.adk.runners import Runner
from google.adk.sessions.base_session_service import BaseSessionService
from google.genai.types import Content, Part
from loguru import logger
from pipecat.frames.frames import (
    FunctionCallCancelFrame,
    FunctionCallInProgressFrame,
    FunctionCallResultFrame,
    InterruptionFrame,
)
from pipecat.processors.aggregators.llm_response import (
    LLMUserContextAggregator,
    LLMAssistantContextAggregator,
    LLMUserAggregatorParams,
    LLMAssistantAggregatorParams,
)
from pipecat.processors.aggregators.openai_llm_context import OpenAILLMContext

from .frames import AdkContextFrame
from .types import SessionParams


class AdkUserContextAggregator(LLMUserContextAggregator):
    """User context aggregator that saves messages directly to ADK session.

    Unlike standard Pipecat aggregators that accumulate messages in an
    OpenAILLMContext, this aggregator:
    1. Saves user messages to the ADK session immediately
    2. Generates an invocation_id for the message
    3. Pushes AdkContextFrame(invocation_id) instead of OpenAILLMContextFrame

    This ensures user messages are persisted even if the pipeline is
    interrupted before the LLM service processes them.
    """

    def __init__(
        self,
        session_service: BaseSessionService,
        session_params: SessionParams,
        *,
        params: Optional[LLMUserAggregatorParams] = None,
    ) -> None:
        """Initialize the user context aggregator.

        Args:
            session_service: ADK session service for event persistence
            session_params: Session identification parameters
            params: Optional aggregator parameters for timeout configuration
        """
        # Pass a dummy context - we don't use it for message storage
        # Keep reference to assert it stays empty
        self._dummy_context = OpenAILLMContext()
        super().__init__(context=self._dummy_context, params=params)
        self.session_service = session_service
        self.session_params = session_params
        self._pending_invocation_id: Optional[str] = None

    def _assert_context_empty(self) -> None:
        """Assert that Pipecat's context has no messages (ADK manages them)."""
        messages = self._dummy_context.get_messages()
        assert len(messages) == 0, (
            f"Context should be empty but has {len(messages)} messages. "
            "ADK manages conversation history, not Pipecat."
        )

    async def handle_aggregation(self, aggregation: str) -> None:
        """Save user message to ADK session immediately.

        This is called when user speech has been fully transcribed.
        We save to ADK session right away so the message persists
        even if the pipeline is interrupted.

        Args:
            aggregation: The transcribed user speech
        """
        self._assert_context_empty()  # Defensive check

        # Create user message content
        content = await self._aggregation_to_content(aggregation)

        # Get session
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

        # Create event with new invocation_id
        invocation_id = Event.new_id()  # Generates "e-<uuid>"
        event = Event(
            invocation_id=invocation_id,
            author="user",
            content=content,
            timestamp=time.time(),
        )

        # Save immediately to ADK session
        await self.session_service.append_event(session, event)

        # Store for push_aggregation
        self._pending_invocation_id = invocation_id

        logger.debug(
            f"Saved user message to ADK session with invocation_id={invocation_id}"
        )

    async def _aggregation_to_content(self, aggregation: str) -> Content:
        """Convert aggregated text to Google Content format.

        Override this method to inject additional context (e.g., current time,
        user preferences) into the user message.

        Args:
            aggregation: The transcribed user speech

        Returns:
            Content object with role="user"
        """
        parts = [Part(text=aggregation)]
        return Content(role="user", parts=parts)

    async def push_aggregation(self) -> None:
        """Push AdkContextFrame instead of OpenAILLMContextFrame.

        This is called after handle_aggregation to trigger the LLM.
        We push AdkContextFrame with the invocation_id so the LLM service
        can resume the invocation.
        """
        self._assert_context_empty()  # Defensive check

        if self._pending_invocation_id:
            await self.push_frame(
                AdkContextFrame(invocation_id=self._pending_invocation_id)
            )
            self._pending_invocation_id = None


class AdkAssistantContextAggregator(LLMAssistantContextAggregator):
    """Handles assistant output from ADK agent.

    This aggregator uses Pipecat's built-in text aggregation to track what
    was actually spoken to the user. On interruption, it adds a synthetic
    event to the ADK session indicating what portion was heard.

    The synthetic event approach allows ADK to maintain an accurate
    conversation history that reflects reality (what the user actually heard).

    Key behaviors:
    - handle_aggregation is a no-op (ADK manages assistant messages)
    - Interruption handling creates synthetic events
    - Function call frames are blocked from context (ADK manages them)
    """

    def __init__(
        self,
        session_service: BaseSessionService,
        session_params: SessionParams,
        runner: Runner,
        *,
        params: Optional[LLMAssistantAggregatorParams] = None,
    ) -> None:
        """Initialize the assistant context aggregator.

        Args:
            session_service: ADK session service for event persistence
            session_params: Session identification parameters
            runner: ADK runner instance
            params: Optional aggregator parameters
        """
        # Pass a dummy context - we don't use it for message storage
        # Keep reference to assert it stays empty
        self._dummy_context = OpenAILLMContext()
        super().__init__(context=self._dummy_context, params=params)
        self.session_service = session_service
        self.session_params = session_params
        self.runner = runner

    def _assert_context_empty(self) -> None:
        """Assert that Pipecat's context has no messages (ADK manages them)."""
        messages = self._dummy_context.get_messages()
        assert len(messages) == 0, (
            f"Context should be empty but has {len(messages)} messages. "
            "ADK manages conversation history, not Pipecat."
        )

    async def handle_aggregation(self, _aggregation: str) -> None:
        """No-op: ADK manages assistant messages internally.

        Args:
            _aggregation: The aggregated assistant text (ignored)
        """
        self._assert_context_empty()  # Defensive check
        # ADK already added the LLM response to the session
        pass

    async def _handle_interruptions(self, frame: InterruptionFrame) -> None:
        """Handle user interruption by creating synthetic event.

        Args:
            frame: The interruption frame
        """
        self._assert_context_empty()  # Defensive check

        # Save the spoken text BEFORE calling super(), because super() will clear it
        spoken_text = self._aggregation

        await super()._handle_interruptions(frame)

        if spoken_text:
            await self._add_interruption_event(spoken_text)

    async def _add_interruption_event(self, spoken_text: str) -> None:
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

        if session is None:
            logger.warning(
                f"Cannot add interruption event: session not found "
                f"(session_id={self.session_params.session_id})"
            )
            return

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
    async def handle_function_call_in_progress(
        self, _frame: FunctionCallInProgressFrame
    ) -> None:
        """Block function call from entering context (ADK manages them)."""
        pass  # Intentionally skip super() - ADK manages function calls

    async def handle_function_call_result(
        self, _frame: FunctionCallResultFrame
    ) -> None:
        """Block function call result from entering context (ADK manages them)."""
        pass  # Intentionally skip super() - ADK manages function calls

    async def handle_function_call_cancel(
        self, _frame: FunctionCallCancelFrame
    ) -> None:
        """Block function call cancel from entering context (ADK manages them)."""
        pass  # Intentionally skip super() - ADK manages function calls


@dataclass
class AdkContextAggregatorPair:
    """Pair of user and assistant context aggregators for ADK.

    This replaces GoogleContextAggregatorPair for ADK pipelines.
    The interface is the same (.user(), .assistant()) for compatibility.
    """

    _user: AdkUserContextAggregator
    _assistant: AdkAssistantContextAggregator

    def user(self) -> AdkUserContextAggregator:
        """Get the user context aggregator."""
        return self._user

    def assistant(self) -> AdkAssistantContextAggregator:
        """Get the assistant context aggregator."""
        return self._assistant
