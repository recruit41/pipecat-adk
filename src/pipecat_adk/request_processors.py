"""Request processors for ADK integration.

This module provides request processors that modify LLM requests before
they are sent to the model, enabling features like interruption handling.
"""

import re
from typing import AsyncGenerator

from google.adk.agents.invocation_context import InvocationContext
from google.adk.events.event import Event
from google.adk.flows.llm_flows._base_llm_processor import BaseLlmRequestProcessor
from google.adk.models.llm_request import LlmRequest
from google.genai.types import Content, Part
from loguru import logger


class InterruptionAwareRequestProcessor(BaseLlmRequestProcessor):
    """Request processor that filters interrupted agent responses.

    Detects synthetic interruption events in conversation history and
    ensures only the portion that was actually heard is included in
    subsequent LLM requests.

    This processor runs BEFORE every LLM request, modifying the
    conversation history to reflect what the user actually heard.

    How it works:
        1. Scans conversation history for interruption markers
        2. Identifies agent responses that were interrupted
        3. Replaces full interrupted responses with spoken portions
        4. Updates session events (in-memory only for this request)
        5. LLM receives filtered conversation reflecting reality

    Example interruption marker:
        <system>Previous agent response was interrupted. Only the following
        portion was heard by the user: 'Hello, how can I...'</system>
    """

    INTERRUPTION_MARKER = "Previous agent response was interrupted"

    async def run_async(
        self, invocation_context: InvocationContext, llm_request: LlmRequest
    ) -> AsyncGenerator[Event, None]:
        """Process LLM request to handle interruptions.

        Filters conversation history to only include text that was
        actually heard by the user.

        Args:
            invocation_context: Context containing session and agent info
            llm_request: The LLM request being prepared

        Yields:
            None - this processor doesn't generate events
        """
        # Get conversation history from session
        session = invocation_context.session
        if not session or not session.events:
            return

        # Scan for interruption markers and filter events
        filtered_events = self._filter_interrupted_events(session.events)

        # Update session with filtered events (in-memory only for this request)
        # This ensures the LLM request will use the filtered conversation
        if len(filtered_events) != len(session.events):
            logger.debug(
                f"Filtered {len(session.events) - len(filtered_events)} events "
                f"due to interruptions"
            )
            session._events = filtered_events

        return
        yield  # Required for AsyncGenerator

    def _filter_interrupted_events(self, events: list[Event]) -> list[Event]:
        """Filter events to handle interruptions.

        Scans events for interruption markers and modifies the conversation
        history to only include text that was actually heard.

        Args:
            events: List of conversation events

        Returns:
            Filtered list of events with interrupted responses replaced
        """
        filtered_events = []
        i = 0

        while i < len(events):
            event = events[i]

            # Check if this is an interruption marker
            if self._is_interruption_event(event):
                # Extract the spoken portion from the marker
                spoken_portion = self._extract_spoken_portion(event)

                # Find and replace the previous agent response
                if filtered_events and filtered_events[-1].author == "agent":
                    # Replace the last agent response with just the spoken portion
                    filtered_events[-1] = self._create_partial_response_event(
                        filtered_events[-1], spoken_portion
                    )
                    logger.debug(
                        f"Replaced interrupted response with spoken portion: "
                        f"'{spoken_portion[:50]}...'"
                    )

                # Skip the interruption marker itself (don't add to filtered list)
                i += 1
                continue

            # Keep the event
            filtered_events.append(event)
            i += 1

        return filtered_events

    def _is_interruption_event(self, event: Event) -> bool:
        """Check if event is an interruption marker.

        Args:
            event: The event to check

        Returns:
            True if this is an interruption marker event
        """
        if not event.content or not event.content.parts:
            return False

        # Check if any part contains the interruption marker
        text = "".join([p.text for p in event.content.parts if p.text])
        return self.INTERRUPTION_MARKER in text

    def _extract_spoken_portion(self, event: Event) -> str:
        """Extract the spoken text from interruption marker.

        Parses the synthetic event text to extract what was actually heard.

        Format:
            <system>Previous agent response was interrupted. Only the following
            portion was heard by the user: 'SPOKEN_TEXT'</system>

        Args:
            event: The interruption marker event

        Returns:
            The spoken portion of text
        """
        text = "".join([p.text for p in event.content.parts if p.text])

        # Extract text between the last pair of single quotes before </system>
        # This handles cases where the spoken text contains single quotes
        # Pattern: "...'SPOKEN_TEXT'</system>" where SPOKEN_TEXT can contain quotes
        # We look for the last occurrence of 'TEXT' before </system>
        match = re.search(r"'(.*)'\s*</system>", text)
        if match:
            return match.group(1)

        # Fallback: try simple pattern (for cases without </system> tag)
        match = re.search(r"'(.*)'", text)
        if match:
            return match.group(1)

        # Fallback: return empty string if parsing fails
        logger.warning(f"Failed to parse spoken portion from: {text[:100]}")
        return ""

    def _create_partial_response_event(
        self, original_event: Event, spoken_portion: str
    ) -> Event:
        """Create a modified event with only the spoken portion.

        Creates a new event that preserves the original metadata but
        replaces the content with just the spoken portion.

        Args:
            original_event: The full agent response event
            spoken_portion: The text that was actually spoken

        Returns:
            New event with modified content
        """
        return Event(
            invocation_id=original_event.invocation_id,
            author=original_event.author,
            content=Content(
                role=original_event.content.role, parts=[Part(text=spoken_portion)]
            ),
            timestamp=original_event.timestamp,
            # Preserve other attributes if they exist
            actions=original_event.actions if hasattr(original_event, "actions") else None,
        )
