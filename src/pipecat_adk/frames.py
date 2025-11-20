"""Frame definitions for ADK state synchronization.

This module provides frames for bidirectional communication between
Pipecat pipelines and ADK sessions.
"""

from dataclasses import dataclass
from typing import Any

from google.adk.events import Event
from google.genai import types as genai_types
from pipecat.frames.frames import SystemFrame


@dataclass
class AdkStateDeltaFrame(SystemFrame):
    """State delta from ADK to be transported to client.

    Emitted when ADK produces state_delta (from tools or events).
    Downstream processors (e.g., transport) can serialize and
    send this to the client.

    Example:
        # In a transport processor
        async def process_frame(self, frame, direction):
            if isinstance(frame, AdkStateDeltaFrame):
                await self.send_to_client({
                    'type': 'state-sync',
                    'delta': frame.state_delta,
                    'source': frame.source
                })
    """
    state_delta: dict[str, Any]
    source: str


@dataclass
class AdkAppendEventFrame(SystemFrame):
    """Request to append an event to ADK session.

    Used when client commands or other processors need to
    persist state to the session without invoking the LLM.

    Example:
        from google.adk.events import Event, EventActions

        event = Event(
            author="user",
            actions=EventActions(state_delta={'quiz_state': new_state})
        )
        await self.push_frame(AdkAppendEventFrame(event=event))
    """
    event: Event


@dataclass
class AdkInvokeAgentFrame(SystemFrame):
    """Request to invoke ADK agent.

    Used when processors need to trigger agent invocation
    outside of normal speech flow (e.g., after processing
    a client command that requires LLM response).

    Example:
        from google.genai import types as genai_types

        content = genai_types.Content(
            role="user",
            parts=[genai_types.Part(text="What's next?")]
        )
        await self.push_frame(AdkInvokeAgentFrame(
            new_content=content,
            state_delta={'last_action': 'submitted_answer'}
        ))
    """
    new_content: genai_types.Content
    state_delta: dict[str, Any] | None = None
