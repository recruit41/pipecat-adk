"""Pipecat-ADK: Integrate Google ADK agents with Pipecat pipelines.

This library provides a seamless integration between Pipecat's real-time
audio/video pipeline framework and Google ADK's agent development kit.

Key Features:
- Use ADK agents instead of standard LLM services
- Automatic interruption handling with synthetic events
- Works with ANY ADK session service
- Keep Pipecat informed of function call lifecycle
- Extension points for custom frames and RTVI integration

Quick Start:
    >>> from pipecat_adk import AdkBasedLLMService, SessionParams
    >>> from google.adk.agents import Agent
    >>> from google.adk.sessions import InMemorySessionService
    >>>
    >>> # Create your ADK agent
    >>> agent = Agent(name="assistant", model="gemini-2.0-flash", ...)
    >>>
    >>> # Create LLM service
    >>> llm = AdkBasedLLMService(
    ...     session_service=InMemorySessionService(),
    ...     session_params=SessionParams(
    ...         app_name="my-app",
    ...         user_id="user",
    ...         session_id="session-123"
    ...     ),
    ...     agent=agent,
    ...     api_key="your-api-key"
    ... )
"""

from .context_aggregators import (
    AdkAssistantContextAggregator,
    AdkUserContextAggregator,
)
from .llm_service import AdkBasedLLMService
from .request_processors import InterruptionAwareRequestProcessor
from .types import SessionParams

__version__ = "0.1.0"

__all__ = [
    "SessionParams",
    "AdkUserContextAggregator",
    "AdkAssistantContextAggregator",
    "InterruptionAwareRequestProcessor",
    "AdkBasedLLMService",
]
