"""Type definitions for pipecat-adk."""

from dataclasses import dataclass
from typing import Optional


@dataclass
class SessionParams:
    """Parameters for identifying an ADK session.

    Attributes:
        app_name: Application name for the ADK session
        user_id: User identifier (ADK requires "user" for ADK web compatibility)
        session_id: Unique session identifier
    """
    app_name: str
    user_id: str
    session_id: str

    def model_dump(self) -> dict:
        """Convert to dictionary for compatibility with ADK."""
        return {
            "app_name": self.app_name,
            "user_id": self.user_id,
            "session_id": self.session_id
        }
