"""
Concise frame observer for pipecat-adk pipeline debugging.

Logs frame activity at INFO level with minimal verbosity.
Format: FrameName field=value field2=value2
"""

from dataclasses import fields, is_dataclass
from typing import Dict, Optional, Set, Type

from google.adk.events import Event
from loguru import logger

from pipecat.frames.frames import (
    BotStartedSpeakingFrame,
    BotStoppedSpeakingFrame,
    Frame,
    InterruptionFrame,
    LLMFullResponseEndFrame,
    LLMFullResponseStartFrame,
    LLMTextFrame,
    StartFrame,
    StartInterruptionFrame,
    TranscriptionFrame,
    TTSTextFrame,
    UserStartedSpeakingFrame,
    UserStoppedSpeakingFrame,
    FunctionCallInProgressFrame,
    FunctionCallResultFrame,
)
from pipecat.observers.base_observer import BaseObserver, FramePushed

# Import ADK frames
from pipecat_adk.frames import AdkContextFrame, AdkInvokeAgentFrame, AdkAppendEventFrame, AdkStateDeltaFrame


class AdkDebugLogObserver(BaseObserver):
    """
    Concise frame observer for pipecat-adk pipeline debugging.

    Logs frame activity at INFO level with minimal verbosity.
    Format: FrameName field=value field2=value2
    """

    # Fields to exclude from logging (binary data, internal fields)
    EXCLUDE_FIELDS = {"audio", "image", "images", "name", "pts", "metadata"}

    # Frame types that should only log specific fields (empty set = no fields)
    FRAME_FIELD_ALLOWLIST: Dict[Type[Frame], Set[str]] = {
        StartFrame: {"id"},
        UserStartedSpeakingFrame: {"id"},
        UserStoppedSpeakingFrame: {"id"},
        BotStartedSpeakingFrame: {"id"},
        BotStoppedSpeakingFrame: {"id"},
        StartInterruptionFrame: {"id"},
        InterruptionFrame: {"id"},
        LLMFullResponseStartFrame: {"id"},
        LLMFullResponseEndFrame: {"id"},
        TranscriptionFrame: {"id", "text"},
        LLMTextFrame: {"id", "text"},
        TTSTextFrame: {"id", "text"},
        FunctionCallInProgressFrame: {"id", "function_name"},
        FunctionCallResultFrame: {"id", "function_name"},
        # ADK frames
        AdkContextFrame: {"id", "invocation_id"},
        AdkInvokeAgentFrame: {"id"},
        AdkAppendEventFrame: {"id"},
        AdkStateDeltaFrame: {"id", "state_delta", "source"},
    }

    # Frame types we care about logging
    DEFAULT_FRAME_TYPES: Set[Type[Frame]] = {
        StartFrame,
        UserStartedSpeakingFrame,
        UserStoppedSpeakingFrame,
        BotStartedSpeakingFrame,
        BotStoppedSpeakingFrame,
        TranscriptionFrame,
        StartInterruptionFrame,
        InterruptionFrame,
        LLMTextFrame,
        LLMFullResponseStartFrame,
        LLMFullResponseEndFrame,
        FunctionCallInProgressFrame,
        FunctionCallResultFrame,
        TTSTextFrame,
        # ADK frames
        AdkContextFrame,
        AdkInvokeAgentFrame,
        AdkAppendEventFrame,
        AdkStateDeltaFrame,
    }

    def __init__(
        self,
        frame_types: Optional[Set[Type[Frame]]] = None,
        **kwargs,
    ):
        """
        Initialize the observer.

        Args:
            frame_types: Set of frame types to log. If None, uses DEFAULT_FRAME_TYPES.
        """
        super().__init__(**kwargs)
        self.frame_types = frame_types if frame_types is not None else self.DEFAULT_FRAME_TYPES
        self._seen_frames: Set[int] = set()  # Track frame IDs to avoid duplicate logging

    def _get_frame_name(self, frame) -> str:
        """Get display name for a frame type."""
        return frame.__class__.__name__

    def _format_value(self, value) -> str:
        """Format a value concisely for logging."""
        if value is None:
            return "None"
        elif isinstance(value, Event):
            return self._format_event(value)
        elif isinstance(value, list):
            if not value:
                return "[]"
            if isinstance(value[0], Event):
                return f"[{', '.join(self._format_event(e) for e in value)}]"
            if len(value) > 3:
                return f"[{len(value)} items]"
            return str(value)
        elif isinstance(value, str):
            # Truncate long strings
            if len(value) > 200:
                return f"'{value[:197]}...'"
            return f"'{value}'"
        elif isinstance(value, (bytes, bytearray)):
            return f"{len(value)}B"
        elif isinstance(value, dict):
            # For dicts, show type key if present, otherwise just count
            if "type" in value:
                return f"{{{value['type']}}}"
            return f"{{{len(value)} keys}}"
        else:
            result = str(value)
            if len(result) > 200:
                return result[:197] + "..."
            return result

    def _format_event(self, event: Event) -> str:
        """Format an ADK Event object compactly."""
        parts = []

        if event.partial:
            parts.append("partial")

        if event.content and event.content.parts:
            text_parts = []
            function_calls = []
            function_responses = []

            for part in event.content.parts:
                if hasattr(part, 'text') and part.text:
                    text_parts.append(part.text)
                if hasattr(part, 'function_call') and part.function_call:
                    function_calls.append(part.function_call.name)
                if hasattr(part, 'function_response') and part.function_response:
                    function_responses.append(part.function_response.name)

            if text_parts:
                combined_text = "".join(text_parts)
                if len(combined_text) > 60:
                    combined_text = combined_text[:57] + "..."
                parts.append(f"'{combined_text}'")

            if function_calls:
                parts.append(f"fn={function_calls}")

            if function_responses:
                parts.append(f"resp={function_responses}")

        return f"Event({', '.join(parts)})" if parts else "Event()"

    def _should_log_frame(self, frame) -> bool:
        """Check if frame type is in our list."""
        for frame_type in self.frame_types:
            if isinstance(frame, frame_type):
                return True
        return False

    def _extract_frame_details(self, frame) -> str:
        """Extract relevant fields from a frame as a concise string."""
        details = []

        # Check if this frame type has an allowlist
        allowlist = None
        for frame_type, allowed_fields in self.FRAME_FIELD_ALLOWLIST.items():
            if isinstance(frame, frame_type):
                allowlist = allowed_fields
                break

        if is_dataclass(frame):
            for field in fields(frame):
                if field.name in self.EXCLUDE_FIELDS:
                    continue
                # If allowlist exists, only include fields in allowlist
                if allowlist is not None and field.name not in allowlist:
                    continue
                value = getattr(frame, field.name)
                if value is None:
                    continue
                formatted = self._format_value(value)
                details.append(f"{field.name}={formatted}")

        return " ".join(details)

    async def on_push_frame(self, data: FramePushed):
        """Log frame push events at INFO level."""
        frame = data.frame
        source = data.source
        destination = data.destination

        if not self._should_log_frame(frame):
            return

        # For StartFrame, log every push to trace propagation
        # For other frames, deduplicate by frame ID
        frame_id = getattr(frame, 'id', None)
        if not isinstance(frame, StartFrame) and frame_id is not None:
            if frame_id in self._seen_frames:
                return
            self._seen_frames.add(frame_id)

        frame_name = self._get_frame_name(frame)
        details = self._extract_frame_details(frame)

        # Include source -> destination for StartFrame to trace propagation
        src_name = source.name if source else "?"
        dst_name = destination.name if destination else "?"

        # Concise format: FrameName details
        if details:
            logger.info(f"{frame_name} {details} ({src_name} -> {dst_name})")
        else:
            logger.info(f"{frame_name} ({src_name} -> {dst_name})")
