# mocks.py
#
# Description:
# This file provides mock implementations of TTSService and STTService for
# unit testing pipecat pipelines. These mocks simulate streaming behavior by
# breaking text and audio data into chunks, and they are designed to be
# perfectly symmetric: text "synthesized" by MockTTSService can be perfectly
# "recognized" by MockSTTService.
#
# They avoid any network calls and have no external dependencies beyond the
# core pipecat framework, making them ideal for fast and reliable tests.
#

import asyncio
import copy
import re
from dataclasses import dataclass
from typing import Any, AsyncGenerator, Callable, Dict, List, Literal, Optional, Type, TypeVar, Union
from typing_extensions import override

from google.adk.events import Event
from google.adk.models.base_llm import BaseLlm
from google.adk.models.llm_request import LlmRequest
from google.adk.models.llm_response import LlmResponse
from google.genai.types import Content, Part

from tests.debug_observer import AdkDebugLogObserver

from loguru import logger
from pydantic import BaseModel

from pipecat.frames.frames import (
    Frame,
    InterimTranscriptionFrame,
    TTSAudioRawFrame,
    TTSStartedFrame,
    TTSStoppedFrame,
    TranscriptionFrame,
    AudioRawFrame,
    UserAudioRawFrame,
    BotStartedSpeakingFrame,
    BotStoppedSpeakingFrame,
    TTSTextFrame,
    StartFrame,
    EndFrame,
    CancelFrame,
)
from pipecat.audio.vad.vad_analyzer import VADAnalyzer, VADParams
from pipecat.processors.frame_processor import FrameDirection
from pipecat.services.stt_service import STTService
from pipecat.services.tts_service import TTSService
from pipecat.utils.time import time_now_iso8601
from pipecat.transports.base_transport import BaseTransport, TransportParams
from pipecat.transports.base_input import BaseInputTransport
from pipecat.transports.base_output import BaseOutputTransport
from pipecat.pipeline.pipeline import Pipeline
from pipecat.pipeline.task import PipelineParams, PipelineTask
from pipecat.pipeline.runner import PipelineRunner
from pipecat.processors.frameworks.rtvi import RTVIConfig, RTVIObserver, RTVIProcessor

from google.adk.sessions import InMemorySessionService
from google.adk.apps import App
from pipecat_adk import AdkBasedLLMService, AdkStateDeltaFrame, SessionParams

# Constants for mock services
SILENCE = b'\x00'
INPUT_SAMPLE_RATE = 16000  # Preserve historical behavior for STT/user audio
OUTPUT_SAMPLE_RATE = 48000  # Higher rate keeps mock PCM packets divisible by 6 bytes
NUM_CHANNELS = 1
PCM_BYTES_PER_SAMPLE = 2
FAKE_AUDIO_CHUNK_MS = 20
FAKE_AUDIO_PACKET_BYTES = (
    int(OUTPUT_SAMPLE_RATE * (FAKE_AUDIO_CHUNK_MS / 1000)) * NUM_CHANNELS * PCM_BYTES_PER_SAMPLE
)
FAKE_AUDIO_PADDING_GRANULARITY = 6

if FAKE_AUDIO_PACKET_BYTES % FAKE_AUDIO_PADDING_GRANULARITY != 0:
    raise ValueError(
        f"Fake audio packet size {FAKE_AUDIO_PACKET_BYTES} must be divisible by "
        f"{FAKE_AUDIO_PADDING_GRANULARITY}"
    )

# Backwards compatibility alias
SAMPLE_RATE = OUTPUT_SAMPLE_RATE

# Type variable for generic frame filtering
F = TypeVar('F', bound=Frame)


@dataclass
class Turn:
    """Represents a single conversational turn (user or bot)."""
    speaker: Literal["user", "bot"]
    text: str

def _chunk_string(s: str) -> list[str]:
    """
    Splits a string into word-based chunks.

    Each chunk contains a word along with its trailing whitespace.
    This ensures the invariant: text == "".join(_chunk_string(text))

    This helper function is used by both mock services to simulate streaming.
    """
    if not s:
        return []

    # Use regex to split into words while preserving whitespace
    # Pattern matches: non-whitespace characters followed by optional whitespace
    chunks = re.findall(r'\S+\s*', s)
    return chunks


def _split_text_to_payload_chunks(text: str, *, max_bytes: int) -> list[str]:
    """Split text into chunks that fit into max_bytes when UTF-8 encoded."""
    if not text:
        return []

    chunks: list[str] = []
    current_chars: list[str] = []
    current_bytes = 0

    for char in text:
        encoded = char.encode("utf-8")
        if current_bytes + len(encoded) > max_bytes and current_chars:
            chunks.append("".join(current_chars))
            current_chars = [char]
            current_bytes = len(encoded)
        else:
            current_chars.append(char)
            current_bytes += len(encoded)

    if current_chars:
        chunks.append("".join(current_chars))

    return chunks


def encode_audio_text(text: str, *, padded: bool = False) -> bytes:
    """Encode text into audio bytes, optionally padding for fake PCM packets."""
    payload = text.encode("utf-8")
    if not padded:
        return payload

    if len(payload) > FAKE_AUDIO_PACKET_BYTES:
        raise ValueError(
            f"Payload length {len(payload)} exceeds fake packet size {FAKE_AUDIO_PACKET_BYTES}"
        )

    padding = FAKE_AUDIO_PACKET_BYTES - len(payload)
    if padding:
        payload += b"\x00" * padding
    return payload


def decode_audio_text(packet: bytes, *, padded: bool = False) -> Optional[str]:
    """Decode audio bytes (optionally padded) back into text."""
    if not packet:
        return None

    data = packet.rstrip(b"\x00") if padded else packet
    if padded and not data:
        return None

    try:
        return data.decode("utf-8")
    except UnicodeDecodeError:
        logger.debug("decode_audio_text: skipping non UTF-8 packet")
        return None


# ============================================================================
# BotOutput - Shared state for RTVI messages
# ============================================================================

class BotOutput:
    """Minimal container for RTVI messages sent to the client.

    This class tracks all RTVI messages flowing through the output transport,
    builds the conversation transcript from those messages, and tracks
    state-sync deltas for client state assertions.
    """

    def __init__(self):
        self._messages: List[Dict[str, Any]] = []
        self._client_state: Dict[str, Any] = {}
        self._cond = asyncio.Condition()

    async def append_message(self, payload: Dict[str, Any]):
        """Append an RTVI message payload."""
        async with self._cond:
            self._messages.append(copy.deepcopy(payload))
            self._apply_state_delta(payload)
            self._cond.notify_all()

    @property
    def messages(self) -> List[Dict[str, Any]]:
        """Get all RTVI messages (deep copy)."""
        return copy.deepcopy(self._messages)

    @property
    def client_state(self) -> Dict[str, Any]:
        """Get merged client state from state-sync deltas."""
        return copy.deepcopy(self._client_state)

    def _apply_state_delta(self, message: Dict[str, Any]) -> None:
        """Shallow merge state-sync deltas into client_state."""
        inner = message.get("data") if message.get("type") == "server-message" else message
        if not inner:
            return
        if inner.get("type") != "state-sync":
            return

        delta = inner.get("state_delta") or {}
        for key, value in delta.items():
            self._client_state[key] = value

    @property
    def transcript(self) -> List[Turn]:
        """Chronological utterance list built from RTVI messages."""
        convo: List[Turn] = []
        bot_buffer: List[str] = []
        in_bot_turn = False

        for msg in self._messages:
            msg_type = msg.get("type")
            data = msg.get("data") or {}

            if msg_type == "bot-started-speaking":
                in_bot_turn = True
                bot_buffer = []
            elif msg_type == "bot-tts-text":
                text = data.get("text", "")
                if in_bot_turn:
                    bot_buffer.append(text)
                else:
                    convo.append(Turn(speaker="bot", text=text))
            elif msg_type == "interruption" and in_bot_turn:
                # Flush whatever the bot has said so far to capture partial speech
                convo.append(Turn(speaker="bot", text=" ".join(bot_buffer)))
                bot_buffer = []
                in_bot_turn = False
            elif msg_type == "bot-stopped-speaking" and in_bot_turn:
                convo.append(Turn(speaker="bot", text=" ".join(bot_buffer)))
                bot_buffer = []
                in_bot_turn = False
            elif msg_type == "user-transcription" and data.get("final"):
                convo.append(Turn(speaker="user", text=data.get("text", "")))

        if bot_buffer:
            convo.append(Turn(speaker="bot", text=" ".join(bot_buffer)))

        return convo

    async def wait_for_message_type(
        self,
        message_type: str,
        *,
        start_index: int = 0,
        timeout: float = 1.0,
    ) -> int:
        """Wait for the next message of a given type and return its index."""
        found_idx: Optional[int] = None

        def _has_message() -> bool:
            nonlocal found_idx
            for idx in range(start_index, len(self._messages)):
                if self._messages[idx].get("type") == message_type:
                    found_idx = idx
                    return True
            return False

        async with self._cond:
            await asyncio.wait_for(self._cond.wait_for(_has_message), timeout=timeout)
            assert found_idx is not None
            return found_idx


class MockLLM(BaseLlm):
    """
    Mock LLM model that returns predefined responses with realistic streaming.

    Follows ADK's testing pattern from google.adk.testing_utils.MockModel.
    This allows tests to control what the agent says without calling real LLM APIs.

    Usage:
        # Simple single response
        mock_llm = MockLLM.single("Hello! Welcome to your interview.")

        # Multi-turn text conversation
        mock_llm = MockLLM.conversation([
            "Hello! Welcome.",
            "How are you today?",
            "Great! Let's begin."
        ])

        # Complex: function calls + text
        mock_llm = MockLLM.from_parts([
            [Part.from_function_call(name="change_section", args={"to_section_key": "technical"})],
            "Let's move to some technical topics now. Have you worked with Spring Boot?"
        ])

        # Complex: multiple function calls in one turn
        mock_llm = MockLLM.from_parts([
            [
                Part.from_function_call(name="change_section", args={...}),
                Part.from_function_call(name="update_notes", args={...})
            ],
            "Let's discuss your Java experience."
        ])
    """
    model: str = "mock"
    requests: List[LlmRequest] = []
    responses: List[List[Part]]  # Each turn is a list of Parts
    response_index: int = -1

    @classmethod
    def single(cls, text: str):
        """
        Create a MockLLM that returns a single text response.

        Args:
            text: The text response to return

        Returns:
            MockLLM instance

        Example:
            mock_llm = MockLLM.single("Hello! Welcome to your interview.")
        """
        return cls(responses=[[Part.from_text(text=text)]])

    @classmethod
    def conversation(cls, texts: List[str]):
        """
        Create a MockLLM that returns multiple text responses (multi-turn).

        Args:
            texts: List of text responses, one per turn

        Returns:
            MockLLM instance

        Example:
            mock_llm = MockLLM.conversation([
                "Hello! Welcome.",
                "How are you today?",
                "Great! Let's begin."
            ])
        """
        return cls(responses=[[Part.from_text(text=text)] for text in texts])

    @classmethod
    def from_parts(cls, turns):
        """
        Create a MockLLM with full control over Parts (text, function calls, etc.).

        Args:
            turns: List where each item is either:
                - str: Text response for that turn
                - List[Part]: Multiple Parts for that turn (function calls, text, etc.)

        Returns:
            MockLLM instance

        Examples:
            # Function call followed by text
            mock_llm = MockLLM.from_parts([
                [Part.from_function_call(name="change_section", args={...})],
                "Let's move to technical topics."
            ])

            # Multiple function calls in one turn
            mock_llm = MockLLM.from_parts([
                [
                    Part.from_function_call(name="change_section", args={...}),
                    Part.from_function_call(name="start_quiz", args={...})
                ],
                "Here's a quiz about Java."
            ])
        """
        normalized = cls._normalize_responses(turns)
        return cls(responses=normalized)

    @classmethod
    def _normalize_responses(cls, responses) -> List[List[Part]]:
        """Convert flexible input format to List[List[Part]]."""
        # Single string -> [[Part]]
        if isinstance(responses, str):
            return [[Part.from_text(text=responses)]]

        # List of items
        if isinstance(responses, list):
            normalized = []
            for item in responses:
                if isinstance(item, str):
                    # String -> [Part]
                    normalized.append([Part.from_text(text=item)])
                elif isinstance(item, list):
                    # List of Parts or strings -> [Part, Part, ...]
                    turn_parts = []
                    for part_or_str in item:
                        if isinstance(part_or_str, str):
                            turn_parts.append(Part.from_text(text=part_or_str))
                        else:
                            # Assume it's already a Part
                            turn_parts.append(part_or_str)
                    normalized.append(turn_parts)
                else:
                    # Single Part -> [Part]
                    normalized.append([item])
            return normalized

        # Fallback: treat as single Part
        return [[responses]]

    @classmethod
    def _split_text_for_streaming(cls, text: str, num_chunks: int = 2) -> List[str]:
        """
        Split text into chunks for realistic streaming simulation.

        Splits can occur mid-word to simulate real streaming behavior.
        Short texts (<20 chars) are not split.
        """
        if len(text) < 20:
            return [text]

        # Split into roughly equal chunks
        chunk_size = len(text) // num_chunks
        chunks = []

        for i in range(num_chunks - 1):
            start = i * chunk_size
            end = (i + 1) * chunk_size
            chunks.append(text[start:end])

        # Last chunk gets the remainder
        chunks.append(text[(num_chunks - 1) * chunk_size:])

        return chunks

    @classmethod
    @override
    def supported_models(cls) -> list[str]:
        return ["mock"]

    @override
    async def generate_content_async(
        self, llm_request: LlmRequest, stream: bool = True
    ) -> AsyncGenerator[LlmResponse, None]:
        """
        Generate mock responses asynchronously.

        Streaming behavior (stream=True):
        - For text: Yields 2-3 partial chunks with partial=True, then complete text with partial=None
        - For function calls: Yields once with partial=True, then with partial=None

        Non-streaming (stream=False):
        - Only yields final response with partial=None

        Function Call Handling:
        When ADK receives a function_call from LLM, it executes the function and calls
        generate_content_async again with the function_response in the request.
        MockLLM increments the response index normally - each call gets the next response.
        """
        # Increment index for each LLM call
        self.response_index += 1
        self.requests.append(llm_request)

        # Check if we have a response for this turn
        if self.response_index >= len(self.responses):
            return

        parts = self.responses[self.response_index]

        if stream:
            # Streaming mode: emit realistic chunks
            for part in parts:
                # Check if this is a text part that can be chunked
                if hasattr(part, 'text') and part.text:
                    # Split text into chunks for streaming
                    chunks = self._split_text_for_streaming(part.text)

                    # Yield each chunk as partial
                    for chunk in chunks:
                        chunk_content = Content(role="model", parts=[Part.from_text(text=chunk)])
                        yield LlmResponse(content=chunk_content, partial=True)

                    # Now yield the entire text as final
                    final_content = Content(role="model", parts=[part])
                    yield LlmResponse(content=final_content, partial=None)
                else:
                    # Non-text parts (function calls, etc.) - yield as-is
                    part_content = Content(role="model", parts=[part])
                    yield LlmResponse(content=part_content)

        else:
            # Non-streaming mode: only yield final event
            final_content = Content(role="model", parts=parts)
            yield LlmResponse(content=final_content, partial=None)


class TestRunner:
    """Test harness for running pipecat pipelines with conversational API.

    This class sets up a complete pipeline with mock services and provides
    a conversational API for driving tests:

        async with TestRunner(app=app) as runner:
            await runner.join_and_wait_for_response()
            assert "Hello" in runner.last_bot_message

            await runner.speak_and_wait_for_response("Hi there")
            assert runner.transcript == [
                Turn("bot", "Hello!"),
                Turn("user", "Hi there"),
                Turn("bot", "Nice to meet you!"),
            ]
    """

    def __init__(
        self,
        app: App,
        *,
        tts_delay: float = 0.0,
    ):
        """Initialize the test runner.

        Args:
            app: The ADK App containing the agent and plugins.
            tts_delay: Per-chunk TTS delay in seconds. Use ~0.05 for
                       interruption tests; leave 0 for fastest runs.
        """
        self.session_params = SessionParams(
            app_name="agents",
            session_id="test_session",
            user_id="test_user",
        )
        self.session_service = InMemorySessionService()

        # Create transport with shared BotOutput tracker
        self.transport = MockTransport()
        self.mock_input = self.transport.input()
        self.mock_output = self.transport.output()
        self._bot_output = self.transport.bot_output()

        # Create ADK service
        self.adk_service = AdkBasedLLMService(
            session_service=self.session_service,
            session_params=self.session_params,
            app=app,
        )
        context_aggregators = self.adk_service.create_context_aggregator()

        # Create RTVI processor for message serialization
        self._rtvi = RTVIProcessor(config=RTVIConfig(config=[]))

        # Build pipeline with RTVI
        self.pipeline = Pipeline([
            self.mock_input,
            self._rtvi,  # RTVI observes pipeline events
            MockSTTService(),
            context_aggregators.user(),
            self.adk_service,
            MockTTSService(tts_delay=tts_delay),
            self.mock_output,
            context_aggregators.assistant(),  # MUST be after output transport
        ])

        self.task: Optional[PipelineTask] = None
        self.runner: Optional[PipelineRunner] = None
        self._pipeline_task: Optional[asyncio.Task] = None
        self._joined = False

    # === Black-box assertion properties ===

    @property
    def messages(self) -> List[Dict[str, Any]]:
        """All RTVI messages sent to the client."""
        return self._bot_output.messages

    @property
    def transcript(self) -> List[Turn]:
        """Conversation transcript built from RTVI messages."""
        return self._bot_output.transcript

    @property
    def bot_messages(self) -> List[str]:
        """List of all bot utterances."""
        return [turn.text for turn in self.transcript if turn.speaker == "bot"]

    @property
    def last_bot_message(self) -> str:
        """Most recent bot utterance."""
        if not self.bot_messages:
            raise ValueError("No bot messages yet. Did you forget to call wait_for_response()?")
        return self.bot_messages[-1]

    @property
    def client_state(self) -> Dict[str, Any]:
        """Merged client state from state-sync deltas."""
        return self._bot_output.client_state

    # === Gray-box inspection methods ===

    async def session_state(self) -> dict:
        """Get current ADK session state (gray-box)."""
        session = await self.session_service.get_session(**self.session_params.model_dump())
        return session.state if session else {}

    async def events(self) -> List[Event]:
        """Get all ADK session events (gray-box)."""
        session = await self.session_service.get_session(**self.session_params.model_dump())
        return session.events if session else []

    # === Conversational API ===

    async def _ensure_started(self):
        """Start the pipeline if it hasn't been started yet."""
        if self.task is not None:
            return
        self.task = PipelineTask(
            self.pipeline,
            params=PipelineParams(allow_interruptions=True),
            observers=[RTVIObserver(self._rtvi), AdkDebugLogObserver()],
        )
        self.runner = PipelineRunner()
        self._pipeline_task = asyncio.create_task(self.runner.run(self.task))
        # Wait for StartFrame to fully propagate through all processors.
        # This ensures __process_queue is created on all processors before
        # any other frames are processed.
        await asyncio.sleep(0.2)

    def _ensure_joined(self):
        """Raise if join() hasn't been called."""
        if not self._joined:
            raise RuntimeError("TestRunner.join() must be called before driving the pipeline")

    async def join(self):
        """Simulate the client joining/connecting.

        Note: Pipeline must be started before calling join(). This happens
        automatically when using the context manager (async with TestRunner).
        """
        if self.task is None:
            raise RuntimeError(
                "Pipeline not started. Use 'async with TestRunner(app) as runner:' "
                "to ensure the pipeline is started before calling join()."
            )
        if self._joined:
            return
        participant = {"id": "test-user", "name": "Test User"}
        await self.transport._call_event_handler("on_participant_joined", participant)
        await self.transport._call_event_handler("on_client_connected", participant)
        self._joined = True

    async def join_and_wait_for_response(self, timeout: float = 60.0):
        """Join and wait for bot's first response."""
        await self.join()
        await self.wait_for_response(timeout=timeout)

    async def speak(self, speech: str):
        """Inject user speech into the pipeline."""
        self._ensure_joined()
        await self.mock_input.push_speech(speech)

    async def speak_and_wait_for_response(self, speech: str, timeout: float = 60.0):
        """Speak and wait for bot's response."""
        await self.speak(speech)
        await self.wait_for_response(timeout=timeout)

    async def push_message(self, message_type: str, data: Union[BaseModel, dict, None] = None):
        """Inject a client message into the pipeline."""
        self._ensure_joined()
        await self.mock_input.push_message(message_type, data)

    async def queue_frame(self, frame: Frame):
        """Queue a frame into the pipeline task.

        Use this to inject frames like AdkAppendEventFrame or AdkInvokeAgentFrame
        directly into the pipeline for processing.

        Raises:
            RuntimeError: If join() hasn't been called or task doesn't exist.
        """
        self._ensure_joined()
        if self.task is None:
            raise RuntimeError("Pipeline task not initialized")
        await self.task.queue_frame(frame)

    async def stay_silent(self, iterations: int = 10, delay: float = 0.01):
        """Push silence frames without waiting for response.

        Use after push_message() when the message doesn't trigger an LLM response
        but you need to ensure it's fully processed.
        """
        self._ensure_joined()
        for _ in range(iterations):
            await self.mock_input.push_silence()
            await asyncio.sleep(delay)

    async def wait_for(
        self,
        predicate: Callable[[BotOutput, List[Dict[str, Any]]], bool],
        timeout: float = 60.0,
    ):
        """Wait until predicate(bot_output, delta_messages) returns True."""
        self._ensure_joined()
        start_index = len(self.messages)
        loop = asyncio.get_running_loop()
        deadline = loop.time() + timeout

        while True:
            delta_messages = self._bot_output.messages[start_index:]
            if predicate(self._bot_output, delta_messages):
                return

            if loop.time() >= deadline:
                raise TimeoutError(f"wait_for timed out after {timeout}s")

            await self.mock_input.push_silence()
            await asyncio.sleep(0.01)

    async def wait_for_bot_to_start_speaking(self, timeout: float = 60.0):
        """Wait until bot-started-speaking arrives."""
        def _started(_: BotOutput, delta_messages: List[Dict[str, Any]]) -> bool:
            return any(msg.get("type") == "bot-started-speaking" for msg in delta_messages)
        await self.wait_for(_started, timeout=timeout)

    async def wait_for_response(self, timeout: float = 60.0):
        """Wait for bot-started-speaking followed by bot-stopped-speaking."""
        def _finished(_: BotOutput, delta_messages: List[Dict[str, Any]]) -> bool:
            started = False
            for msg in delta_messages:
                if msg.get("type") == "bot-started-speaking":
                    started = True
                if started and msg.get("type") == "bot-stopped-speaking":
                    return True
            return False
        await self.wait_for(_finished, timeout=timeout)

    async def interrupt_bot(self, message: str, timeout: float = 60.0):
        """Wait for bot to start speaking, then inject interruption."""
        def _started_and_spoke(_: BotOutput, delta_messages: List[Dict[str, Any]]) -> bool:
            started = False
            spoke = False
            for msg in delta_messages:
                if msg.get("type") == "bot-started-speaking":
                    started = True
                if started and msg.get("type") == "bot-tts-text":
                    data = msg.get("data") or {}
                    if data.get("text"):
                        spoke = True
                if started and spoke:
                    return True
            return False
        await self.wait_for(_started_and_spoke, timeout=timeout)
        await self.speak(message)

    # === Lifecycle ===

    async def cleanup(self):
        """Clean up resources."""
        if self._pipeline_task and not self._pipeline_task.done():
            self._pipeline_task.cancel()
            try:
                await self._pipeline_task
            except asyncio.CancelledError:
                pass

    async def __aenter__(self):
        """Support async context manager."""
        # Create session before pipeline starts
        session = await self.session_service.create_session(
            **self.session_params.model_dump()
        )
        assert session is not None

        # Start the pipeline in __aenter__ (not in join)
        # This ensures StartFrame propagates to all processors before
        # any user input can flow through the pipeline
        await self._ensure_started()

        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """Cleanup on context manager exit."""
        await self.cleanup()
        return False


class MockTransport(BaseTransport):
    """Mock transport that supports event handlers for testing.

    This transport connects MockInputTransport and MockOutputTransport,
    and provides event handler registration for testing transport callbacks.

    The transport creates a shared BotOutput that represents the
    communication channel between bot (output) and user (input). Everything
    written by MockOutputTransport is tracked here.
    """

    def __init__(self, *args, **kwargs):
        """Initialize MockTransport with input/output transports."""
        super().__init__(*args, **kwargs)

        # Create shared tracker for bot output (RTVI messages)
        self._bot_output = BotOutput()

        # Create input and output transports with required dependencies
        self._input_transport = MockInputTransport(
            parent_transport=self,
            bot_output=self._bot_output
        )
        self._output_transport = MockOutputTransport(
            parent_transport=self,
            bot_output=self._bot_output
        )

        # Register transport events (similar to DailyTransport)
        self._register_event_handler("on_client_connected")
        self._register_event_handler("on_client_disconnected")
        self._register_event_handler("on_participant_joined")
        self._register_event_handler("on_participant_left")

    def input(self) -> 'MockInputTransport':
        """Return the input transport."""
        return self._input_transport

    def output(self) -> 'MockOutputTransport':
        """Return the output transport."""
        return self._output_transport

    def bot_output(self) -> BotOutput:
        """Return the shared BotOutput tracker."""
        return self._bot_output

    async def start_recording(self, streaming_settings=None, stream_id=None, force_new=None):
        """No-op for testing - real transports would start recording.

        Signature matches DailyTransport.start_recording for compatibility.
        """
        logger.debug(f"{self}: Mock start_recording called (no-op)")

    async def stop_recording(self):
        """No-op for testing - real transports would stop recording."""
        logger.debug(f"{self}: Mock stop_recording called (no-op)")


class MockVADAnalyzer(VADAnalyzer):
    """
    Mock VAD analyzer that detects "speech" in UTF-8 encoded audio.

    Logic:
    - Non-zero bytes = speaking (confidence 1.0)
    - All zero bytes = silence (confidence 0.0)

    This pairs perfectly with MockDailyTransportClient which generates:
    - UTF-8 encoded text during Speak() commands
    - Zero bytes during Pause() commands
    """

    def __init__(self, *, sample_rate: Optional[int] = None, params: Optional[VADParams] = None):
        """Initialize with minimal VAD params optimized for testing.

        Uses very short thresholds so we don't need to pad large silence buffers.
        At 16kHz with num_frames_required=1, each frame is 1/16000 = 0.0000625 seconds.
        - start_secs=0.0001: needs ~2 frames of speech to trigger
        - stop_secs=0.0001: needs ~2 frames of silence to stop
        """
        if params is None:
            params = VADParams(
                confidence=0.5,      # Low threshold for mock
                start_secs=0.0001,   # ~2 frames to start
                stop_secs=0.0001,    # ~2 frames to stop
                min_volume=0.0       # No volume requirement for mock
            )
        super().__init__(sample_rate=sample_rate, params=params)

    def num_frames_required(self) -> int:
        "We can analyze 1 frame itself with absolute confidene"
        return 1

    def voice_confidence(self, buffer: bytes) -> float:
        """
        If the buffer has non-zero bytes, it means it has voice
        """
        # Check if any byte is non-zero
        has_content = any(byte != 0 for byte in buffer)
        return 1.0 if has_content else 0.0


class MockInputTransport(BaseInputTransport):
    """Mock input transport that lets tests inject user speech or client messages.

    This transport provides direct methods for injecting user input:
    - push_speech(text): Inject user speech as UTF-8 audio chunks
    - push_message(type, data): Inject client messages
    - push_silence(): Inject silence frame for liveliness
    """

    def __init__(
        self,
        parent_transport: 'MockTransport',
        bot_output: BotOutput,
        **kwargs
    ):
        """Initialize the mock input transport.

        Args:
            parent_transport: The parent MockTransport that owns this input transport.
            bot_output: Shared BotOutput for tracking messages.
            **kwargs: Additional arguments passed to BaseInputTransport.
        """
        params = TransportParams(
            audio_in_enabled=True,
            audio_in_sample_rate=INPUT_SAMPLE_RATE,
            audio_in_channels=NUM_CHANNELS,
            audio_in_stream_on_start=True,
            audio_in_passthrough=True,
            vad_analyzer=MockVADAnalyzer(),
        )
        super().__init__(params=params, **kwargs)
        self._parent_transport: MockTransport = parent_transport
        self._bot_output: BotOutput = bot_output

    @override
    async def start(self, frame: StartFrame):
        """Start transport."""
        await super().start(frame)
        await self.set_transport_ready(frame)

    @override
    async def stop(self, frame: EndFrame):
        """Stop transport."""
        await super().stop(frame)

    @override
    async def cancel(self, frame: CancelFrame):
        """Cancel transport."""
        await super().cancel(frame)

    async def push_speech(self, speech: str):
        """Inject user speech as UTF-8 audio chunks."""
        for chunk in _chunk_string(speech):
            audio_bytes = encode_audio_text(chunk)
            frame = UserAudioRawFrame(
                audio=audio_bytes,
                sample_rate=INPUT_SAMPLE_RATE,
                num_channels=NUM_CHANNELS,
            )
            await self.push_audio_frame(frame)

    async def push_message(
        self,
        message_type: str,
        data: Union[BaseModel, dict, None] = None,
    ):
        """Inject a client message into the pipeline."""
        # Create message payload for RTVI
        from pipecat.frames.frames import InputTransportMessageFrame
        from pipecat.processors.frameworks.rtvi import RTVI_MESSAGE_LABEL
        import uuid

        msg_id = str(uuid.uuid4())
        message_data = None
        if data is not None:
            if isinstance(data, BaseModel):
                message_data = data.model_dump(exclude_none=True)
            else:
                message_data = data

        transport_message = {
            "label": RTVI_MESSAGE_LABEL,
            "type": "client-message",
            "id": msg_id,
            "data": {
                "t": message_type,
                "d": message_data,
            },
        }
        frame = InputTransportMessageFrame(message=transport_message)
        await self.push_frame(frame, FrameDirection.DOWNSTREAM)

    async def push_silence(self):
        """Inject a single silence frame to keep audio stream alive."""
        frame = UserAudioRawFrame(
            audio=SILENCE,
            sample_rate=INPUT_SAMPLE_RATE,
            num_channels=NUM_CHANNELS,
        )
        await self.push_audio_frame(frame)


class MockOutputTransport(BaseOutputTransport):
    """Mock output transport that tracks RTVI messages.

    This transport uses ONLY the public contract methods (write_audio_frame,
    send_message) to track what's sent to the user.

    RTVI messages are tracked via send_message() and stored in the shared
    BotOutput for test assertions.
    """

    def __init__(
        self,
        parent_transport: 'MockTransport',
        bot_output: BotOutput,
        **kwargs
    ):
        """Initialize the mock output transport.

        Args:
            parent_transport: The parent MockTransport that owns this output transport.
            bot_output: Shared BotOutput for tracking messages.
            **kwargs: Additional arguments passed to BaseOutputTransport.
        """
        params = TransportParams(
            audio_out_enabled=True,
            audio_out_sample_rate=OUTPUT_SAMPLE_RATE,
            audio_out_channels=NUM_CHANNELS,
            audio_out_10ms_chunks=FAKE_AUDIO_CHUNK_MS // 10,
        )
        super().__init__(params=params, **kwargs)
        self._parent_transport: MockTransport = parent_transport
        self._bot_output: BotOutput = bot_output

    async def start(self, frame: StartFrame):
        """Initialize base transport."""
        await super().start(frame)
        # set_transport_ready() automatically creates the MediaSender
        await self.set_transport_ready(frame)

    async def write_audio_frame(self, frame: Frame) -> bool:
        """Write audio frame to output (contract method).

        Audio chunks are ignored for now; playback is validated via RTVI messages.
        """
        return True

    async def write_video_frame(self, frame: Frame) -> bool:
        """Write video frame (no-op for audio-only tests)."""
        return True

    async def send_message(self, frame: Frame):
        """Send transport message to client (contract method).

        RTVIObserver serializes every observable pipeline event (bot speech, TTS chunks,
        transcriptions, etc.) into an RTVI message. RTVIProcessor then forwards that
        payload through send_message, so everything a real client would receive flows
        through here. BotOutput can therefore derive the full conversation state by
        inspecting these serialized messages alone.
        """
        message_payload = getattr(frame, "message", None)
        msg_type = None
        payload_dict = None
        if isinstance(message_payload, BaseModel):
            msg_type = message_payload.__class__.__name__
            payload_dict = message_payload.model_dump(exclude_none=True)
        elif isinstance(message_payload, dict):
            msg_type = message_payload.get("type")
            payload_dict = copy.deepcopy(message_payload)
        logger.debug(f"{self}: send_message called with type={msg_type}")
        if payload_dict is not None:
            await self._bot_output.append_message(payload_dict)


class MockTTSService(TTSService):
    """
    A mock implementation of TTSService for unit testing with word-based chunking.

    This class simulates a streaming TTS service by encoding a string to a
    UTF-8 byte array, broken into word-based chunks. It is the symmetric
    counterpart to MockSTTService.

    Args:
        tts_delay: Per-chunk delay in seconds. Use ~0.05 for interruption tests
                   to give bot time to produce partial speech before being cut off.
    """

    def __init__(self, *, tts_delay: float = 0.0, **kwargs):
        super().__init__(sample_rate=OUTPUT_SAMPLE_RATE, **kwargs)
        self._tts_delay = tts_delay

    def can_generate_metrics(self) -> bool:
        """Indicates that this service can generate metrics."""
        return True

    def to_audio(self, text: str) -> AudioRawFrame:
        audio_frame = AudioRawFrame(
            audio=encode_audio_text(text, padded=True),
            sample_rate=OUTPUT_SAMPLE_RATE,
            num_channels=NUM_CHANNELS,
        )
        return audio_frame

    async def run_tts(self, text: str) -> AsyncGenerator[Frame, None]:
        """
        Mocks the TTS process by encoding text to bytes in word-based chunks.
        """
        logger.debug(f"{self}: MOCK generating TTS for text: [{text}]")

        await self.start_ttfb_metrics()
        # Start usage metrics with the full, original text.
        await self.start_tts_usage_metrics(text)

        yield TTSStartedFrame()

        # Split the input text into payload-sized chunks.
        text_chunks = _split_text_to_payload_chunks(text, max_bytes=FAKE_AUDIO_PACKET_BYTES)

        for i, chunk in enumerate(text_chunks):
            mock_audio_data = encode_audio_text(chunk, padded=True)
            if mock_audio_data:
                audio_frame = TTSAudioRawFrame(
                    audio=mock_audio_data,
                    sample_rate=OUTPUT_SAMPLE_RATE,
                    num_channels=NUM_CHANNELS,
                )
                audio_frame.id = i
                yield audio_frame
                if self._tts_delay > 0:
                    await asyncio.sleep(self._tts_delay)
            # Stop TTFB metrics after yielding the very first chunk of audio.
            if i == 0:
                await self.stop_ttfb_metrics()

        yield TTSStoppedFrame()


class MockSTTService(STTService):
    """
    A mock implementation of STTService for unit testing with streaming behavior.

    This service processes audio chunks and uses silence frames (b'\x00') as
    transcription boundaries. Audio between silence frames generates interim
    transcriptions that accumulate until a silence frame triggers the final
    transcription.

    The service maintains state across multiple calls to run_stt:
    - _buffer: Accumulates text from non-silence audio chunks
    - _is_transcribing: Tracks whether we're in an active transcription
    """

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self._buffer = ""
        self._is_transcribing = False

    def can_generate_metrics(self) -> bool:
        """Indicates that this service can generate metrics."""
        return True

    def to_text(self, frame: AudioRawFrame) -> str:
        """For unit tests to easily convert audio bytes to text."""
        return decode_audio_text(frame.audio, padded=False) or ""

    async def run_stt(self, audio: bytes) -> AsyncGenerator[Frame, None]:
        """
        Mocks the STT process with silence-based transcription boundaries.

        - If audio is silence (SILENCE constant), finalizes current buffer as TranscriptionFrame
        - Otherwise, decodes audio, adds to buffer, and yields InterimTranscriptionFrame
        """

        # Check if this is a silence frame (boundary marker)
        is_silence = audio == SILENCE or all(b == 0 for b in audio)

        if is_silence:
            # Silence marks the end of a transcription
            if self._buffer:
                # We have accumulated text, so emit final transcription
                await self.start_ttfb_metrics()
                await self.start_processing_metrics()

                await self.push_frame(
                    TranscriptionFrame(
                        text=self._buffer,
                        user_id=self._user_id,
                        timestamp=time_now_iso8601(),
                    )
                )

                await self.stop_ttfb_metrics()
                await self.stop_processing_metrics()

                # Clear buffer for next transcription
                self._buffer = ""

            self._is_transcribing = False
        else:
            # Non-silence audio: decode and accumulate
            text = decode_audio_text(audio, padded=False) or ""
            self._buffer += text

            if not self._is_transcribing:
                await self.start_ttfb_metrics()
                await self.start_processing_metrics()
                self._is_transcribing = True

            # Emit interim transcription with accumulated buffer
            await self.push_frame(
                InterimTranscriptionFrame(
                    text=self._buffer,
                    user_id=self._user_id,
                    timestamp=time_now_iso8601(),
                )
            )

            # Stop TTFB after first frame
            if len(self._buffer) == len(text):  # This is the first chunk
                await self.stop_ttfb_metrics()

        yield None  # type: ignore
