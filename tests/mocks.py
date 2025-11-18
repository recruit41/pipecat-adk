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
import re
from dataclasses import dataclass
from typing import AsyncGenerator, List, Sequence, Union, Optional
from typing import AsyncGenerator, List
from typing_extensions import override

from google.adk.events import Event
from google.adk.models.base_llm import BaseLlm
from google.adk.models.llm_request import LlmRequest
from google.adk.models.llm_response import LlmResponse
from google.genai.types import Content, Part
from pipecat.observers.loggers.debug_log_observer import DebugLogObserver

from loguru import logger

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
from pipecat.processors.frame_processor import FrameProcessor, FrameDirection
from pipecat.services.stt_service import STTService
from pipecat.services.tts_service import TTSService
from pipecat.utils.time import time_now_iso8601
from pipecat.transports.base_transport import BaseTransport, TransportParams
from pipecat.transports.base_input import BaseInputTransport
from pipecat.transports.base_output import BaseOutputTransport
from pipecat.pipeline.pipeline import Pipeline
from pipecat.pipeline.task import PipelineTask
from pipecat.pipeline.runner import PipelineRunner

from google.adk.sessions import InMemorySessionService
from google.adk.agents import Agent
from pipecat_adk import AdkBasedLLMService, SessionParams

# Constants for mock services
SILENCE = b'\x00'
SAMPLE_RATE = 16000
NUM_CHANNELS = 1

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

# ============================================================================
# UserAction DSL for MockInputTransport
# ============================================================================

@dataclass
class Say:
    """User speaks the given text."""
    text: str

@dataclass
class Join:
    """User joins the session (designed but not implemented yet)."""
    pass


@dataclass
class Leave:
    """User leaves the session (designed but not implemented yet)."""
    pass

@dataclass
class WaitForSomeTime:
    time: float  # seconds to wait

# Type alias for type hints
UserAction = Union[
    Say, Join, Leave, WaitForSomeTime
]


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
    """Test harness for running pipecat pipelines with appropriate services
    """

    def __init__(
        self,
        agent: Agent
    ):
        self.session_params = SessionParams(
            app_name="agents",
            session_id="test_session",
            user_id="test_user",
        )
        self.session_service = InMemorySessionService()

        self.mock_input = MockInputTransport(actions=[])
        self.mock_output = MockOutputTransport()
        self.transport = MockTransport(
            input_transport=self.mock_input,
            output_transport=self.mock_output
        )
        adk_service = AdkBasedLLMService(
            session_service=self.session_service,
            session_params=self.session_params,
            agent=agent
        )
        context_aggregators = adk_service.create_context_aggregator()
        self.pipeline = Pipeline([
            self.mock_input,
            MockSTTService(),
            context_aggregators.user(),
            adk_service,
            context_aggregators.assistant(),
            MockTTSService(),
            self.mock_output,
        ])
        self.task = None
        self.runner = None
        self._pipeline_task = None


    async def session_state(self) -> dict:
        """Get current session state from session service."""
        session = await self.session_service.get_session(**self.session_params.model_dump())
        return session.state if session else {}

    async def events(self) -> List[Event]:
        """Get all events from the current session."""
        session = await self.session_service.get_session(**self.session_params.model_dump())
        return session.events if session else []
    
    
    async def run(self, user_actions: List[UserAction], timeout: float = 0.5) -> "BotResponse":
        session = await self.session_service.create_session(
            **self.session_params.model_dump()
        )
        assert session is not None

        if self.task is None:
            # First run - start pipeline
            self.task = PipelineTask(self.pipeline, observers=[DebugLogObserver()]) 
            self.runner = PipelineRunner()
            self._pipeline_task = asyncio.create_task(
                self.runner.run(self.task)
            )
            # Give pipeline time to start
            await asyncio.sleep(0.1)

        # Clear previous frames for clean assertions
        self.mock_output.clear_frames()

        # Add new actions to transport
        self.mock_input.add_actions(user_actions)

        # Wait for bot to respond (simple timeout-based approach)
        # In production, could use more sophisticated synchronization
        await asyncio.sleep(timeout)

        # Return collected frames wrapped in BotResponse for easier assertions
        return BotResponse(self.mock_output.get_frames())

    async def cleanup(self):
        """Clean up resources."""
        if self._pipeline_task and not self._pipeline_task.done():
            self._pipeline_task.cancel()
            try:
                await self._pipeline_task
            except asyncio.CancelledError:
                pass

    async def __aenter__(self):
        """
        Support async context manager (async with statement).

        Creates all necessary database records before the test starts.
        """
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """Cleanup on context manager exit."""
        await self.cleanup()
        return False


class MockTransport(BaseTransport):
    """Mock transport that supports event handlers for testing.

    This transport connects MockInputTransport and MockOutputTransport,
    and provides event handler registration for testing transport callbacks.
    """

    def __init__(
        self,
        input_transport: Optional['MockInputTransport'] = None,
        output_transport: Optional['MockOutputTransport'] = None,
        *args,
        **kwargs
    ):
        super().__init__(*args, **kwargs)
        self._input_transport = input_transport
        self._output_transport = output_transport

        # Register transport events (similar to DailyTransport)
        self._register_event_handler("on_client_connected")
        self._register_event_handler("on_client_disconnected")
        self._register_event_handler("on_participant_joined")
        self._register_event_handler("on_participant_left")

        # Link this transport to input/output if provided
        if self._input_transport:
            self._input_transport._parent_transport = self
        if self._output_transport:
            self._output_transport._parent_transport = self

    def input(self) -> FrameProcessor:
        if not self._input_transport:
            raise ValueError("MockTransport requires input_transport")
        return self._input_transport

    def output(self) -> FrameProcessor:
        if not self._output_transport:
            raise ValueError("MockTransport requires output_transport")
        return self._output_transport

    async def start_recording(self):
        """No-op for testing - real transports would start recording."""
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
    """Mock input transport that executes a sequence of user actions.

    This transport simulates user behavior by executing a predefined sequence
    of actions like speaking text, waiting for bot responses, and interrupting.
    It's designed for end-to-end testing of pipecat pipelines.

    Example:
        user_actions = [
            Say("Hello"),
            WaitForResponse(),
            Say("My name is John"),
            WaitTillBotSays("position"),
            Say("Software Engineer")
        ]
        transport = MockInputTransport(actions=user_actions)
    """

    def __init__(self, actions: Sequence[UserAction], **kwargs):
        """Initialize the mock input transport with user actions.

        Args:
            actions: Sequence of UserAction instances to execute sequentially.
            **kwargs: Additional arguments passed to BaseInputTransport.
        """
        # Create TransportParams with audio enabled
        params = TransportParams(
            audio_in_enabled=True,
            audio_in_sample_rate=SAMPLE_RATE,
            audio_in_channels=NUM_CHANNELS,
            audio_in_stream_on_start=True,
            audio_in_passthrough=True,  # Pass audio frames downstream after VAD
            vad_analyzer=MockVADAnalyzer()
        )
        super().__init__(params=params, **kwargs)

        # Action queue for dynamic action addition
        self._action_queue: asyncio.Queue = asyncio.Queue()
        self._new_actions_event = asyncio.Event()

        # Populate initial actions
        for action in actions:
            self._action_queue.put_nowait(action)

        self._streaming_task = None
        self._parent_transport: Optional[MockTransport] = None

    def add_actions(self, actions: List[UserAction]):
        """Add new actions to be processed dynamically.

        This allows multi-turn testing by adding actions after pipeline starts.

        Args:
            actions: List of UserAction instances to add to the queue.
        """
        for action in actions:
            self._action_queue.put_nowait(action)
        self._new_actions_event.set()

    async def start(self, frame: StartFrame):
        """Start transport and begin streaming actions."""
        await super().start(frame)
        await self.set_transport_ready(frame)
        if self._params.audio_in_stream_on_start:
            await self.start_audio_in_streaming()

    async def start_audio_in_streaming(self):
        """Start the audio streaming task."""
        if not self._params.audio_in_enabled:
            return

        logger.debug(f"{self}: Starting audio streaming")

        if not self._streaming_task:
            self._streaming_task = self.create_task(
                self._audio_streaming_task_handler()
            )

    async def _audio_streaming_task_handler(self):
        """Main loop processing actions from queue."""
        logger.debug(f"{self}: Starting action handler")

        while True:
            try:
                # Try to get action immediately
                action = self._action_queue.get_nowait()
            except asyncio.QueueEmpty:
                # Wait for new actions with timeout
                try:
                    await asyncio.wait_for(
                        self._new_actions_event.wait(),
                        timeout=30.0
                    )
                    self._new_actions_event.clear()
                    continue
                except asyncio.TimeoutError:
                    logger.debug(f"{self}: Action queue timeout, continuing to wait")
                    continue

            # Process action
            if isinstance(action, Say):
                await self._handle_say(action)
            elif isinstance(action, Join):
                await self._handle_join(action)
            elif isinstance(action, Leave):
                await self._handle_leave(action)
            elif isinstance(action, WaitForSomeTime):
                await asyncio.sleep(action.time)
            else:
                logger.warning(f"{self}: Unknown action type: {action.__class__.__name__}")

    async def stop(self, frame: EndFrame):
        """Stop transport and clean up streaming task."""
        await super().stop(frame)
        await self._cancel_streaming_task()

    async def cancel(self, frame: CancelFrame):
        """Cancel transport and clean up streaming task."""
        await super().cancel(frame)
        await self._cancel_streaming_task()

    async def _cancel_streaming_task(self):
        """Cancel the action streaming task if running."""
        if self._streaming_task:
            await self.cancel_task(self._streaming_task)
            self._streaming_task = None

    async def _handle_say(self, action: Say):
        """Emit user audio frames for the given text.

        Emits: silence (2 bytes) -> speech -> silence (2 bytes)
        The MockVADAnalyzer is configured with minimal thresholds to detect
        transitions with just 2 bytes of silence.
        """
        logger.debug(f"{self}: User says: [{action.text}]")

        # Silence before speech - triggers VAD to start from QUIET state
        # Need 2 frames (2 bytes each = 4 bytes total) to trigger state change
        silence_before = UserAudioRawFrame(
            audio=SILENCE * 4,
            sample_rate=SAMPLE_RATE,
            num_channels=NUM_CHANNELS
        )
        await self.push_audio_frame(silence_before)

        # Speech audio
        speech_bytes = action.text.encode("utf-8")
        speech_frame = UserAudioRawFrame(
            audio=speech_bytes,
            sample_rate=SAMPLE_RATE,
            num_channels=NUM_CHANNELS
        )
        await self.push_audio_frame(speech_frame)

        # Silence after speech - triggers VAD SPEAKING -> QUIET transition
        # Need 2 frames (2 bytes each = 4 bytes total) to trigger state change
        silence_after = UserAudioRawFrame(
            audio=SILENCE * 4,
            sample_rate=SAMPLE_RATE,
            num_channels=NUM_CHANNELS
        )
        await self.push_audio_frame(silence_after)

    async def _emit_silence(self):
        """Emit a single silence frame for liveliness.

        Silence is a single byte b'\x00' emitted every 200ms to prevent
        downstream from thinking the connection has issues.
        """
        frame = UserAudioRawFrame(
            audio=SILENCE,  # Single byte b'\x00'
            sample_rate=SAMPLE_RATE,
            num_channels=NUM_CHANNELS
        )
        await self.push_audio_frame(frame)

    async def _handle_join(self, action: Join):
        """Handle Join action by triggering transport callbacks."""
        logger.debug(f"{self}: User joined")
        if self._parent_transport:
            # Simulate participant joining - pass a mock participant object
            participant = {"id": "test-user", "name": "Test User"}
            await self._parent_transport._call_event_handler("on_participant_joined", participant)
            await self._parent_transport._call_event_handler("on_client_connected", participant)

    async def _handle_leave(self, action: Leave):
        """Handle Leave action by triggering transport callbacks."""
        logger.debug(f"{self}: User left")
        if self._parent_transport:
            participant = {"id": "test-user", "name": "Test User"}
            await self._parent_transport._call_event_handler("on_participant_left", participant, "user_left")


class MockOutputTransport(BaseOutputTransport):
    """Mock output transport that records frames using the default media sender."""

    def __init__(self, **kwargs):
        params = TransportParams(
            audio_out_enabled=True,
            audio_out_sample_rate=SAMPLE_RATE,
        )
        super().__init__(params=params, **kwargs)
        self._parent_transport: Optional[MockTransport] = None
        self._frames: List[Frame] = []

    async def start(self, frame: StartFrame):
        """Initialize base transport and register default media sender."""
        await super().start(frame)
        await self.set_transport_ready(frame)

    async def write_audio_frame(self, frame: Frame) -> bool:
        """Record audio frames and signal success."""
        self._frames.append(frame)
        return True

    async def write_video_frame(self, frame: Frame) -> bool:
        """Record video frames and signal success."""
        self._frames.append(frame)
        return True

    def get_frames(self) -> List[Frame]:
        """Get all collected frames since the last clear."""
        return self._frames.copy()

    def clear_frames(self):
        """Clear recorded frames."""
        self._frames.clear()


class MockTTSService(TTSService):
    """
    A mock implementation of TTSService for unit testing with word-based chunking.

    This class simulates a streaming TTS service by encoding a string to a
    UTF-8 byte array, broken into word-based chunks. It is the symmetric
    counterpart to MockSTTService.
    """

    def can_generate_metrics(self) -> bool:
        """Indicates that this service can generate metrics."""
        return True

    def to_audio(self, text: str) -> AudioRawFrame:
        audio_frame = AudioRawFrame(audio=text.encode("utf-8"), sample_rate=SAMPLE_RATE, num_channels=NUM_CHANNELS)
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

        # Split the input text into word-based chunks.
        text_chunks = _chunk_string(text)

        for i, chunk in enumerate(text_chunks):
            mock_audio_data = chunk.encode("utf-8")
            if mock_audio_data:
                audio_frame = TTSAudioRawFrame(
                    audio=mock_audio_data,
                    sample_rate=self.sample_rate or SAMPLE_RATE,
                    num_channels=NUM_CHANNELS
                )
                audio_frame.id = i
                yield audio_frame
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
        """ For unit tests to easily convert audio bytes to text."""
        return frame.audio.decode("utf-8")

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
            text = audio.decode("utf-8")
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

        yield None # type: ignore


class BotResponse:
    """Wrapper around bot output frames for easier test assertions.
    """

    def __init__(self, frames: List[Frame]):
        self._frames = frames
        self._text_frames: List[TTSTextFrame] = self._extract_text_frames()

    def _extract_text_frames(self) -> List[TTSTextFrame]:
        """Extract all TTS text frames."""
        return [f for f in self._frames if isinstance(f, TTSTextFrame)]

    # === TEXT/SPEECH ===

    @property
    def text(self) -> str:
        """All spoken text concatenated with spaces."""
        return " ".join(f.text for f in self._text_frames)

    @property
    def text_frames(self) -> List[TTSTextFrame]:
        """All TTS text frames in order."""
        return self._text_frames

    def said(self, text: str, case_sensitive: bool = False) -> bool:
        """Check if bot said something containing this text.

        Args:
            text: Text to search for in bot's speech
            case_sensitive: Whether to do case-sensitive matching (default: False)

        Returns:
            True if the text appears in any of the bot's speech
        """
        needle = text if case_sensitive else text.lower()
        haystack = self.text if case_sensitive else self.text.lower()
        return needle in haystack

    # === DEBUGGING ===

    @property
    def all_frames(self) -> List[Frame]:
        """All raw frames for advanced assertions."""
        return self._frames

    def debug_dump(self) -> str:
        """Return formatted string showing all frames (for debugging failing tests)."""
        lines = ["=== BotResponse Debug Dump ==="]
        lines.append(f"Total frames: {len(self._frames)}")
        lines.append(f"\nSpoken text: {self.text!r}")
        lines.append(f"\nAll frame types:")
        for frame in self._frames:
            lines.append(f"  - {type(frame).__name__}")
        return "\n".join(lines)

    def __repr__(self) -> str:
        return f"<BotResponse: {len(self._frames)} frames, text={self.text!r}>"
