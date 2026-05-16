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
import uuid
from dataclasses import dataclass
from typing import Any, AsyncGenerator, Callable, Dict, List, Literal, Optional, Union
from typing_extensions import override

from google.adk.models.base_llm import BaseLlm
from google.adk.models.llm_request import LlmRequest
from google.adk.models.llm_response import LlmResponse
from google.genai.types import Content, Part

from tests.debug_observer import AdkDebugLogObserver

from loguru import logger
from pydantic import BaseModel

from pipecat.frames.frames import (
    AudioRawFrame,
    CancelFrame,
    EndFrame,
    Frame,
    InputTransportMessageFrame,
    InterimTranscriptionFrame,
    StartFrame,
    TranscriptionFrame,
    TTSAudioRawFrame,
    UserAudioRawFrame,
)
from pipecat.audio.vad.vad_analyzer import VADAnalyzer, VADParams
from pipecat.pipeline.pipeline import Pipeline
from pipecat.pipeline.runner import PipelineRunner
from pipecat.pipeline.task import PipelineParams, PipelineTask
from pipecat.processors.aggregators.llm_response_universal import (
    LLMUserAggregatorParams,
    UserTurnStrategies,
)
from pipecat.processors.frame_processor import FrameDirection
from pipecat.processors.frameworks.rtvi import (
    RTVIObserver,
    RTVIProcessor,
)
from pipecat.processors.frameworks.rtvi.models import MESSAGE_LABEL as RTVI_MESSAGE_LABEL
from pipecat.services.stt_service import STTService
from pipecat.services.tts_service import TTSService
from pipecat.transports.base_input import BaseInputTransport
from pipecat.transports.base_output import BaseOutputTransport
from pipecat.transports.base_transport import BaseTransport, TransportParams
from pipecat.turns.user_start import TranscriptionUserTurnStartStrategy, VADUserTurnStartStrategy
from pipecat.turns.user_stop import SpeechTimeoutUserTurnStopStrategy
from pipecat.utils.time import time_now_iso8601

from google.adk.apps import App
from google.adk.sessions import InMemorySessionService
from pipecat_adk import AdkLLMService, VqlTTSMixin, SessionParams


# ============================================================================
# Constants
# ============================================================================

# Two bytes = one complete 16-bit PCM sample; avoid odd-length buffers that
# break analyzers treating audio as 16-bit PCM.
SILENCE = b"\x00\x00"

INPUT_SAMPLE_RATE = 16000   # STT / user audio
OUTPUT_SAMPLE_RATE = 48000  # TTS / bot audio — keeps fake PCM packets divisible by 6 bytes
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


# ============================================================================
# Audio encoding helpers
# ============================================================================

def _chunk_string(s: str) -> list[str]:
    """Split a string into word-based chunks, preserving trailing whitespace.

    Invariant: "".join(_chunk_string(text)) == text
    """
    if not s:
        return []
    return re.findall(r"\S+\s*", s)


def _split_text_to_payload_chunks(text: str, *, max_bytes: int) -> list[str]:
    """Split text into chunks whose UTF-8 encoding fits within max_bytes."""
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
    """Encode text into audio bytes.

    padded=False: PCM-aligned (even length) — suitable for STT input frames.
    padded=True:  Fixed FAKE_AUDIO_PACKET_BYTES length — suitable for TTS output frames.
    """
    payload = text.encode("utf-8")

    if not padded:
        # Align to 16-bit PCM boundary so analyzers treating audio as 16-bit samples
        # don't receive odd-length buffers.
        if len(payload) % PCM_BYTES_PER_SAMPLE != 0:
            payload += b"\x00"
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
    """Decode audio bytes back into text.

    padded=True strips trailing null bytes before decoding (symmetric with
    encode_audio_text(padded=True)).
    """
    if not packet:
        return None

    data = packet.rstrip(b"\x00") if padded else packet
    if padded and not data:
        return None

    try:
        return data.decode("utf-8")
    except UnicodeDecodeError:
        logger.debug("decode_audio_text: skipping non-UTF-8 packet")
        return None


def create_client_message_frame(
    message_type: str,
    data: Union[BaseModel, dict, None] = None,
    msg_id: Optional[str] = None,
) -> InputTransportMessageFrame:
    """Create an InputTransportMessageFrame for RTVI client messages."""
    if msg_id is None:
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
    return InputTransportMessageFrame(message=transport_message)


# ============================================================================
# Conversation types
# ============================================================================

@dataclass
class Turn:
    speaker: Literal["user", "bot"]
    text: str


# ============================================================================
# BotOutput — tracks RTVI messages from the output transport
# ============================================================================

class BotOutput:
    """Tracks all RTVI messages sent to the client and derives conversation state."""

    def __init__(self):
        self._messages: List[Dict[str, Any]] = []
        self._client_state: Dict[str, Any] = {}
        self._errors: List[str] = []
        self._cond = asyncio.Condition()

    async def append_message(self, payload: Dict[str, Any]):
        async with self._cond:
            self._messages.append(copy.deepcopy(payload))
            self._apply_state_delta(payload)
            self._extract_errors(payload)
            self._cond.notify_all()

    @property
    def messages(self) -> List[Dict[str, Any]]:
        return copy.deepcopy(self._messages)

    @property
    def client_state(self) -> Dict[str, Any]:
        return copy.deepcopy(self._client_state)

    @property
    def errors(self) -> List[str]:
        """All RTVI error messages received from the pipeline."""
        return list(self._errors)

    def _apply_state_delta(self, message: Dict[str, Any]) -> None:
        """Shallow-merge state-sync deltas into client_state."""
        inner = message.get("data") if message.get("type") == "server-message" else message
        if not inner:
            return
        if inner.get("type") != "state-sync":
            return
        for key, value in (inner.get("state_delta") or {}).items():
            self._client_state[key] = value

    def _extract_errors(self, message: Dict[str, Any]) -> None:
        if message.get("type") == "error":
            error_text = message.get("data", {}).get("error")
            if error_text:
                self._errors.append(error_text)

    @property
    def transcript(self) -> List[Turn]:
        """Chronological utterance list built from RTVI speaking events.

        Uses bot-started/stopped-speaking as turn boundaries and accumulates
        bot-tts-text within each speaking window. This is tied to actual audio
        playback timing through the output transport, making it more reliable
        than LLM-event-based tracking where LLMFullResponseEndFrame can race
        ahead of audio frames in the serialization queue.

        bot-tts-text arriving outside a speaking window (e.g. before
        bot-started-speaking when using push_start_frame=True) is captured
        as a standalone Turn so no text is ever lost.
        """
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
                    # Text arrived outside a speaking window — emit immediately.
                    if text.strip():
                        convo.append(Turn(speaker="bot", text=text))
            elif msg_type == "interruption" and in_bot_turn:
                # Flush partial speech captured so far.
                joined = " ".join(t.strip() for t in bot_buffer if t.strip())
                if joined:
                    convo.append(Turn(speaker="bot", text=joined))
                bot_buffer = []
                in_bot_turn = False
            elif msg_type == "bot-stopped-speaking" and in_bot_turn:
                joined = " ".join(t.strip() for t in bot_buffer if t.strip())
                if joined:
                    convo.append(Turn(speaker="bot", text=joined))
                bot_buffer = []
                in_bot_turn = False
            elif msg_type == "user-transcription" and data.get("final"):
                convo.append(Turn(speaker="user", text=data.get("text", "")))

        # Flush any open bot turn (e.g. pipeline ended mid-speech).
        if bot_buffer:
            joined = " ".join(t.strip() for t in bot_buffer if t.strip())
            if joined:
                convo.append(Turn(speaker="bot", text=joined))

        return convo

    async def wait_for_message_type(
        self,
        message_type: str,
        *,
        start_index: int = 0,
        timeout: float = 1.0,
    ) -> int:
        """Wait for the next message of the given type; return its index."""
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


# ============================================================================
# MockLLM — ADK model that returns predefined responses
# ============================================================================

class MockLLM(BaseLlm):
    """Mock ADK LLM that returns predefined responses with realistic streaming.

    Usage:
        MockLLM.single("Hello!")
        MockLLM.conversation(["Hello!", "How are you?"])
        MockLLM.from_parts([
            [Part.from_function_call(name="fn", args={})],
            "Result text",
        ])
    """

    model: str = "mock"
    requests: List[LlmRequest] = []
    responses: List[List[Part]]
    response_index: int = -1

    @classmethod
    def single(cls, text: str) -> "MockLLM":
        return cls(responses=[[Part.from_text(text=text)]])

    @classmethod
    def conversation(cls, texts: List[str]) -> "MockLLM":
        return cls(responses=[[Part.from_text(text=t)] for t in texts])

    @classmethod
    def from_parts(cls, turns) -> "MockLLM":
        return cls(responses=cls._normalize_responses(turns))

    @classmethod
    def _normalize_responses(cls, responses) -> List[List[Part]]:
        if isinstance(responses, str):
            return [[Part.from_text(text=responses)]]
        if isinstance(responses, list):
            normalized = []
            for item in responses:
                if isinstance(item, str):
                    normalized.append([Part.from_text(text=item)])
                elif isinstance(item, list):
                    turn_parts = []
                    for p in item:
                        if isinstance(p, str):
                            turn_parts.append(Part.from_text(text=p))
                        else:
                            turn_parts.append(p)
                    normalized.append(turn_parts)
                else:
                    normalized.append([item])
            return normalized
        return [[responses]]

    @classmethod
    def _split_text_for_streaming(cls, text: str, num_chunks: int = 2) -> List[str]:
        if len(text) < 20:
            return [text]
        chunk_size = len(text) // num_chunks
        chunks = [text[i * chunk_size:(i + 1) * chunk_size] for i in range(num_chunks - 1)]
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
        self.response_index += 1
        self.requests.append(llm_request)

        if self.response_index >= len(self.responses):
            return

        parts = self.responses[self.response_index]

        if stream:
            for part in parts:
                if hasattr(part, "text") and part.text:
                    for chunk in self._split_text_for_streaming(part.text):
                        yield LlmResponse(
                            content=Content(role="model", parts=[Part.from_text(text=chunk)]),
                            partial=True,
                        )
                    yield LlmResponse(
                        content=Content(role="model", parts=[part]),
                        partial=None,
                    )
                else:
                    yield LlmResponse(content=Content(role="model", parts=[part]))
        else:
            yield LlmResponse(
                content=Content(role="model", parts=parts),
                partial=None,
            )


# ============================================================================
# MockTransport — wires input/output and emits Daily-style lifecycle events
# ============================================================================

class MockTransport(BaseTransport):
    """Mock transport for testing.

    Registers Daily-style event names (on_client_connected, on_participant_joined,
    etc.) so TestRunner can simulate participant lifecycle via _call_event_handler.
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self._bot_output = BotOutput()
        self._input_transport = MockInputTransport(
            parent_transport=self,
            bot_output=self._bot_output,
        )
        self._output_transport = MockOutputTransport(
            parent_transport=self,
            bot_output=self._bot_output,
        )

        for event_name in [
            "on_client_connected",
            "on_client_disconnected",
            "on_participant_joined",
            "on_participant_left",
            "on_before_leave",
        ]:
            self._register_event_handler(event_name)

    async def leave(self):
        """Trigger on_before_leave, mirroring DailyTransport.leave()."""
        await self._call_event_handler("on_before_leave")

    def input(self) -> "MockInputTransport":
        return self._input_transport

    def output(self) -> "MockOutputTransport":
        return self._output_transport

    def bot_output(self) -> BotOutput:
        return self._bot_output

    async def start_recording(self, streaming_settings=None, stream_id=None, force_new=None):
        """No-op; returns (None, None) matching DailyTransport.start_recording."""
        logger.debug(f"{self}: Mock start_recording called (no-op)")
        return None, None

    async def stop_recording(self):
        logger.debug(f"{self}: Mock stop_recording called (no-op)")


# ============================================================================
# MockVADAnalyzer
# ============================================================================

class MockVADAnalyzer(VADAnalyzer):
    """VAD analyzer for mock audio: non-zero bytes = speech, zeros = silence.

    Uses very short thresholds so tests don't need large silence padding.
    At 16kHz with num_frames_required=1, each frame is ~0.0001s.
    """

    def __init__(self, *, sample_rate: Optional[int] = None, params: Optional[VADParams] = None):
        if params is None:
            params = VADParams(
                confidence=0.5,
                start_secs=0.0001,
                stop_secs=0.0001,
                min_volume=0.0,
            )
        super().__init__(sample_rate=sample_rate, params=params)

    def num_frames_required(self) -> int:
        return 1

    def voice_confidence(self, buffer: bytes) -> float:
        return 1.0 if any(b != 0 for b in buffer) else 0.0


# ============================================================================
# MockInputTransport
# ============================================================================

class MockInputTransport(BaseInputTransport):
    """Input transport that lets tests inject user speech and client messages."""

    def __init__(self, parent_transport: MockTransport, bot_output: BotOutput, **kwargs):
        params = TransportParams(
            audio_in_enabled=True,
            audio_in_sample_rate=INPUT_SAMPLE_RATE,
            audio_in_channels=NUM_CHANNELS,
            audio_in_stream_on_start=True,
            audio_in_passthrough=True,
        )
        super().__init__(params=params, **kwargs)
        self._parent_transport = parent_transport
        self._bot_output = bot_output

    @override
    async def start(self, frame: StartFrame):
        await super().start(frame)
        await self.set_transport_ready(frame)

    @override
    async def stop(self, frame: EndFrame):
        await super().stop(frame)
        # Trigger on_before_leave, mirroring DailyTransport teardown behavior.
        await self._parent_transport.leave()

    @override
    async def cancel(self, frame: CancelFrame):
        await super().cancel(frame)

    async def push_speech(self, speech: str):
        """Inject user speech as PCM-aligned UTF-8 audio chunks."""
        for chunk in _chunk_string(speech):
            audio_bytes = encode_audio_text(chunk)  # padded=False, PCM-aligned
            await self.push_audio_frame(UserAudioRawFrame(
                audio=audio_bytes,
                sample_rate=INPUT_SAMPLE_RATE,
                num_channels=NUM_CHANNELS,
            ))

    async def push_message(
        self,
        message_type: str,
        data: Union[BaseModel, dict, None] = None,
    ):
        """Inject a client RTVI message into the pipeline."""
        frame = create_client_message_frame(message_type, data)
        await self.push_frame(frame, FrameDirection.DOWNSTREAM)

    async def push_silence(self):
        """Inject one silence frame to keep the audio stream alive."""
        await self.push_audio_frame(UserAudioRawFrame(
            audio=SILENCE,
            sample_rate=INPUT_SAMPLE_RATE,
            num_channels=NUM_CHANNELS,
        ))


# ============================================================================
# MockOutputTransport
# ============================================================================

class MockOutputTransport(BaseOutputTransport):
    """Output transport that captures RTVI messages via the public contract methods.

    Uses only write_audio_frame and send_message — does not peek at internal
    frame flow. Everything a real client would receive flows through send_message,
    which is tracked in BotOutput.
    """

    def __init__(self, parent_transport: MockTransport, bot_output: BotOutput, **kwargs):
        params = TransportParams(
            audio_out_enabled=True,
            audio_out_sample_rate=OUTPUT_SAMPLE_RATE,
            audio_out_channels=NUM_CHANNELS,
            audio_out_10ms_chunks=FAKE_AUDIO_CHUNK_MS // 10,
        )
        super().__init__(params=params, **kwargs)
        self._parent_transport = parent_transport
        self._bot_output = bot_output

    async def start(self, frame: StartFrame):
        await super().start(frame)
        await self.set_transport_ready(frame)

    async def write_audio_frame(self, frame: Frame) -> bool:
        return True  # no-op; playback is validated via RTVI messages

    async def write_video_frame(self, frame: Frame) -> bool:
        return True

    async def send_message(self, frame: Frame):
        """Capture every RTVI message the pipeline would send to a real client."""
        message_payload = getattr(frame, "message", None)
        payload_dict = None
        if isinstance(message_payload, BaseModel):
            payload_dict = message_payload.model_dump(exclude_none=True)
        elif isinstance(message_payload, dict):
            payload_dict = copy.deepcopy(message_payload)
        if payload_dict is not None:
            logger.debug(f"{self}: send_message type={payload_dict.get('type')}")
            await self._bot_output.append_message(payload_dict)


# ============================================================================
# MockTTSService
# ============================================================================

class MockTTSService(VqlTTSMixin, TTSService):
    """Mock TTS: encodes text as padded UTF-8 audio chunks.

    Follows the HTTP TTS pattern used by real pipecat services (OpenAI,
    ElevenLabs HTTP, etc.): pass push_start_frame=True and push_stop_frames=True
    so the base class manages TTSStartedFrame / TTSStoppedFrame and the audio
    context lifecycle. run_tts yields only TTSAudioRawFrame.

    Symmetric with MockSTTService: audio produced here can be decoded back to
    the original text.
    """

    def __init__(self, *, tts_delay: float = 0.0, **kwargs):
        super().__init__(
            sample_rate=OUTPUT_SAMPLE_RATE,
            push_start_frame=True,   # base class creates audio context + TTSStartedFrame
            push_stop_frames=True,   # base class pushes TTSStoppedFrame
            **kwargs,
        )
        self._tts_delay = tts_delay

    def can_generate_metrics(self) -> bool:
        return True

    def to_audio(self, text: str) -> AudioRawFrame:
        """Utility for tests: convert text directly to an AudioRawFrame."""
        return AudioRawFrame(
            audio=encode_audio_text(text, padded=True),
            sample_rate=OUTPUT_SAMPLE_RATE,
            num_channels=NUM_CHANNELS,
        )

    async def run_tts(self, text: str, context_id: str = "") -> AsyncGenerator[Frame, None]:
        """Yield TTSAudioRawFrame chunks for the given text.

        The base class handles TTSStartedFrame and TTSStoppedFrame; this method
        yields only audio data.
        """
        logger.debug(f"{self}: MockTTS synthesizing [{text!r}]")

        await self.start_ttfb_metrics()
        await self.start_tts_usage_metrics(text)

        # Split by word first (natural streaming), then by byte limit per packet.
        text_chunks: list[str] = []
        for word_chunk in _chunk_string(text):
            text_chunks.extend(
                _split_text_to_payload_chunks(word_chunk, max_bytes=FAKE_AUDIO_PACKET_BYTES)
            )

        for i, chunk in enumerate(text_chunks):
            if not chunk:
                continue
            if self._tts_delay:
                await asyncio.sleep(self._tts_delay)
            audio_frame = TTSAudioRawFrame(
                audio=encode_audio_text(chunk, padded=True),
                sample_rate=self.sample_rate or OUTPUT_SAMPLE_RATE,
                num_channels=NUM_CHANNELS,
            )
            audio_frame.id = i
            yield audio_frame
            if i == 0:
                await self.stop_ttfb_metrics()


# ============================================================================
# MockSTTService
# ============================================================================

class MockSTTService(STTService):
    """Mock STT: silence boundaries trigger final transcription.

    State is maintained across run_stt calls:
    - _buffer: accumulates decoded text from non-silence frames
    - _is_transcribing: True while in an active utterance

    Symmetric with MockTTSService: audio produced by MockTTSService can be
    decoded back to the original text by this service.
    """

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self._buffer = ""
        self._is_transcribing = False

    def can_generate_metrics(self) -> bool:
        return True

    def to_text(self, frame: AudioRawFrame) -> str:
        """Utility for tests: convert an audio frame back to text."""
        return decode_audio_text(frame.audio, padded=True) or ""

    async def run_stt(self, audio: bytes) -> AsyncGenerator[Frame, None]:
        """Emit InterimTranscriptionFrame per chunk, TranscriptionFrame on silence."""
        is_silence = audio == SILENCE or all(b == 0 for b in audio)

        if is_silence:
            if self._buffer:
                await self.start_ttfb_metrics()
                await self.start_processing_metrics()
                await self.push_frame(TranscriptionFrame(
                    text=self._buffer,
                    user_id=self._user_id,
                    timestamp=time_now_iso8601(),
                ))
                await self.stop_ttfb_metrics()
                await self.stop_processing_metrics()
                self._buffer = ""
            self._is_transcribing = False
        else:
            text = decode_audio_text(audio, padded=True)
            if text is None:
                yield None  # type: ignore
                return

            first_chunk = not self._is_transcribing
            self._buffer += text

            if first_chunk:
                await self.start_ttfb_metrics()
                await self.start_processing_metrics()
                self._is_transcribing = True
                await self.stop_ttfb_metrics()

            await self.push_frame(InterimTranscriptionFrame(
                text=self._buffer,
                user_id=self._user_id,
                timestamp=time_now_iso8601(),
            ))

        yield None  # type: ignore


# ============================================================================
# TestRunner — orchestrates a full ADK pipeline for conversational tests
# ============================================================================

class TestRunner:
    """Test harness for pipecat-adk pipelines.

    Sets up a complete pipeline with mock services and exposes a conversational
    DSL for driving tests:

        async with TestRunner(app=app) as runner:
            await runner.join()
            await runner.speak_and_wait_for_response("Hi")
            assert runner.transcript == [
                Turn("user", "Hi"),
                Turn("bot", "Hello!"),
            ]

    Args:
        app: The ADK App (name must be "agents" to match hardcoded session params).
             Mutually exclusive with ``llm_service``.
        llm_service: A pre-built LLM service to drive the pipeline (e.g.
             ``WebSocketLLMService``). Use this instead of ``app`` to test a
             non-ADK backend. No ADK session is created in this mode, so
             ``events()`` / ``session_state()`` are unavailable.
        tts_delay: Per-chunk TTS delay in seconds. Use ~0.05 for interruption
                   tests so the bot produces partial speech before being cut off.
    """

    def __init__(
        self,
        app: Optional[App] = None,
        *,
        llm_service: Optional[Any] = None,
        tts_delay: float = 0.0,
    ):
        if (app is None) == (llm_service is None):
            raise ValueError("Provide exactly one of 'app' or 'llm_service'")

        self.session_params = SessionParams(
            app_name="agents",
            session_id="test_session",
            user_id="test_user",
        )

        self.transport = MockTransport()
        self.mock_input = self.transport.input()
        self.mock_output = self.transport.output()
        self._bot_output = self.transport.bot_output()

        if llm_service is not None:
            # Non-ADK backend (e.g. WebSocketLLMService): no ADK session.
            self.session_service = None
            self.llm_service = llm_service
        else:
            self.session_service = InMemorySessionService()
            self.llm_service = AdkLLMService(
                agent=app.root_agent,
                session_service=self.session_service,
                session_params=self.session_params,
                plugins=app.plugins,
            )

        mock_vad = MockVADAnalyzer(sample_rate=INPUT_SAMPLE_RATE)
        user_params = LLMUserAggregatorParams(
            vad_analyzer=mock_vad,
            user_turn_strategies=UserTurnStrategies(
                start=[
                    VADUserTurnStartStrategy(),
                    TranscriptionUserTurnStartStrategy(),
                ],
                stop=[SpeechTimeoutUserTurnStopStrategy(user_speech_timeout=0.2)],
            ),
        )
        context_aggregators = self.llm_service.create_context_aggregator(
            user_params=user_params
        )

        self._rtvi = RTVIProcessor()

        self.pipeline = Pipeline([
            self.mock_input,
            self._rtvi,
            MockSTTService(),
            context_aggregators.user(),
            self.llm_service,
            MockTTSService(tts_delay=tts_delay),
            self.mock_output,
            context_aggregators.assistant(),
        ])

        self.task: Optional[PipelineTask] = None
        self.runner: Optional[PipelineRunner] = None
        self._pipeline_task: Optional[asyncio.Task] = None
        self._joined = False

    # ── Black-box assertion properties ──────────────────────────────────────

    @property
    def messages(self) -> List[Dict[str, Any]]:
        """All RTVI messages sent to the client."""
        return self._bot_output.messages

    @property
    def transcript(self) -> List[Turn]:
        """Conversation transcript built from RTVI speaking events."""
        return self._bot_output.transcript

    @property
    def bot_messages(self) -> List[str]:
        return [t.text for t in self.transcript if t.speaker == "bot"]

    @property
    def last_bot_message(self) -> str:
        if not self.bot_messages:
            raise ValueError("No bot messages yet — did you call wait_for_response()?")
        return self.bot_messages[-1]

    @property
    def client_state(self) -> Dict[str, Any]:
        return self._bot_output.client_state

    @property
    def errors(self) -> List[str]:
        """All RTVI error messages received from the pipeline."""
        return self._bot_output.errors

    # ── Gray-box inspection ──────────────────────────────────────────────────

    async def session_state(self) -> dict:
        if self.session_service is None:
            raise RuntimeError("session_state() requires an ADK-backed TestRunner")
        session = await self.session_service.get_session(**self.session_params.model_dump())
        return session.state if session else {}

    async def events(self):
        if self.session_service is None:
            raise RuntimeError("events() requires an ADK-backed TestRunner")
        session = await self.session_service.get_session(**self.session_params.model_dump())
        return session.events if session else []

    # ── Conversational API ───────────────────────────────────────────────────

    async def _ensure_started(self):
        if self.task is not None:
            return
        self.task = PipelineTask(
            self.pipeline,
            params=PipelineParams(allow_interruptions=True),
            observers=[RTVIObserver(self._rtvi), AdkDebugLogObserver()],
        )
        self.runner = PipelineRunner()
        self._pipeline_task = asyncio.create_task(self.runner.run(self.task))
        # Allow StartFrame to fully propagate before any test input.
        await asyncio.sleep(0.2)

    def _ensure_joined(self):
        if not self._joined:
            raise RuntimeError("Call TestRunner.join() before driving the pipeline")

    async def join(self):
        """Simulate a participant joining (fires on_participant_joined and on_client_connected)."""
        if self.task is None:
            raise RuntimeError(
                "Pipeline not started. Use 'async with TestRunner(app) as runner:'"
            )
        if self._joined:
            return
        participant = {"id": "test-user", "name": "Test User"}
        await self.transport._call_event_handler("on_participant_joined", participant)
        await self.transport._call_event_handler("on_client_connected", participant)
        self._joined = True

    async def speak(self, speech: str):
        self._ensure_joined()
        await self.mock_input.push_speech(speech)

    async def speak_and_wait_for_response(self, speech: str, timeout: float = 60.0):
        await self.speak(speech)
        await self.wait_for_response(timeout=timeout)

    async def push_message(self, message_type: str, data: Union[BaseModel, dict, None] = None):
        self._ensure_joined()
        await self.mock_input.push_message(message_type, data)

    async def queue_frame(self, frame: Frame):
        """Inject a frame directly into the pipeline task."""
        self._ensure_joined()
        if self.task is None:
            raise RuntimeError("Pipeline task not initialized")
        await self.task.queue_frame(frame)

    async def stay_silent(self, iterations: int = 10, delay: float = 0.01):
        """Push silence frames to drain async processing without triggering a response."""
        self._ensure_joined()
        for _ in range(iterations):
            await self.mock_input.push_silence()
            await asyncio.sleep(delay)

    async def wait_for(
        self,
        predicate: Callable[[BotOutput, List[Dict[str, Any]]], bool],
        timeout: float = 60.0,
    ):
        """Poll until predicate(bot_output, delta_messages) returns True."""
        self._ensure_joined()
        start_index = len(self.messages)
        loop = asyncio.get_running_loop()
        deadline = loop.time() + timeout

        while True:
            delta = self._bot_output.messages[start_index:]
            if predicate(self._bot_output, delta):
                return
            if loop.time() >= deadline:
                raise TimeoutError(f"wait_for timed out after {timeout}s")
            await self.mock_input.push_silence()
            await asyncio.sleep(0.01)

    async def wait_for_bot_to_start_speaking(self, timeout: float = 60.0):
        def _started(_: BotOutput, delta: List[Dict[str, Any]]) -> bool:
            return any(m.get("type") == "bot-started-speaking" for m in delta)
        await self.wait_for(_started, timeout=timeout)

    async def wait_for_response(self, timeout: float = 60.0):
        """Wait for a complete bot speaking turn (started → stopped)."""
        def _finished(_: BotOutput, delta: List[Dict[str, Any]]) -> bool:
            started = False
            for m in delta:
                if m.get("type") == "bot-started-speaking":
                    started = True
                if started and m.get("type") == "bot-stopped-speaking":
                    return True
            return False
        await self.wait_for(_finished, timeout=timeout)

    async def interrupt_bot(self, message: str, timeout: float = 60.0):
        """Wait for the bot to start speaking, then inject an interruption."""
        await self.wait_for_bot_to_start_speaking(timeout=timeout)
        await self.speak(message)

    # ── Lifecycle ────────────────────────────────────────────────────────────

    async def cleanup(self):
        if self._pipeline_task and not self._pipeline_task.done():
            self._pipeline_task.cancel()
            try:
                await self._pipeline_task
            except asyncio.CancelledError:
                pass

    async def __aenter__(self):
        if self.session_service is not None:
            session = await self.session_service.create_session(
                **self.session_params.model_dump()
            )
            assert session is not None
        await self._ensure_started()
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        await self.cleanup()
        return False
