"""Basic assistant bot using pipecat-adk with small-webrtc-prebuilt.

This example demonstrates:
- Google ADK agents for conversation management
- Pipecat for audio/video pipeline
- Automatic interruption handling via InterruptionHandlerPlugin
- WebRTC transport via small-webrtc-prebuilt
"""

import os
import sys

from dotenv import load_dotenv
from google.adk.sessions import InMemorySessionService
from loguru import logger

from pipecat.audio.vad.silero import SileroVADAnalyzer
from pipecat.pipeline.pipeline import Pipeline
from pipecat.pipeline.runner import PipelineRunner
from pipecat.pipeline.task import PipelineParams, PipelineTask
from pipecat.services.google.stt import GoogleSTTService
from pipecat.services.google.tts import GoogleTTSService
from pipecat.transcriptions.language import Language
from pipecat.transports.base_transport import TransportParams
from pipecat.transports.network.small_webrtc import SmallWebRTCTransport

from pipecat_adk import AdkBasedLLMService, SessionParams
from agent import app

load_dotenv(override=True)

logger.remove(0)
logger.add(sys.stderr, level="DEBUG")


async def run_bot(webrtc_connection):
    """Run the assistant bot with the given WebRTC connection."""

    # Get API key from environment
    api_key = os.getenv("GEMINI_API_KEY")
    if not api_key:
        raise ValueError("GEMINI_API_KEY environment variable not set")

    # Create session service (in-memory for this example)
    session_service = InMemorySessionService()

    # Create session parameters
    session_params = SessionParams(
        app_name=app.name,  # Use the app name from agent.py
        user_id="user",
        session_id="session-123",
    )
    await session_service.create_session(**session_params.model_dump())

    # Create ADK-based LLM service with the app from agent.py
    # The app includes the agent and InterruptionHandlerPlugin
    llm = AdkBasedLLMService(
        session_service=session_service,
        session_params=session_params,
        app=app
    )

    # Create context aggregators
    context_aggregator = llm.create_context_aggregator()

    # Create STT service (speech-to-text)
    stt = GoogleSTTService(
        params=GoogleSTTService.InputParams(
            languages=Language.EN_US,
            model="latest_long",
            enable_automatic_punctuation=True,
            enable_interim_results=True,
        )
    )

    # Create TTS service (text-to-speech)
    tts = GoogleTTSService(
        voice_id="en-IN-Chirp3-HD-Achird",
        params=GoogleTTSService.InputParams(language=Language.EN_IN),
    )

    # Create transport with video and audio enabled
    transport_params = TransportParams(
        audio_in_enabled=True,
        audio_out_enabled=True,
        vad_enabled=True,
        vad_analyzer=SileroVADAnalyzer(),
    )

    pipecat_transport = SmallWebRTCTransport(
        webrtc_connection=webrtc_connection, params=transport_params
    )

    # Create pipeline
    pipeline = Pipeline(
        [
            pipecat_transport.input(),  # Audio input from user
            stt,  # Speech-to-text
            context_aggregator.user(),  # Package user input for ADK
            llm,  # ADK agent
            tts,  # Text-to-speech
            pipecat_transport.output(),  # Audio output to user
            context_aggregator.assistant(),  # Handle interruptions
        ]
    )

    # Create pipeline task
    task = PipelineTask(
        pipeline,
        params=PipelineParams(allow_interruptions=True),
    )

    # Add transport event handlers
    @pipecat_transport.event_handler("on_client_connected")
    async def on_client_connected(transport, client):
        logger.info("Client connected")
        # Send greeting prompt to start the conversation
        await task.queue_frames([context_aggregator.user().get_context_frame()])

    @pipecat_transport.event_handler("on_client_disconnected")
    async def on_client_disconnected(transport, client):
        logger.info("Client disconnected")
        await task.cancel()

    # Run the pipeline
    runner = PipelineRunner(handle_sigint=False)
    await runner.run(task)
