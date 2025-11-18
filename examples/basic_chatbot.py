"""Basic chatbot example using pipecat-adk.

This example demonstrates a minimal working chatbot using:
- Google ADK agents for conversation management
- Pipecat for audio/video pipeline
- Automatic interruption handling

Requirements:
    - Set GEMINI_API_KEY environment variable
    - Install: uv pip install -e /path/to/pipecat-adk
"""

import asyncio
import os

from google.adk.agents import Agent
from google.adk.sessions import InMemorySessionService
from pipecat.pipeline.pipeline import Pipeline
from pipecat.pipeline.task import PipelineParams, PipelineTask
from pipecat.services.google import GoogleSTTService, GoogleTTSService
from pipecat.transcriptions.language import Language
from pipecat.transports.daily.transport import DailyParams, DailyTransport

from pipecat_adk import AdkBasedLLMService, SessionParams


async def main():
    """Run the basic chatbot."""

    # Get API key from environment
    api_key = os.getenv("GEMINI_API_KEY")
    if not api_key:
        raise ValueError("GEMINI_API_KEY environment variable not set")

    # Create ADK agent
    agent = Agent(
        name="helpful_assistant",
        model="gemini-2.0-flash",
        instruction=(
            "You are a helpful AI assistant. Be concise and friendly. "
            "If the user interrupts you, acknowledge it gracefully."
        ),
    )

    # Create session service (in-memory for this example)
    session_service = InMemorySessionService()

    # Create session parameters
    session_params = SessionParams(
        app_name="basic-chatbot",
        user_id="user",  # ADK requires "user" for web compatibility
        session_id="session-123",
    )

    # Create ADK-based LLM service
    llm = AdkBasedLLMService(
        session_service=session_service,
        session_params=session_params,
        agent=agent,
        api_key=api_key,
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
        voice_id="en-US-Wavenet-C",
        params=GoogleTTSService.InputParams(language=Language.EN_US),
    )

    # Create Daily transport for audio/video
    # You'll need to provide your own room_url and token
    room_url = os.getenv("DAILY_ROOM_URL")
    bot_token = os.getenv("DAILY_BOT_TOKEN")

    if not room_url or not bot_token:
        print("Warning: DAILY_ROOM_URL and DAILY_BOT_TOKEN not set")
        print("This example requires Daily.co for audio/video")
        print("\nTo run:")
        print("1. Create a Daily.co room")
        print("2. Set DAILY_ROOM_URL environment variable")
        print("3. Set DAILY_BOT_TOKEN environment variable")
        return

    transport = DailyTransport(
        room_url,
        bot_token,
        "Chatbot",
        DailyParams(
            audio_in_enabled=True,
            audio_out_enabled=True,
            transcription_enabled=False,
            vad_enabled=True,
        ),
    )

    # Create pipeline
    pipeline = Pipeline(
        [
            transport.input(),  # Audio input from user
            stt,  # Speech-to-text
            context_aggregator.user(),  # Package user input for ADK
            llm,  # ADK agent
            tts,  # Text-to-speech
            transport.output(),  # Audio output to user
            context_aggregator.assistant(),  # Handle interruptions
        ]
    )

    # Create pipeline task
    task = PipelineTask(pipeline, params=PipelineParams(allow_interruptions=True))

    # Add transport event handlers
    @transport.event_handler("on_participant_joined")
    async def on_participant_joined(transport, participant):
        print(f"Participant joined: {participant['id']}")
        # Greet the user
        await task.queue_frames(
            [
                context_aggregator.user()._context.add_message(
                    {"role": "user", "content": "<system>User joined. Greet them.</system>"}
                )
            ]
        )

    @transport.event_handler("on_participant_left")
    async def on_participant_left(transport, participant, reason):
        print(f"Participant left: {participant['id']} (reason: {reason})")
        await task.cancel()

    print("Starting chatbot...")
    print(f"Room URL: {room_url}")
    print("\nFeatures:")
    print("- Natural conversation with AI")
    print("- Automatic interruption handling")
    print("- Voice input/output")
    print("\nPress Ctrl+C to stop")

    try:
        await task.run()
    except KeyboardInterrupt:
        print("\nStopping chatbot...")
    finally:
        await transport.cleanup()


if __name__ == "__main__":
    asyncio.run(main())
