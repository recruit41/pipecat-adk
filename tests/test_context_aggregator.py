import unittest

from pipecat.frames.frames import (
    UserStartedSpeakingFrame,
    UserStoppedSpeakingFrame,
    TranscriptionFrame,
    InterimTranscriptionFrame,
    BotStartedSpeakingFrame,
    BotStoppedSpeakingFrame,
)
from pipecat.utils.time import time_now_iso8601
from pipecat.pipeline.pipeline import Pipeline
from pipecat.tests.utils import run_test, SleepFrame
from tests.mocks import MockLLM
from pipecat_adk import AdkBasedLLMService, SessionParams
from google.adk.agents import Agent
from google.adk.sessions import InMemorySessionService

class TestContextAggregator(unittest.IsolatedAsyncioTestCase):
    async def test_basic_interaction(self):
        session_service = InMemorySessionService()
        session_params = SessionParams(
            app_name="agents",
            session_id="test_session",
            user_id="test_user",
        )
        await session_service.create_session(
            app_name=session_params.app_name,
            session_id=session_params.session_id,
            user_id=session_params.user_id,
        )
        
        mock_llm = MockLLM.single("The capital of India is New Delhi.")
        agent = Agent(
            name="test_agent",
            description="A test agent",
            model=mock_llm
        )

        adk_service = AdkBasedLLMService(
            session_service=session_service,
            session_params=session_params,
            agent=agent,
            api_key="test_api_key",
        )
        context_aggregators = adk_service.create_context_aggregator()
        pipeline = Pipeline([
            context_aggregators.user(),
            adk_service,
            context_aggregators.assistant()
        ])

        frames_to_send = [
            UserStartedSpeakingFrame(),
            TranscriptionFrame(user_id="test_user", timestamp=time_now_iso8601(), text="What is the capital of India?"),
            SleepFrame(sleep=0.01),  # Give time for transcription to be processed
            UserStoppedSpeakingFrame(),
            SleepFrame(sleep=0.1),  # Give time for aggregation to be processed
        ]
        await run_test(
            pipeline,
            frames_to_send=frames_to_send
        )
        session = await session_service.get_session(
            app_name=session_params.app_name,
            session_id=session_params.session_id,   
            user_id=session_params.user_id
        )
        assert session
        self.assertTrue(len(session.events) == 2)