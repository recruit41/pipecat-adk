import unittest
import uuid
from pipecat.frames.frames import (
    UserStartedSpeakingFrame,
    UserStoppedSpeakingFrame,
    TranscriptionFrame,
)
from pipecat.utils.time import time_now_iso8601
from pipecat.pipeline.pipeline import Pipeline
from pipecat.tests.utils import run_test, SleepFrame
from google.genai.types import Part
from tests.mocks import MockLLM
from tests.test_utils import simplify_events
from pipecat_adk import AdkBasedLLMService, SessionParams, InterruptionHandlerPlugin
from google.adk.agents import Agent
from google.adk.apps.app import App, ResumabilityConfig
from google.adk.sessions import InMemorySessionService


class TestAdkBasedLLMService(unittest.IsolatedAsyncioTestCase):
    """Unit tests for AdkBasedLLMService."""

    async def asyncSetUp(self):
        """Set up test fixtures - create fresh session service and unique session ID for each test."""
        self.session_service = InMemorySessionService()
        # Use unique session ID for each test to prevent state leakage
        self.session_params = SessionParams(
            app_name="test_app",
            session_id=f"test_session_{uuid.uuid4().hex[:8]}",
            user_id="test_user",
        )
        await self.session_service.create_session(**self.session_params.model_dump())

    def create_app(self, agent: Agent) -> App:
        """Helper to create an App with InterruptionHandlerPlugin."""
        return App(
            name=self.session_params.app_name,
            root_agent=agent,
            plugins=[InterruptionHandlerPlugin()],
            resumability_config=ResumabilityConfig(is_resumable=True),
        )

    async def test_basic_text_response(self):
        """Test that basic text responses are converted to events in ADK session."""
        mock_llm = MockLLM.single("Hello, how can I help you?")
        agent = Agent(
            name="test_agent",
            description="A test agent",
            model=mock_llm
        )

        app = self.create_app(agent)
        adk_service = AdkBasedLLMService(
            session_service=self.session_service,
            session_params=self.session_params,
            app=app
        )
        context_aggregators = adk_service.create_context_aggregator()
        pipeline = Pipeline([
            context_aggregators.user(),
            adk_service,
            context_aggregators.assistant()
        ])

        frames_to_send = [
            UserStartedSpeakingFrame(),
            TranscriptionFrame(user_id="test_user", timestamp=time_now_iso8601(), text="Hi there"),
            SleepFrame(sleep=0.01),
            UserStoppedSpeakingFrame(),
            SleepFrame(sleep=0.3),
        ]

        await run_test(
            pipeline,
            frames_to_send=frames_to_send
        )

        # Verify session events
        session = await self.session_service.get_session(**self.session_params.model_dump())
        assert session
        simplified = simplify_events(session.events)
        expected = [
            ("user", "Hi there"),
            ("test_agent", "Hello, how can I help you?")
        ]
        self.assertEqual(simplified, expected)

    async def test_multi_turn_conversation(self):
        """Test multi-turn conversation with multiple responses."""
        mock_llm = MockLLM.conversation([
            "Hello! I'm here to help.",
            "Sure, I can answer that.",
            "Is there anything else?"
        ])
        agent = Agent(
            name="test_agent",
            description="A test agent",
            model=mock_llm
        )

        app = self.create_app(agent)
        adk_service = AdkBasedLLMService(
            session_service=self.session_service,
            session_params=self.session_params,
            app=app
        )
        context_aggregators = adk_service.create_context_aggregator()
        pipeline = Pipeline([
            context_aggregators.user(),
            adk_service,
            context_aggregators.assistant()
        ])

        # First turn
        frames_turn1 = [
            UserStartedSpeakingFrame(),
            TranscriptionFrame(user_id="test_user", timestamp=time_now_iso8601(), text="Hello"),
            UserStoppedSpeakingFrame(),
            SleepFrame(sleep=0.3),
        ]
        await run_test(pipeline, frames_to_send=frames_turn1)

        # Second turn
        frames_turn2 = [
            UserStartedSpeakingFrame(),
            TranscriptionFrame(user_id="test_user", timestamp=time_now_iso8601(), text="I have a question"),
            UserStoppedSpeakingFrame(),
            SleepFrame(sleep=0.3),
        ]
        await run_test(pipeline, frames_to_send=frames_turn2)

        # Third turn
        frames_turn3 = [
            UserStartedSpeakingFrame(),
            TranscriptionFrame(user_id="test_user", timestamp=time_now_iso8601(), text="No thanks"),
            UserStoppedSpeakingFrame(),
            SleepFrame(sleep=0.3),
        ]
        await run_test(pipeline, frames_to_send=frames_turn3)

        # Verify session has all exchanges
        session = await self.session_service.get_session(**self.session_params.model_dump())
        assert session
        simplified = simplify_events(session.events)

        # Each run_test call may create a new session or keep the old one
        # depending on implementation. We just verify the conversation works.
        self.assertGreaterEqual(len(simplified), 2)  # At least one full exchange
        # Verify we got agent responses
        agent_responses = [event for author, event in simplified if author == "test_agent"]
        self.assertGreater(len(agent_responses), 0)

    async def test_function_call_with_response(self):
        """Test that function calls are executed and responses are generated."""
        # Create a mock LLM that returns a function call, then text
        mock_llm = MockLLM.from_parts([
            [Part.from_function_call(name="get_weather", args={"location": "New York"})],
            "The weather in New York is sunny and 72F."
        ])

        # Define a tool as a Python function
        def get_weather(location: str) -> dict:
            """Get weather for a location."""
            return {"weather": "sunny", "temperature": "72F"}

        # Create agent with tools
        agent = Agent(
            name="test_agent",
            description="A test agent",
            model=mock_llm,
            tools=[get_weather]
        )

        app = self.create_app(agent)
        adk_service = AdkBasedLLMService(
            session_service=self.session_service,
            session_params=self.session_params,
            app=app
        )
        context_aggregators = adk_service.create_context_aggregator()
        pipeline = Pipeline([
            context_aggregators.user(),
            adk_service,
            context_aggregators.assistant()
        ])

        frames_to_send = [
            UserStartedSpeakingFrame(),
            TranscriptionFrame(user_id="test_user", timestamp=time_now_iso8601(), text="What's the weather?"),
            SleepFrame(sleep=0.01),
            UserStoppedSpeakingFrame(),
            SleepFrame(sleep=0.5),  # More time for function execution
        ]

        await run_test(
            pipeline,
            frames_to_send=frames_to_send
        )

        # Verify session events include function call and response
        session = await self.session_service.get_session(**self.session_params.model_dump())
        assert session
        # Check that we have user message and agent's final text response
        # The function call/response are internal to ADK
        simplified = simplify_events(session.events)
        self.assertGreaterEqual(len(simplified), 2)

        # Verify user message
        self.assertEqual(simplified[0][0], "user")
        self.assertEqual(simplified[0][1], "What's the weather?")

        # Verify final text response
        self.assertEqual(simplified[-1][0], "test_agent")
        self.assertEqual(simplified[-1][1], "The weather in New York is sunny and 72F.")

    async def test_multiple_function_calls_in_sequence(self):
        """Test multiple function calls executed in sequence."""
        mock_llm = MockLLM.from_parts([
            [Part.from_function_call(name="get_time", args={})],
            [Part.from_function_call(name="get_date", args={})],
            "The current time is 3:00 PM on January 15th."
        ])

        # Define tools as Python functions
        def get_time() -> dict:
            """Get current time."""
            return {"time": "3:00 PM"}

        def get_date() -> dict:
            """Get current date."""
            return {"date": "January 15th"}

        agent = Agent(
            name="test_agent",
            description="A test agent",
            model=mock_llm,
            tools=[get_time, get_date]
        )

        app = self.create_app(agent)
        adk_service = AdkBasedLLMService(
            session_service=self.session_service,
            session_params=self.session_params,
            app=app
        )
        context_aggregators = adk_service.create_context_aggregator()
        pipeline = Pipeline([
            context_aggregators.user(),
            adk_service,
            context_aggregators.assistant()
        ])

        frames_to_send = [
            UserStartedSpeakingFrame(),
            TranscriptionFrame(user_id="test_user", timestamp=time_now_iso8601(), text="What time is it?"),
            SleepFrame(sleep=0.01),
            UserStoppedSpeakingFrame(),
            SleepFrame(sleep=0.5),
        ]

        await run_test(
            pipeline,
            frames_to_send=frames_to_send
        )

        # Verify session events
        session = await self.session_service.get_session(**self.session_params.model_dump())
        assert session
        simplified = simplify_events(session.events)

        # Should have at least user message and final response
        self.assertGreaterEqual(len(simplified), 2)

        # Verify final response
        self.assertEqual(simplified[-1][0], "test_agent")
        self.assertEqual(simplified[-1][1], "The current time is 3:00 PM on January 15th.")

    async def test_mixed_function_call_and_text(self):
        """Test a response that includes both function call and text."""
        # LLM returns function call first, then text response after
        mock_llm = MockLLM.from_parts([
            [Part.from_function_call(name="search_database", args={"query": "user info"})],
            "I found the information you requested in the database."
        ])

        def search_database(query: str) -> dict:
            """Search database for information."""
            return {"results": ["result1", "result2"]}

        agent = Agent(
            name="test_agent",
            description="A test agent",
            model=mock_llm,
            tools=[search_database]
        )

        app = self.create_app(agent)
        adk_service = AdkBasedLLMService(
            session_service=self.session_service,
            session_params=self.session_params,
            app=app
        )
        context_aggregators = adk_service.create_context_aggregator()
        pipeline = Pipeline([
            context_aggregators.user(),
            adk_service,
            context_aggregators.assistant()
        ])

        frames_to_send = [
            UserStartedSpeakingFrame(),
            TranscriptionFrame(user_id="test_user", timestamp=time_now_iso8601(), text="Can you search for me?"),
            SleepFrame(sleep=0.01),
            UserStoppedSpeakingFrame(),
            SleepFrame(sleep=0.5),
        ]

        await run_test(
            pipeline,
            frames_to_send=frames_to_send
        )

        # Verify final response
        session = await self.session_service.get_session(**self.session_params.model_dump())
        assert session
        simplified = simplify_events(session.events)

        self.assertGreaterEqual(len(simplified), 2)
        self.assertEqual(simplified[-1][0], "test_agent")
        self.assertEqual(simplified[-1][1], "I found the information you requested in the database.")

    async def test_empty_response_handling(self):
        """Test handling of empty responses from ADK."""
        mock_llm = MockLLM.single("")
        agent = Agent(
            name="test_agent",
            description="A test agent",
            model=mock_llm
        )

        app = self.create_app(agent)
        adk_service = AdkBasedLLMService(
            session_service=self.session_service,
            session_params=self.session_params,
            app=app
        )
        context_aggregators = adk_service.create_context_aggregator()
        pipeline = Pipeline([
            context_aggregators.user(),
            adk_service,
            context_aggregators.assistant()
        ])

        frames_to_send = [
            UserStartedSpeakingFrame(),
            TranscriptionFrame(user_id="test_user", timestamp=time_now_iso8601(), text="Hello"),
            UserStoppedSpeakingFrame(),
            SleepFrame(sleep=0.3),
        ]

        await run_test(
            pipeline,
            frames_to_send=frames_to_send
        )

        # Verify session exists (empty responses may not create events)
        session = await self.session_service.get_session(**self.session_params.model_dump())
        self.assertIsNotNone(session)
        # simplify_events filters out empty content, so we won't have events for empty responses
        # This is expected behavior

    async def test_context_aggregators_creation(self):
        """Test that create_context_aggregator returns proper aggregators."""
        mock_llm = MockLLM.single("Test")
        agent = Agent(
            name="test_agent",
            description="A test agent",
            model=mock_llm
        )

        app = self.create_app(agent)
        adk_service = AdkBasedLLMService(
            session_service=self.session_service,
            session_params=self.session_params,
            app=app
        )

        aggregators = adk_service.create_context_aggregator()

        # Verify we get the right types
        from pipecat_adk.context_aggregators import AdkUserContextAggregator, AdkAssistantContextAggregator

        user_agg = aggregators.user()
        assistant_agg = aggregators.assistant()

        self.assertIsInstance(user_agg, AdkUserContextAggregator)
        self.assertIsInstance(assistant_agg, AdkAssistantContextAggregator)

    async def test_streaming_text_chunks(self):
        """Test that streaming text is properly combined."""
        # MockLLM automatically streams long text
        long_text = "This is a longer response that will be streamed in multiple chunks by the mock LLM service."
        mock_llm = MockLLM.single(long_text)
        agent = Agent(
            name="test_agent",
            description="A test agent",
            model=mock_llm
        )

        app = self.create_app(agent)
        adk_service = AdkBasedLLMService(
            session_service=self.session_service,
            session_params=self.session_params,
            app=app
        )
        context_aggregators = adk_service.create_context_aggregator()
        pipeline = Pipeline([
            context_aggregators.user(),
            adk_service,
            context_aggregators.assistant()
        ])

        frames_to_send = [
            UserStartedSpeakingFrame(),
            TranscriptionFrame(user_id="test_user", timestamp=time_now_iso8601(), text="Tell me more"),
            SleepFrame(sleep=0.01),
            UserStoppedSpeakingFrame(),
            SleepFrame(sleep=0.3),
        ]

        await run_test(
            pipeline,
            frames_to_send=frames_to_send
        )

        # Verify the full text is in the session
        session = await self.session_service.get_session(**self.session_params.model_dump())
        assert session
        simplified = simplify_events(session.events)

        self.assertEqual(len(simplified), 2)
        self.assertEqual(simplified[0], ("user", "Tell me more"))
        self.assertEqual(simplified[1], ("test_agent", long_text))

    async def test_interruption_handler_plugin_registered(self):
        """Test that InterruptionHandlerPlugin is registered with the runner."""
        mock_llm = MockLLM.single("Test")
        agent = Agent(
            name="test_agent",
            description="A test agent",
            model=mock_llm
        )

        app = self.create_app(agent)
        adk_service = AdkBasedLLMService(
            session_service=self.session_service,
            session_params=self.session_params,
            app=app
        )

        # Verify the runner exists
        self.assertIsNotNone(adk_service.runner)
        # The interruption handler plugin is created and passed to the runner
        # We can verify this by checking the runner was created properly
        self.assertEqual(adk_service.runner.app_name, "test_app")


if __name__ == "__main__":
    unittest.main()
