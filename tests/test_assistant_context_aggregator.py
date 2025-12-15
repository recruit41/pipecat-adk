import unittest
from pipecat.frames.frames import (
    LLMFullResponseStartFrame,
    TTSTextFrame,
    InterruptionFrame,
)
from pipecat.tests.utils import run_test, SleepFrame
from google.adk.sessions import InMemorySessionService
from google.adk.agents import Agent
from google.adk.apps.app import App, ResumabilityConfig
from google.adk.runners import Runner
from pipecat_adk import SessionParams, InterruptionHandlerPlugin
from pipecat_adk.context_aggregators import AdkAssistantContextAggregator
from tests.mocks import MockLLM
from tests.test_utils import simplify_events


class TestAdkAssistantContextAggregator(unittest.IsolatedAsyncioTestCase):
    """Unit tests for AdkAssistantContextAggregator in isolation."""

    async def asyncSetUp(self):
        """Set up test fixtures."""
        self.session_service = InMemorySessionService()
        self.session_params = SessionParams(
            app_name="test_app",
            session_id="test_session",
            user_id="test_user",
        )

        # Create session
        await self.session_service.create_session(**self.session_params.model_dump())

        # Create mock agent, app, and runner
        mock_llm = MockLLM.single("Test response")
        self.agent = Agent(
            name="test_agent",
            description="A test agent",
            model=mock_llm
        )
        app = App(
            name=self.session_params.app_name,
            root_agent=self.agent,
            plugins=[InterruptionHandlerPlugin()],
            resumability_config=ResumabilityConfig(is_resumable=True),
        )
        self.runner = Runner(
            app=app,
            session_service=self.session_service
        )

    async def test_text_accumulation_between_start_and_end(self):
        """Test that text frames are accumulated between LLM start and end frames."""
        aggregator = AdkAssistantContextAggregator(
            session_service=self.session_service,
            session_params=self.session_params,
            runner=self.runner
        )

        frames_to_send = [
            LLMFullResponseStartFrame(),
            TTSTextFrame(text="Hello "),
            TTSTextFrame(text="world!"),
        ]

        await run_test(aggregator, frames_to_send=frames_to_send)

        # Check that text is being accumulated
        # Note: aggregator adds space before each frame, so "Hello " + " world!" = "Hello  world!"
        self.assertEqual(aggregator._aggregation, "Hello  world!")

    async def test_interruption_with_accumulated_text(self):
        """Test that interruption creates a synthetic event when text is accumulated."""
        aggregator = AdkAssistantContextAggregator(
            session_service=self.session_service,
            session_params=self.session_params,
            runner=self.runner
        )

        frames_to_send = [
            LLMFullResponseStartFrame(),
            TTSTextFrame(text="The capital of India is New Delhi."),
            SleepFrame(sleep=0.01),  # Give time for text to be processed
            InterruptionFrame(),
        ]

        await run_test(aggregator, frames_to_send=frames_to_send)

        # Verify the session has the interruption event
        session = await self.session_service.get_session(
            app_name=self.session_params.app_name,
            session_id=self.session_params.session_id,
            user_id=self.session_params.user_id
        )
        assert session

        # Should have one event (the interruption event)
        self.assertEqual(len(session.events), 1)

        # Check the event content using simplify_events
        simplified = simplify_events(session.events)
        self.assertEqual(len(simplified), 1)
        author, content = simplified[0]
        self.assertEqual(author, "test_agent")
        self.assertIn("<interruption>", content)
        self.assertIn("The capital of India is New Delhi.", content)
        self.assertIn("</interruption>", content)

    async def test_interruption_without_text_does_nothing(self):
        """Test that interruption without accumulated text doesn't create an event."""
        aggregator = AdkAssistantContextAggregator(
            session_service=self.session_service,
            session_params=self.session_params,
            runner=self.runner
        )

        frames_to_send = [
            InterruptionFrame(),
        ]

        await run_test(aggregator, frames_to_send=frames_to_send)

        # Verify no events were added to the session
        session = await self.session_service.get_session(
            app_name=self.session_params.app_name,
            session_id=self.session_params.session_id,
            user_id=self.session_params.user_id
        )
        assert session

        self.assertEqual(len(session.events), 0)

    async def test_partial_interruption(self):
        """Test interruption with only part of the response spoken."""
        aggregator = AdkAssistantContextAggregator(
            session_service=self.session_service,
            session_params=self.session_params,
            runner=self.runner
        )

        frames_to_send = [
            LLMFullResponseStartFrame(),
            TTSTextFrame(text="The capital of India is"),
            SleepFrame(sleep=0.01),  # Give time for text to be processed
            InterruptionFrame(),
        ]

        await run_test(aggregator, frames_to_send=frames_to_send)

        # Verify the interruption event has only the partial text
        session = await self.session_service.get_session(
            app_name=self.session_params.app_name,
            session_id=self.session_params.session_id,
            user_id=self.session_params.user_id
        )
        assert session

        # Check using simplify_events
        simplified = simplify_events(session.events)
        self.assertEqual(len(simplified), 1)
        author, content = simplified[0]
        self.assertEqual(author, "test_agent")
        self.assertIn("The capital of India is", content)
        self.assertNotIn("New Delhi", content)

    async def test_text_ignored_before_llm_start(self):
        """Test that text frames before LLMFullResponseStartFrame are ignored."""
        aggregator = AdkAssistantContextAggregator(
            session_service=self.session_service,
            session_params=self.session_params,
            runner=self.runner
        )

        frames_to_send = [
            TTSTextFrame(text="This should be ignored"),
            LLMFullResponseStartFrame(),
            TTSTextFrame(text="This should be captured"),
        ]

        await run_test(aggregator, frames_to_send=frames_to_send)

        # Should only have the text sent after start
        self.assertEqual(aggregator._aggregation, "This should be captured")

    async def test_aggregation_cleared_after_interruption(self):
        """Test that aggregation is cleared after interruption."""
        aggregator = AdkAssistantContextAggregator(
            session_service=self.session_service,
            session_params=self.session_params,
            runner=self.runner
        )

        frames_to_send = [
            LLMFullResponseStartFrame(),
            TTSTextFrame(text="Some text"),
            SleepFrame(sleep=0.01),  # Give time for text to be processed
            InterruptionFrame(),
        ]

        await run_test(aggregator, frames_to_send=frames_to_send)

        # Aggregation should be cleared
        self.assertEqual(aggregator._aggregation, "")

    async def test_multiple_text_frames_accumulation(self):
        """Test that multiple text frames accumulate correctly with proper spacing."""
        aggregator = AdkAssistantContextAggregator(
            session_service=self.session_service,
            session_params=self.session_params,
            runner=self.runner
        )

        frames_to_send = [
            LLMFullResponseStartFrame(),
            TTSTextFrame(text="Hello"),
            TTSTextFrame(text="world"),
            TTSTextFrame(text="this"),
            TTSTextFrame(text="is"),
            TTSTextFrame(text="a"),
            TTSTextFrame(text="test"),
        ]

        await run_test(aggregator, frames_to_send=frames_to_send)

        # Should be joined with spaces
        expected = "Hello world this is a test"
        self.assertEqual(aggregator._aggregation, expected)


if __name__ == "__main__":
    unittest.main()
