"""
Unit tests for AdkAssistantContextAggregator edge cases.

These tests exercise the aggregator in isolation to verify edge case behavior
that's difficult to trigger through full integration tests.
"""

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


class TestAssistantAggregatorEdgeCases(unittest.IsolatedAsyncioTestCase):
    """Edge case tests for AdkAssistantContextAggregator."""

    async def asyncSetUp(self):
        """Set up test fixtures."""
        self.session_service = InMemorySessionService()
        self.session_params = SessionParams(
            app_name="test_app",
            session_id="test_session",
            user_id="test_user",
        )

        await self.session_service.create_session(**self.session_params.model_dump())

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

    async def test_interruption_creates_synthetic_event(self):
        """Test that interruption with accumulated text creates a synthetic event."""
        aggregator = AdkAssistantContextAggregator(
            session_service=self.session_service,
            session_params=self.session_params,
            runner=self.runner
        )

        frames_to_send = [
            LLMFullResponseStartFrame(),
            TTSTextFrame(text="The capital of India is New Delhi.", aggregated_by="sentence"),
            SleepFrame(sleep=0.01),
            InterruptionFrame(),
        ]

        await run_test(aggregator, frames_to_send=frames_to_send)

        session = await self.session_service.get_session(
            app_name=self.session_params.app_name,
            session_id=self.session_params.session_id,
            user_id=self.session_params.user_id
        )
        assert session

        # Should have one event (the interruption event)
        self.assertEqual(len(session.events), 1)

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
            TTSTextFrame(text="The capital of India is", aggregated_by="sentence"),
            SleepFrame(sleep=0.01),
            InterruptionFrame(),
        ]

        await run_test(aggregator, frames_to_send=frames_to_send)

        session = await self.session_service.get_session(
            app_name=self.session_params.app_name,
            session_id=self.session_params.session_id,
            user_id=self.session_params.user_id
        )
        assert session

        simplified = simplify_events(session.events)
        self.assertEqual(len(simplified), 1)
        author, content = simplified[0]
        self.assertEqual(author, "test_agent")
        self.assertIn("The capital of India is", content)
        self.assertNotIn("New Delhi", content)


if __name__ == "__main__":
    unittest.main()
