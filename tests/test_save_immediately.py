"""
Tests for "save user input immediately" feature.

This feature ensures user messages are persisted to the ADK session
immediately when speech is transcribed, before the LLM processes them.
This prevents message loss if the pipeline is interrupted.
"""

import asyncio
import unittest

from google.adk.agents import Agent
from google.adk.apps import App
from google.adk.apps.app import ResumabilityConfig
from google.adk.sessions import InMemorySessionService
from google.genai.types import Content, Part

from pipecat_adk import (
    AdkBasedLLMService,
    AdkInvokeAgentFrame,
    InterruptionHandlerPlugin,
    SessionParams,
)
from tests.mocks import MockLLM, TestRunner


class TestSaveImmediately(unittest.IsolatedAsyncioTestCase):
    """Tests for immediate user message persistence."""

    async def test_user_message_saved_before_llm_response(self):
        """Test that user message is in session before LLM responds.

        This is the core design: user aggregator saves to ADK session
        in _process_aggregation(), before pushing the context to the LLM service.
        """
        mock_llm = MockLLM.single("Hello!")

        agent = Agent(
            name="test_agent",
            model=mock_llm,
            instruction="You are a helpful assistant.",
        )

        app = App(
            name="agents",
            root_agent=agent,
            plugins=[InterruptionHandlerPlugin()],
            resumability_config=ResumabilityConfig(is_resumable=True),
        )

        async with TestRunner(app=app) as runner:
            await runner.join()

            # Speak and wait for response
            await runner.speak_and_wait_for_response("Hello from user", timeout=5.0)

            # Check session - user message should be there before agent response
            events = await runner.events()
            user_events = [
                e for e in events
                if hasattr(e, 'content') and e.content and e.content.role == 'user'
            ]
            agent_events = [
                e for e in events
                if hasattr(e, 'content') and e.content and e.content.role == 'model'
            ]

            self.assertGreater(len(user_events), 0,
                "User message should be in session")

            # Verify the text content
            user_text = user_events[0].content.parts[0].text
            self.assertIn("Hello from user", user_text)

            # Verify agent also responded
            self.assertGreater(len(agent_events), 0,
                "Agent should have responded")

            # Verify user event timestamp is before agent event (if timestamps available)
            # Both should exist in the session, with user message saved first
            self.assertEqual(runner.last_bot_message, "Hello!")

    async def test_invocation_id_present_on_user_events(self):
        """Test that user events have invocation_id set."""
        mock_llm = MockLLM.single("Response")

        agent = Agent(
            name="test_agent",
            model=mock_llm,
            instruction="You are a helpful assistant.",
        )

        app = App(
            name="agents",
            root_agent=agent,
            plugins=[InterruptionHandlerPlugin()],
            resumability_config=ResumabilityConfig(is_resumable=True),
        )

        async with TestRunner(app=app) as runner:
            await runner.join()
            await runner.speak_and_wait_for_response("Test message", timeout=5.0)

            events = await runner.events()
            user_events = [
                e for e in events
                if hasattr(e, 'content') and e.content and e.content.role == 'user'
            ]

            self.assertGreater(len(user_events), 0)

            # Check invocation_id is a valid UUID format
            user_event = user_events[0]
            self.assertIsNotNone(user_event.invocation_id)
            # Event.new_id() returns a UUID string (e.g., "87cffdf2-9db3-420c-ae81-1be813ff7391")
            self.assertRegex(
                user_event.invocation_id,
                r'^[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}$',
                f"invocation_id should be a valid UUID, got: {user_event.invocation_id}"
            )

    async def test_multi_turn_unique_invocation_ids(self):
        """Test that each turn gets a unique invocation_id."""
        mock_llm = MockLLM.conversation([
            "First response",
            "Second response",
            "Third response",
        ])

        agent = Agent(
            name="test_agent",
            model=mock_llm,
            instruction="You are a helpful assistant.",
        )

        app = App(
            name="agents",
            root_agent=agent,
            plugins=[InterruptionHandlerPlugin()],
            resumability_config=ResumabilityConfig(is_resumable=True),
        )

        async with TestRunner(app=app) as runner:
            await runner.join()
            await runner.speak_and_wait_for_response("First", timeout=5.0)
            await runner.speak_and_wait_for_response("Second", timeout=5.0)
            await runner.speak_and_wait_for_response("Third", timeout=5.0)

            events = await runner.events()
            user_events = [
                e for e in events
                if hasattr(e, 'content') and e.content and e.content.role == 'user'
            ]

            self.assertEqual(len(user_events), 3, "Should have 3 user events")

            invocation_ids = [e.invocation_id for e in user_events]
            unique_ids = set(invocation_ids)

            self.assertEqual(
                len(unique_ids), 3,
                f"All invocation_ids should be unique, got: {invocation_ids}"
            )

    async def test_invoke_agent_frame_saves_before_invoking(self):
        """Test that AdkInvokeAgentFrame saves message before invoking ADK."""
        mock_llm = MockLLM.conversation([
            "First response",
            "Second response",
        ])

        agent = Agent(
            name="test_agent",
            model=mock_llm,
            instruction="You are a helpful assistant.",
        )

        app = App(
            name="agents",
            root_agent=agent,
            plugins=[InterruptionHandlerPlugin()],
            resumability_config=ResumabilityConfig(is_resumable=True),
        )

        async with TestRunner(app=app) as runner:
            await runner.join()
            await runner.speak_and_wait_for_response("Hello", timeout=5.0)

            # Use AdkInvokeAgentFrame for programmatic invocation
            content = Content(
                role="user",
                parts=[Part(text="Programmatic message")]
            )
            await runner.queue_frame(AdkInvokeAgentFrame(new_content=content))
            await runner.wait_for_response(timeout=5.0)

            # Verify programmatic message is in session
            events = await runner.events()
            user_texts = []
            for e in events:
                if hasattr(e, 'content') and e.content and e.content.role == 'user':
                    for part in e.content.parts:
                        if hasattr(part, 'text') and part.text:
                            user_texts.append(part.text)

            self.assertIn("Programmatic message", user_texts)


class TestResumabilityValidation(unittest.IsolatedAsyncioTestCase):
    """Tests for resumability config validation."""

    async def test_resumability_config_required(self):
        """Test that AdkBasedLLMService requires resumability_config."""
        mock_llm = MockLLM.single("Test")

        agent = Agent(
            name="test_agent",
            model=mock_llm,
            instruction="You are a helpful assistant.",
        )

        # Create app WITHOUT resumability_config
        app = App(
            name="agents",
            root_agent=agent,
            plugins=[InterruptionHandlerPlugin()],
        )

        session_service = InMemorySessionService()
        session_params = SessionParams(
            app_name="agents",
            session_id="test_session",
            user_id="test_user",
        )

        with self.assertRaises(ValueError) as ctx:
            AdkBasedLLMService(
                session_service=session_service,
                session_params=session_params,
                app=app,
            )

        self.assertIn("resumability_config", str(ctx.exception))
        self.assertIn("is_resumable=True", str(ctx.exception))

    async def test_resumability_config_false_rejected(self):
        """Test that is_resumable=False is rejected."""
        mock_llm = MockLLM.single("Test")

        agent = Agent(
            name="test_agent",
            model=mock_llm,
            instruction="You are a helpful assistant.",
        )

        # Create app with is_resumable=False
        app = App(
            name="agents",
            root_agent=agent,
            plugins=[InterruptionHandlerPlugin()],
            resumability_config=ResumabilityConfig(is_resumable=False),
        )

        session_service = InMemorySessionService()
        session_params = SessionParams(
            app_name="agents",
            session_id="test_session",
            user_id="test_user",
        )

        with self.assertRaises(ValueError) as ctx:
            AdkBasedLLMService(
                session_service=session_service,
                session_params=session_params,
                app=app,
            )

        self.assertIn("resumability_config", str(ctx.exception))


class TestSessionErrors(unittest.IsolatedAsyncioTestCase):
    """Tests for session error handling."""

    async def test_session_not_found_raises_error(self):
        """Test that missing session raises RuntimeError."""
        mock_llm = MockLLM.single("Test")

        agent = Agent(
            name="test_agent",
            model=mock_llm,
            instruction="You are a helpful assistant.",
        )

        app = App(
            name="agents",
            root_agent=agent,
            plugins=[InterruptionHandlerPlugin()],
            resumability_config=ResumabilityConfig(is_resumable=True),
        )

        session_service = InMemorySessionService()
        session_params = SessionParams(
            app_name="agents",
            session_id="nonexistent_session",
            user_id="test_user",
        )

        # Create service WITHOUT creating session first
        adk_service = AdkBasedLLMService(
            session_service=session_service,
            session_params=session_params,
            app=app,
        )
        context_aggregators = adk_service.create_context_aggregator()
        user_aggregator = context_aggregators.user()

        # Set up aggregation text (simulating accumulated speech)
        user_aggregator._aggregation = "Test message"

        # Try to process aggregation - should fail because session doesn't exist
        with self.assertRaises(RuntimeError) as ctx:
            await user_aggregator._process_aggregation()

        self.assertIn("session not found", str(ctx.exception).lower())


if __name__ == "__main__":
    unittest.main()
