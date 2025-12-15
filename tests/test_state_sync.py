"""
Test state synchronization frames for pipecat-adk.

This test validates the AdkStateDeltaFrame, AdkAppendEventFrame, and
AdkInvokeAgentFrame primitives for state synchronization between
Pipecat pipelines and ADK sessions.
"""

import unittest

from google.adk.agents import Agent
from google.adk.apps import App
from google.adk.apps.app import ResumabilityConfig
from google.adk.events import Event
from google.adk.events.event_actions import EventActions
from google.adk.tools import ToolContext
from google.genai.types import Content, Part

from pipecat_adk import (
    AdkAppendEventFrame,
    AdkInvokeAgentFrame,
    AdkStateDeltaFrame,
    InterruptionHandlerPlugin,
)
from tests.mocks import MockLLM, TestRunner, Turn


class TestStateSyncFrames(unittest.IsolatedAsyncioTestCase):
    """Test state synchronization frame primitives."""

    async def test_tool_state_delta_updates_session(self):
        """Test that tools with state_delta update ADK session state."""
        # Define a tool that sets state_delta
        def set_quiz_state(quiz_id: str, tool_context: ToolContext) -> dict:
            """Set up a quiz state."""
            tool_context.actions.state_delta = {
                'quiz_state': {
                    'quiz_id': quiz_id,
                    'questions': ['Q1', 'Q2'],
                    'current_index': 0
                }
            }
            return {'status': 'quiz_started'}

        # Bot will call the tool, then respond
        mock_llm = MockLLM.from_parts([
            [Part.from_function_call(
                name="set_quiz_state",
                args={"quiz_id": "quiz-123"}
            )],
            "I've set up the quiz for you!",
        ])

        agent = Agent(
            name="test_agent",
            model=mock_llm,
            instruction="You are a quiz assistant.",
            tools=[set_quiz_state],
        )

        app = App(
            name="agents",
            root_agent=agent,
            plugins=[InterruptionHandlerPlugin()],
            resumability_config=ResumabilityConfig(is_resumable=True),
        )

        async with TestRunner(app=app) as runner:
            await runner.join()
            await runner.speak_and_wait_for_response("Start the quiz", timeout=5.0)

            # Verify bot responded with exact expected text
            self.assertEqual(runner.last_bot_message, "I've set up the quiz for you!")

            # Verify session state was updated
            session_state = await runner.session_state()
            self.assertIn('quiz_state', session_state)
            self.assertEqual(session_state['quiz_state']['quiz_id'], 'quiz-123')
            self.assertEqual(session_state['quiz_state']['questions'], ['Q1', 'Q2'])

    async def test_append_event_frame_persists_state(self):
        """Test that AdkAppendEventFrame appends event to ADK session."""
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
            # First, trigger a normal conversation to start the pipeline
            await runner.join()
            await runner.speak_and_wait_for_response("Hello", timeout=5.0)
            self.assertEqual(runner.last_bot_message, "Hello!")

            # Now push an AdkAppendEventFrame through the pipeline
            event = Event(
                author="user",
                actions=EventActions(state_delta={
                    'form_data': {'name': 'John', 'email': 'john@example.com'}
                })
            )
            append_frame = AdkAppendEventFrame(event=event)

            # Queue frame into pipeline (no LLM response expected)
            await runner.queue_frame(append_frame)
            await runner.stay_silent()

            # Verify event was appended to session
            session_state = await runner.session_state()
            self.assertIn('form_data', session_state)
            self.assertEqual(session_state['form_data']['name'], 'John')

    async def test_invoke_agent_frame_triggers_llm(self):
        """Test that AdkInvokeAgentFrame invokes the agent."""
        mock_llm = MockLLM.conversation([
            "Hello! How can I help?",
            "Here's the summary you requested.",
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
            # First turn - normal conversation
            await runner.join()
            await runner.speak_and_wait_for_response("Hello", timeout=5.0)
            self.assertEqual(runner.last_bot_message, "Hello! How can I help?")

            # Now push AdkInvokeAgentFrame to trigger second response
            content = Content(
                role="user",
                parts=[Part(text="Please summarize.")]
            )
            invoke_frame = AdkInvokeAgentFrame(
                new_content=content,
                state_delta={'last_action': 'requested_summary'}
            )

            await runner.queue_frame(invoke_frame)
            await runner.wait_for_response(timeout=5.0)

            # Verify second response and session state
            self.assertEqual(runner.last_bot_message, "Here's the summary you requested.")
            session_state = await runner.session_state()
            self.assertEqual(session_state.get('last_action'), 'requested_summary')

    async def test_invoke_agent_frame_without_state_delta(self):
        """Test AdkInvokeAgentFrame works without state_delta."""
        mock_llm = MockLLM.conversation([
            "First response.",
            "Second response.",
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
            # First turn
            await runner.join()
            await runner.speak_and_wait_for_response("Hello", timeout=5.0)
            self.assertEqual(runner.last_bot_message, "First response.")

            # Invoke agent without state_delta
            content = Content(
                role="user",
                parts=[Part(text="Continue")]
            )
            invoke_frame = AdkInvokeAgentFrame(new_content=content)

            await runner.queue_frame(invoke_frame)
            await runner.wait_for_response(timeout=5.0)

            # Verify second response
            self.assertEqual(runner.last_bot_message, "Second response.")

    async def test_multiple_tools_with_state_delta(self):
        """Test multiple tools can emit state_delta in same turn."""
        # Define tools that set different state_delta
        def set_temperature(degrees: float, tool_context: ToolContext) -> dict:
            """Set room temperature."""
            tool_context.actions.state_delta = {
                'temperature': degrees
            }
            return {'status': 'temperature_set'}

        def set_lights(brightness: int, tool_context: ToolContext) -> dict:
            """Set room lights."""
            tool_context.actions.state_delta = {
                'brightness': brightness
            }
            return {'status': 'lights_set'}

        # Bot calls both tools
        mock_llm = MockLLM.from_parts([
            [
                Part.from_function_call(
                    name="set_temperature",
                    args={"degrees": 72.0}
                ),
                Part.from_function_call(
                    name="set_lights",
                    args={"brightness": 80}
                ),
            ],
            "I've adjusted the temperature and lights.",
        ])

        agent = Agent(
            name="test_agent",
            model=mock_llm,
            instruction="You are a smart home assistant.",
            tools=[set_temperature, set_lights],
        )

        app = App(
            name="agents",
            root_agent=agent,
            plugins=[InterruptionHandlerPlugin()],
            resumability_config=ResumabilityConfig(is_resumable=True),
        )

        async with TestRunner(app=app) as runner:
            await runner.join()
            await runner.speak_and_wait_for_response(
                "Make the room comfortable",
                timeout=5.0,
            )

            # Verify bot responded with exact expected text
            self.assertEqual(
                runner.last_bot_message,
                "I've adjusted the temperature and lights."
            )

            # Verify session state was updated
            session_state = await runner.session_state()
            self.assertEqual(session_state.get('temperature'), 72.0)
            self.assertEqual(session_state.get('brightness'), 80)

    async def test_append_event_without_state_delta(self):
        """Test AdkAppendEventFrame with event that has no state_delta."""
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
            # Start pipeline first
            await runner.join()
            await runner.speak_and_wait_for_response("Hello", timeout=5.0)
            self.assertEqual(runner.last_bot_message, "Hello!")

            # Get initial session state (excluding internal keys)
            initial_state = await runner.session_state()
            initial_public_state = {
                k: v for k, v in initial_state.items() if not k.startswith('_')
            }

            # Push an event without state_delta (just content)
            event = Event(
                author="user",
                content=Content(
                    role="user",
                    parts=[Part(text="A note")]
                )
            )
            append_frame = AdkAppendEventFrame(event=event)

            # Queue frame (no LLM response expected)
            await runner.queue_frame(append_frame)
            await runner.stay_silent()

            # Verify session state didn't change (no state_delta)
            final_state = await runner.session_state()
            final_public_state = {
                k: v for k, v in final_state.items() if not k.startswith('_')
            }
            self.assertEqual(initial_public_state, final_public_state)


if __name__ == "__main__":
    unittest.main()
