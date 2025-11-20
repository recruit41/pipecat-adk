"""
Test state synchronization frames for pipecat-adk.

This test validates the AdkStateDeltaFrame, AdkAppendEventFrame, and
AdkInvokeAgentFrame primitives for state synchronization between
Pipecat pipelines and ADK sessions.
"""

import asyncio
import unittest

from google.adk.agents import Agent
from google.adk.apps import App
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
from tests.mocks import (
    MockLLM,
    TestRunner,
    Say,
    WaitForResponse,
)


class TestStateSyncFrames(unittest.IsolatedAsyncioTestCase):
    """Test state synchronization frame primitives."""

    async def test_tool_state_delta_emits_frame(self):
        """Test that tools with state_delta emit AdkStateDeltaFrame."""
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
        )

        async with TestRunner(app=app) as runner:
            response = await runner.simulate_user([
                Say("Start the quiz"),
                WaitForResponse(),
            ], timeout=1.5)

            # Verify bot responded
            self.assertTrue(response.said("quiz"))

            # Verify AdkStateDeltaFrame was captured
            state_delta_frames = response.get_frames_of_type(AdkStateDeltaFrame)
            quiz_frames = [f for f in state_delta_frames if 'quiz_state' in f.state_delta]
            self.assertGreater(len(quiz_frames), 0, "Should emit AdkStateDeltaFrame with quiz_state")

            # Verify the state_delta content
            quiz_state = quiz_frames[-1].state_delta['quiz_state']
            self.assertEqual(quiz_state['quiz_id'], 'quiz-123')
            self.assertEqual(quiz_state['questions'], ['Q1', 'Q2'])

    async def test_append_event_frame_persists_and_emits_state_delta(self):
        """Test that AdkAppendEventFrame appends event to ADK session and emits state delta."""
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
        )

        async with TestRunner(app=app) as runner:
            # First, trigger a normal conversation to start the pipeline
            response = await runner.simulate_user([
                Say("Hello"),
                WaitForResponse(),
            ], timeout=1.0)

            self.assertTrue(response.said("Hello"))

            # Now push an AdkAppendEventFrame through the pipeline
            event = Event(
                author="user",
                actions=EventActions(state_delta={
                    'form_data': {'name': 'John', 'email': 'john@example.com'}
                })
            )
            append_frame = AdkAppendEventFrame(event=event)

            # Queue frame into pipeline properly
            await runner.queue_frame(append_frame)

            await asyncio.sleep(0.1)  # Give time for frame to propagate

            # Verify event was appended to session
            session_state = await runner.session_state()
            self.assertIn('form_data', session_state)
            self.assertEqual(session_state['form_data']['name'], 'John')

            # Verify AdkStateDeltaFrame was emitted - get fresh response snapshot
            response = await runner.simulate_user([], timeout=0.1)
            state_delta_frames = response.get_frames_of_type(AdkStateDeltaFrame)
            form_data_frames = [f for f in state_delta_frames if 'form_data' in f.state_delta]
            self.assertGreater(len(form_data_frames), 0, "Should emit AdkStateDeltaFrame for appended event")

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
        )

        async with TestRunner(app=app) as runner:
            # First turn - normal conversation
            response = await runner.simulate_user([
                Say("Hello"),
                WaitForResponse(),
            ], timeout=1.5)

            self.assertTrue(response.said("Hello"))

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

            # Give time for processing
            await asyncio.sleep(0.5)

            # Get fresh response snapshot with all frames
            response = await runner.simulate_user([], timeout=0.1)

            # Check state delta was emitted
            state_delta_frames = response.get_frames_of_type(AdkStateDeltaFrame)
            action_frames = [f for f in state_delta_frames if 'last_action' in f.state_delta]
            self.assertGreater(len(action_frames), 0, "Should emit state_delta from invoke frame")

            # Verify both responses were generated
            self.assertIn("summary", response.text.lower())

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
        )

        async with TestRunner(app=app) as runner:
            # First turn
            response = await runner.simulate_user([
                Say("Hello"),
                WaitForResponse(),
            ], timeout=1.5)

            self.assertTrue(response.said("First"))

            # Invoke agent without state_delta
            content = Content(
                role="user",
                parts=[Part(text="Continue")]
            )
            invoke_frame = AdkInvokeAgentFrame(new_content=content)

            await runner.queue_frame(invoke_frame)

            await asyncio.sleep(0.5)

            # Get fresh response snapshot
            response = await runner.simulate_user([], timeout=0.1)

            # Verify both responses were generated
            self.assertIn("First", response.text)
            self.assertIn("Second", response.text)

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
        )

        async with TestRunner(app=app) as runner:
            response = await runner.simulate_user([
                Say("Make the room comfortable"),
                WaitForResponse(),
            ], timeout=1.5)

            # Verify bot responded
            self.assertTrue(response.said("adjusted"))

            # Verify state_delta frames were emitted
            state_delta_frames = response.get_frames_of_type(AdkStateDeltaFrame)
            temp_frames = [f for f in state_delta_frames if 'temperature' in f.state_delta]
            brightness_frames = [f for f in state_delta_frames if 'brightness' in f.state_delta]
            self.assertGreater(len(temp_frames), 0, "Should emit state_delta with temperature")
            self.assertGreater(len(brightness_frames), 0, "Should emit state_delta with brightness")

            # Verify values
            self.assertEqual(temp_frames[-1].state_delta['temperature'], 72.0)
            self.assertEqual(brightness_frames[-1].state_delta['brightness'], 80)

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
        )

        async with TestRunner(app=app) as runner:
            # Start pipeline first
            await runner.simulate_user([
                Say("Hello"),
                WaitForResponse(),
            ], timeout=1.0)

            # Get initial frame count
            response = await runner.simulate_user([], timeout=0.1)
            initial_count = len(response.get_frames_of_type(AdkStateDeltaFrame))

            # Push an event without state_delta (just content)
            event = Event(
                author="user",
                content=Content(
                    role="user",
                    parts=[Part(text="A note")]
                )
            )
            append_frame = AdkAppendEventFrame(event=event)

            await runner.queue_frame(append_frame)

            await asyncio.sleep(0.2)

            # Verify no additional state_delta frames were emitted
            response = await runner.simulate_user([], timeout=0.1)
            final_count = len(response.get_frames_of_type(AdkStateDeltaFrame))
            self.assertEqual(initial_count, final_count)


if __name__ == "__main__":
    unittest.main()
