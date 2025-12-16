"""
Integration tests for pipecat-adk using TestRunner.

These tests validate end-to-end pipeline flows using the conversational DSL.
All tests use TestRunner with MockLLM to simulate real conversations.
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
    InterruptionHandlerPlugin,
)
from tests.mocks import MockLLM, TestRunner, Turn


class TestBasicConversation(unittest.IsolatedAsyncioTestCase):
    """Test basic conversation flows."""

    async def test_basic_interaction(self):
        """Test basic user-bot interaction."""
        mock_llm = MockLLM.single("Hi, I am a bot")

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
            await runner.speak_and_wait_for_response("Hi, I am John", timeout=5.0)

            self.assertEqual(runner.transcript, [
                Turn("user", "Hi, I am John"),
                Turn("bot", "Hi, I am a bot"),
            ])

    async def test_multi_turn_conversation(self):
        """Test multi-turn conversation flow."""
        mock_llm = MockLLM.conversation([
            "Hi! How can I help you today?",
            "That sounds interesting! Tell me more.",
            "Great, I'd be happy to assist with that.",
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
            await runner.speak_and_wait_for_response("I need help with a project", timeout=5.0)
            await runner.speak_and_wait_for_response("Can you assist me?", timeout=5.0)

            self.assertEqual(runner.transcript, [
                Turn("user", "Hello"),
                Turn("bot", "Hi! How can I help you today?"),
                Turn("user", "I need help with a project"),
                Turn("bot", "That sounds interesting! Tell me more."),
                Turn("user", "Can you assist me?"),
                Turn("bot", "Great, I'd be happy to assist with that."),
            ])


class TestInterruptions(unittest.IsolatedAsyncioTestCase):
    """Test interruption handling."""

    async def test_interruption_handling(self):
        """Test that user can interrupt bot's response."""
        mock_llm = MockLLM.single(
            "Hello! I'm so glad you're interested in learning about our company. "
            "We have a very long history that spans over 50 years, and we've been "
            "pioneers in many different areas..."
        )

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

        async with TestRunner(app=app, tts_delay=0.05) as runner:
            await runner.join()
            await runner.speak("Tell me about your company")
            await runner.interrupt_bot("Wait, I have a question", timeout=5.0)

            # Wait for interruption to be processed
            def _has_interruption_transcription(bot_output, delta_messages):
                return any(
                    msg.get("type") == "user-transcription" and
                    msg.get("data", {}).get("final") and
                    "question" in msg.get("data", {}).get("text", "").lower()
                    for msg in delta_messages
                )
            await runner.wait_for(_has_interruption_transcription, timeout=5.0)

            # Verify the session recorded the conversation
            events = await runner.events()
            self.assertGreater(len(events), 0, "Should have events in session")

            has_user_message = any(
                hasattr(e, 'content') and e.content and e.content.role == 'user'
                for e in events
            )
            self.assertTrue(has_user_message, "Should have at least one user message")


class TestFunctionCalls(unittest.IsolatedAsyncioTestCase):
    """Test function call handling."""

    async def test_function_call_handling(self):
        """Test that function calls are properly handled."""
        def get_weather(location: str) -> dict:
            """Get the current weather for a location."""
            return {"weather": "sunny", "temperature": "72 degrees"}

        mock_llm = MockLLM.from_parts([
            [Part.from_function_call(
                name="get_weather",
                args={"location": "San Francisco"}
            )],
            "The weather in San Francisco is sunny and 72 degrees!",
        ])

        agent = Agent(
            name="test_agent",
            model=mock_llm,
            instruction="You are a helpful weather assistant.",
            tools=[get_weather],
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
                "What's the weather in San Francisco?",
                timeout=5.0,
            )

            self.assertEqual(
                runner.last_bot_message,
                "The weather in San Francisco is sunny and 72 degrees!"
            )

            # Verify function call in session history
            events = await runner.events()
            has_function_call = any(
                e.content.parts and hasattr(e.content.parts[0], 'function_call')
                for e in events
                if hasattr(e, 'content') and e.content and e.content.parts
            )
            self.assertTrue(has_function_call, "Function call should be in session")

    async def test_multiple_function_calls_in_turn(self):
        """Test handling multiple function calls in a single turn."""
        def set_temperature(degrees: float) -> dict:
            """Set the room temperature."""
            return {"status": "success", "temperature": degrees}

        def set_lights(brightness: int) -> dict:
            """Set the room lights."""
            return {"status": "success", "brightness": brightness}

        mock_llm = MockLLM.from_parts([
            [
                Part.from_function_call(
                    name="set_temperature",
                    args={"degrees": 72}
                ),
                Part.from_function_call(
                    name="set_lights",
                    args={"brightness": 80}
                ),
            ],
            "I've set the temperature to 72 degrees and the lights to 80% brightness.",
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

            self.assertEqual(
                runner.last_bot_message,
                "I've set the temperature to 72 degrees and the lights to 80% brightness."
            )

            # Verify both function calls in session
            events = await runner.events()
            function_calls = []
            for e in events:
                if hasattr(e, 'content') and e.content and e.content.parts:
                    for part in e.content.parts:
                        if hasattr(part, 'function_call') and part.function_call:
                            function_calls.append(part.function_call.name)

            self.assertIn("set_temperature", function_calls)
            self.assertIn("set_lights", function_calls)


class TestStateSynchronization(unittest.IsolatedAsyncioTestCase):
    """Test state synchronization between Pipecat and ADK."""

    async def test_tool_state_delta_updates_session(self):
        """Test that tools with state_delta update ADK session state."""
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

            self.assertEqual(runner.last_bot_message, "I've set up the quiz for you!")

            session_state = await runner.session_state()
            self.assertIn('quiz_state', session_state)
            self.assertEqual(session_state['quiz_state']['quiz_id'], 'quiz-123')

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
            await runner.join()
            await runner.speak_and_wait_for_response("Hello", timeout=5.0)
            self.assertEqual(runner.last_bot_message, "Hello!")

            # Push AdkAppendEventFrame
            event = Event(
                author="user",
                actions=EventActions(state_delta={
                    'form_data': {'name': 'John', 'email': 'john@example.com'}
                })
            )
            await runner.queue_frame(AdkAppendEventFrame(event=event))
            await runner.stay_silent()

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
            await runner.join()
            await runner.speak_and_wait_for_response("Hello", timeout=5.0)
            self.assertEqual(runner.last_bot_message, "Hello! How can I help?")

            # Invoke agent via frame
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
            await runner.join()
            await runner.speak_and_wait_for_response("Hello", timeout=5.0)
            self.assertEqual(runner.last_bot_message, "First response.")

            content = Content(
                role="user",
                parts=[Part(text="Continue")]
            )
            await runner.queue_frame(AdkInvokeAgentFrame(new_content=content))
            await runner.wait_for_response(timeout=5.0)

            self.assertEqual(runner.last_bot_message, "Second response.")

    async def test_multiple_tools_with_state_delta(self):
        """Test multiple tools can emit state_delta in same turn."""
        def set_temperature(degrees: float, tool_context: ToolContext) -> dict:
            """Set room temperature."""
            tool_context.actions.state_delta = {'temperature': degrees}
            return {'status': 'temperature_set'}

        def set_lights(brightness: int, tool_context: ToolContext) -> dict:
            """Set room lights."""
            tool_context.actions.state_delta = {'brightness': brightness}
            return {'status': 'lights_set'}

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

            self.assertEqual(
                runner.last_bot_message,
                "I've adjusted the temperature and lights."
            )

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
            await runner.join()
            await runner.speak_and_wait_for_response("Hello", timeout=5.0)
            self.assertEqual(runner.last_bot_message, "Hello!")

            initial_state = await runner.session_state()
            initial_public_state = {
                k: v for k, v in initial_state.items() if not k.startswith('_')
            }

            # Push event without state_delta
            event = Event(
                author="user",
                content=Content(
                    role="user",
                    parts=[Part(text="A note")]
                )
            )
            await runner.queue_frame(AdkAppendEventFrame(event=event))
            await runner.stay_silent()

            final_state = await runner.session_state()
            final_public_state = {
                k: v for k, v in final_state.items() if not k.startswith('_')
            }
            self.assertEqual(initial_public_state, final_public_state)


if __name__ == "__main__":
    unittest.main()
