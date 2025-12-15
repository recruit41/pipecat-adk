"""
Test pipecat-adk integration using mock services.

This test validates the full pipeline flow using MockLLM, MockSTTService,
MockTTSService, and MockInputTransport/MockOutputTransport with RTVI.
"""

import unittest

from google.adk.agents import Agent
from google.adk.apps import App
from google.genai.types import Part
from pipecat_adk import InterruptionHandlerPlugin

from tests.mocks import MockLLM, TestRunner, Turn


class TestWithMocks(unittest.IsolatedAsyncioTestCase):
    """Test full pipeline with mock services."""

    async def test_basic_interaction(self):
        """Test basic user-bot interaction with MockLLM."""
        # Create MockLLM that says "Hi, I am a bot"
        mock_llm = MockLLM.single("Hi, I am a bot")

        # Create ADK agent with MockLLM
        agent = Agent(
            name="test_agent",
            model=mock_llm,
            instruction="You are a helpful assistant.",
        )

        # Create App with InterruptionHandlerPlugin
        # Note: app name must be "agents" to match TestRunner's hardcoded session_params
        app = App(
            name="agents",
            root_agent=agent,
            plugins=[InterruptionHandlerPlugin()],
        )

        # Create test runner with the app
        async with TestRunner(app=app) as runner:
            # User joins and speaks
            await runner.join()
            await runner.speak_and_wait_for_response("Hi, I am John", timeout=5.0)

            # Verify bot said the expected text
            self.assertIn("Hi, I am a bot", runner.last_bot_message)

    async def test_interruption_handling(self):
        """Test that user can interrupt bot's response."""
        # Bot will give a long response that gets interrupted
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
        )

        async with TestRunner(app=app, tts_delay=0.05) as runner:
            await runner.join()
            # User speaks to trigger bot response
            await runner.speak("Tell me about your company")
            # Interrupt the bot while it's speaking
            await runner.interrupt_bot("Wait, I have a question", timeout=5.0)

            # Wait for interruption to be processed by checking for user transcription
            def _has_interruption_transcription(bot_output, delta_messages):
                return any(
                    msg.get("type") == "user-transcription" and
                    msg.get("data", {}).get("final") and
                    "question" in msg.get("data", {}).get("text", "").lower()
                    for msg in delta_messages
                )
            await runner.wait_for(_has_interruption_transcription, timeout=5.0)

            # Verify the session has recorded some conversation
            events = await runner.events()
            self.assertGreater(len(events), 0, "Should have events in session")

            # Verify we have at least a user message in the session
            has_user_message = any(
                hasattr(e, 'content') and e.content and e.content.role == 'user'
                for e in events
            )
            self.assertTrue(has_user_message, "Should have at least one user message in session")

    async def test_function_call_handling(self):
        """Test that function calls are properly handled in the pipeline."""
        # Define a mock tool as a Python function
        def get_weather(location: str) -> dict:
            """Get the current weather for a location."""
            return {"weather": "sunny", "temperature": "72 degrees"}

        # Bot will make a function call, then respond with the result
        mock_llm = MockLLM.from_parts([
            # First turn: function call
            [Part.from_function_call(
                name="get_weather",
                args={"location": "San Francisco"}
            )],
            # Second turn: respond after function execution
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
        )

        async with TestRunner(app=app) as runner:
            await runner.join()
            await runner.speak_and_wait_for_response(
                "What's the weather in San Francisco?",
                timeout=5.0,
            )

            # Verify bot responded with weather info
            self.assertIn("sunny and 72 degrees", runner.last_bot_message)

            # Verify session contains function call and response events
            events = await runner.events()

            # Should have a function_call event
            has_function_call = any(
                e.content.parts and hasattr(e.content.parts[0], 'function_call')
                for e in events
                if hasattr(e, 'content') and e.content and e.content.parts
            )
            self.assertTrue(has_function_call, "Function call should be in session history")

    async def test_multiple_function_calls_in_turn(self):
        """Test handling multiple function calls in a single turn."""
        # Define mock tools as Python functions
        def set_temperature(degrees: float) -> dict:
            """Set the room temperature."""
            return {"status": "success", "temperature": degrees}

        def set_lights(brightness: int) -> dict:
            """Set the room lights."""
            return {"status": "success", "brightness": brightness}

        # Bot makes multiple function calls in one turn
        mock_llm = MockLLM.from_parts([
            # First turn: two function calls
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
            # Second turn: confirm actions
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
        )

        async with TestRunner(app=app) as runner:
            await runner.join()
            await runner.speak_and_wait_for_response(
                "Make the room comfortable",
                timeout=5.0,
            )

            # Verify bot confirmed both actions
            self.assertIn("72 degrees", runner.last_bot_message)
            self.assertIn("80% brightness", runner.last_bot_message)

            # Verify session has both function calls
            events = await runner.events()
            function_calls = []
            for e in events:
                if (hasattr(e, 'content') and e.content and e.content.parts):
                    for part in e.content.parts:
                        if hasattr(part, 'function_call') and part.function_call:
                            function_calls.append(part.function_call.name)

            self.assertIn("set_temperature", function_calls)
            self.assertIn("set_lights", function_calls)

    async def test_multi_turn_conversation(self):
        """Test multi-turn conversation flow."""
        # Bot will respond across multiple turns
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
        )

        async with TestRunner(app=app) as runner:
            # First turn
            await runner.join()
            await runner.speak_and_wait_for_response("Hello", timeout=5.0)
            self.assertIn("How can I help", runner.last_bot_message)

            # Second turn
            await runner.speak_and_wait_for_response("I need help with a project", timeout=5.0)
            self.assertIn("Tell me more", runner.last_bot_message)

            # Third turn
            await runner.speak_and_wait_for_response("Can you assist me?", timeout=5.0)
            self.assertIn("happy to assist", runner.last_bot_message)


if __name__ == "__main__":
    unittest.main()
