"""
Test pipecat-adk integration using mock services.

This test validates the full pipeline flow using MockLLM, MockSTTService,
MockTTSService, and MockInputTransport/MockOutputTransport.
"""

import unittest

from google.adk.agents import Agent
from google.adk.apps import App
from google.genai.types import Part
from pipecat_adk import InterruptionHandlerPlugin

from tests.mocks import MockLLM, TestRunner, Say, WaitTillBotSays, InterruptAfter, WaitForResponse


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
            # User says "Hi, I am John" and waits for bot to respond
            user_actions = [
                Say("Hi, I am John"),
                WaitTillBotSays("Hi, I am a bot"),
            ]

            # Run the interaction
            response = await runner.simulate_user(user_actions, timeout=2.0)

            # Verify bot said the expected text
            self.assertTrue(response.said("Hi, I am a bot"))

    async def test_interruption_handling(self):
        """Test that user can interrupt bot's response."""
        # Bot will give a long response that gets interrupted
        mock_llm = MockLLM.single(
            "Hello! I'm so glad you're interested in learning about our company. We have a very long history that spans over 50 years, and we've been pioneers in many different areas..."
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

        async with TestRunner(app=app) as runner:
            # User triggers response then interrupts mid-response
            user_actions = [
                Say("Tell me about your company"),
                InterruptAfter("Hello!", "Wait, I have a question"),
            ]

            # Give enough time for bot to start speaking and be interrupted
            response = await runner.simulate_user(user_actions, timeout=3.0)

            # Verify bot started speaking before being interrupted
            self.assertTrue(response.said("Hello"), f"Bot should have started speaking, got: {response.text!r}")

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
            user_actions = [
                Say("What's the weather in San Francisco?"),
                WaitForResponse(),
            ]

            response = await runner.simulate_user(user_actions, timeout=3.0)

            # Verify bot responded with weather info
            self.assertTrue(response.said("sunny and 72 degrees"))

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
            user_actions = [
                Say("Make the room comfortable"),
                WaitForResponse(),
            ]

            response = await runner.simulate_user(user_actions, timeout=3.0)

            # Verify bot confirmed both actions
            self.assertTrue(response.said("72 degrees"))
            self.assertTrue(response.said("80% brightness"))

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
            response1 = await runner.simulate_user([
                Say("Hello"),
                WaitForResponse(),
            ], timeout=2.0)

            self.assertTrue(response1.said("How can I help"))

            # Second turn
            response2 = await runner.simulate_user([
                Say("I need help with a project"),
                WaitForResponse(),
            ], timeout=2.0)

            self.assertTrue(response2.said("Tell me more"))

            # Third turn
            response3 = await runner.simulate_user([
                Say("Can you assist me?"),
                WaitForResponse(),
            ], timeout=2.0)

            self.assertTrue(response3.said("happy to assist"))


if __name__ == "__main__":
    unittest.main()
