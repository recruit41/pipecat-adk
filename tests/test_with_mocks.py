"""
Test pipecat-adk integration using mock services.

This test validates the full pipeline flow using MockLLM, MockSTTService,
MockTTSService, and MockInputTransport/MockOutputTransport.
"""

import unittest

from google.adk.agents import Agent
from google.adk.apps import App
from pipecat_adk import InterruptionHandlerPlugin

from tests.mocks import MockLLM, TestRunner, Say, WaitTillBotSays


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
            response = await runner.run(user_actions, timeout=2.0)

            # Verify bot said the expected text
            self.assertTrue(response.said("Hi, I am a bot"))


if __name__ == "__main__":
    unittest.main()
