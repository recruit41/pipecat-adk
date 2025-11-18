"""Tests for InterruptionHandlerPlugin.

This test suite directly invokes before_model_callback with LlmRequest objects
to exercise various code paths in the interruption handling logic.
"""

import unittest
from unittest.mock import MagicMock

from google.adk.agents.callback_context import CallbackContext
from google.adk.models.llm_request import LlmRequest
from google.genai.types import Content, Part

from pipecat_adk.plugin import InterruptionHandlerPlugin


class TestInterruptionHandlerPlugin(unittest.IsolatedAsyncioTestCase):
    """Test suite for InterruptionHandlerPlugin."""

    def setUp(self):
        """Set up test fixtures."""
        self.plugin = InterruptionHandlerPlugin()
        self.callback_context = MagicMock(spec=CallbackContext)

    async def test_plugin_initialization(self):
        """Test that plugin initializes with correct name."""
        plugin = InterruptionHandlerPlugin()
        self.assertEqual(plugin.name, "interruption_handler")

    async def test_before_model_callback_with_none_request(self):
        """Test handling of None LlmRequest."""
        result = await self.plugin.before_model_callback(
            callback_context=self.callback_context,
            llm_request=None
        )
        self.assertIsNone(result)

    async def test_before_model_callback_without_contents_attribute(self):
        """Test handling of LlmRequest without contents attribute."""
        llm_request = MagicMock(spec=LlmRequest)
        # Remove contents attribute
        del llm_request.contents

        result = await self.plugin.before_model_callback(
            callback_context=self.callback_context,
            llm_request=llm_request
        )
        self.assertIsNone(result)

    async def test_before_model_callback_with_empty_contents(self):
        """Test handling of LlmRequest with empty contents list."""
        llm_request = MagicMock(spec=LlmRequest)
        llm_request.contents = []

        result = await self.plugin.before_model_callback(
            callback_context=self.callback_context,
            llm_request=llm_request
        )
        self.assertIsNone(result)

    async def test_before_model_callback_without_interruption(self):
        """Test that content without interruption markers passes through unchanged."""
        llm_request = MagicMock(spec=LlmRequest)
        llm_request.contents = [
            Content(role="user", parts=[Part(text="Hello")]),
            Content(role="model", parts=[Part(text="Hi there!")]),
        ]

        result = await self.plugin.before_model_callback(
            callback_context=self.callback_context,
            llm_request=llm_request
        )

        self.assertIsNone(result)
        self.assertEqual(len(llm_request.contents), 2)
        self.assertEqual(llm_request.contents[0].parts[0].text, "Hello")
        self.assertEqual(llm_request.contents[1].parts[0].text, "Hi there!")

    async def test_interruption_replaces_model_response(self):
        """Test that interruption marker replaces previous model response."""
        llm_request = MagicMock(spec=LlmRequest)
        llm_request.contents = [
            Content(role="user", parts=[Part(text="What's your name?")]),
            Content(role="model", parts=[Part(text="Hello, my name is Assistant. How can I help you today?")]),
            Content(role="user", parts=[Part(text="<interruption>Hello, my name is Assistant.</interruption>")]),
        ]

        result = await self.plugin.before_model_callback(
            callback_context=self.callback_context,
            llm_request=llm_request
        )

        self.assertIsNone(result)
        # Should have 2 contents: user message and modified model response
        self.assertEqual(len(llm_request.contents), 2)
        self.assertEqual(llm_request.contents[0].parts[0].text, "What's your name?")
        self.assertEqual(llm_request.contents[1].parts[0].text, "Hello, my name is Assistant.")

    async def test_interruption_with_punctuation_variations(self):
        """Test fuzzy matching handles punctuation differences."""
        llm_request = MagicMock(spec=LlmRequest)
        llm_request.contents = [
            Content(role="model", parts=[Part(text="Hello! How are you? I'm doing great!")]),
            Content(role="user", parts=[Part(text="<interruption>Hello How are you</interruption>")]),
        ]

        result = await self.plugin.before_model_callback(
            callback_context=self.callback_context,
            llm_request=llm_request
        )

        self.assertIsNone(result)
        self.assertEqual(len(llm_request.contents), 1)
        self.assertEqual(llm_request.contents[0].parts[0].text, "Hello How are you")

    async def test_interruption_with_case_differences(self):
        """Test fuzzy matching is case-insensitive."""
        llm_request = MagicMock(spec=LlmRequest)
        llm_request.contents = [
            Content(role="model", parts=[Part(text="HELLO WORLD")]),
            Content(role="user", parts=[Part(text="<interruption>hello world</interruption>")]),
        ]

        result = await self.plugin.before_model_callback(
            callback_context=self.callback_context,
            llm_request=llm_request
        )

        self.assertIsNone(result)
        self.assertEqual(len(llm_request.contents), 1)
        self.assertEqual(llm_request.contents[0].parts[0].text, "hello world")

    async def test_interruption_with_whitespace_differences(self):
        """Test fuzzy matching normalizes whitespace."""
        llm_request = MagicMock(spec=LlmRequest)
        llm_request.contents = [
            Content(role="model", parts=[Part(text="Hello    world   today")]),
            Content(role="user", parts=[Part(text="<interruption>Hello world</interruption>")]),
        ]

        result = await self.plugin.before_model_callback(
            callback_context=self.callback_context,
            llm_request=llm_request
        )

        self.assertIsNone(result)
        self.assertEqual(len(llm_request.contents), 1)
        self.assertEqual(llm_request.contents[0].parts[0].text, "Hello world")

    async def test_interruption_with_partial_match(self):
        """Test that interruption text can be a substring of model response."""
        llm_request = MagicMock(spec=LlmRequest)
        llm_request.contents = [
            Content(role="model", parts=[Part(text="Have you worked with Java? What frameworks do you prefer?")]),
            Content(role="user", parts=[Part(text="<interruption>Have you worked with Java?</interruption>")]),
        ]

        result = await self.plugin.before_model_callback(
            callback_context=self.callback_context,
            llm_request=llm_request
        )

        self.assertIsNone(result)
        self.assertEqual(len(llm_request.contents), 1)
        self.assertEqual(llm_request.contents[0].parts[0].text, "Have you worked with Java?")

    async def test_interruption_no_previous_model_response(self):
        """Test fallback when no previous model response exists."""
        llm_request = MagicMock(spec=LlmRequest)
        llm_request.contents = [
            Content(role="user", parts=[Part(text="Hello")]),
            Content(role="user", parts=[Part(text="<interruption>Some text</interruption>")]),
        ]

        result = await self.plugin.before_model_callback(
            callback_context=self.callback_context,
            llm_request=llm_request
        )

        self.assertIsNone(result)
        self.assertEqual(len(llm_request.contents), 2)
        self.assertEqual(llm_request.contents[0].parts[0].text, "Hello")
        # Should contain fallback message
        fallback_text = llm_request.contents[1].parts[0].text
        self.assertIn("User interrupted agent", fallback_text)
        self.assertIn("Some text", fallback_text)
        self.assertIn("No previous model response found", fallback_text)

    async def test_interruption_fuzzy_match_fails(self):
        """Test fallback when fuzzy matching fails."""
        llm_request = MagicMock(spec=LlmRequest)
        llm_request.contents = [
            Content(role="model", parts=[Part(text="Hello world")]),
            Content(role="user", parts=[Part(text="<interruption>Goodbye universe</interruption>")]),
        ]

        result = await self.plugin.before_model_callback(
            callback_context=self.callback_context,
            llm_request=llm_request
        )

        self.assertIsNone(result)
        self.assertEqual(len(llm_request.contents), 2)
        self.assertEqual(llm_request.contents[0].parts[0].text, "Hello world")
        # Should contain fallback message
        fallback_text = llm_request.contents[1].parts[0].text
        self.assertIn("User interrupted agent", fallback_text)
        self.assertIn("Goodbye universe", fallback_text)
        self.assertIn("Could not match interruption text", fallback_text)

    async def test_interruption_with_empty_text(self):
        """Test that empty interruption text is ignored."""
        llm_request = MagicMock(spec=LlmRequest)
        llm_request.contents = [
            Content(role="model", parts=[Part(text="Hello")]),
            Content(role="user", parts=[Part(text="<interruption></interruption>")]),
        ]

        result = await self.plugin.before_model_callback(
            callback_context=self.callback_context,
            llm_request=llm_request
        )

        self.assertIsNone(result)
        # Empty interruption should be passed through as-is
        self.assertEqual(len(llm_request.contents), 2)
        self.assertEqual(llm_request.contents[1].parts[0].text, "<interruption></interruption>")

    async def test_interruption_with_whitespace_only(self):
        """Test that whitespace-only interruption text is ignored."""
        llm_request = MagicMock(spec=LlmRequest)
        llm_request.contents = [
            Content(role="model", parts=[Part(text="Hello")]),
            Content(role="user", parts=[Part(text="<interruption>   \n\t  </interruption>")]),
        ]

        result = await self.plugin.before_model_callback(
            callback_context=self.callback_context,
            llm_request=llm_request
        )

        self.assertIsNone(result)
        # Whitespace-only interruption should be passed through as-is
        self.assertEqual(len(llm_request.contents), 2)
        self.assertEqual(llm_request.contents[1].parts[0].text, "<interruption>   \n\t  </interruption>")

    async def test_multiple_interruptions(self):
        """Test handling of multiple interruption markers."""
        llm_request = MagicMock(spec=LlmRequest)
        llm_request.contents = [
            Content(role="user", parts=[Part(text="First question")]),
            Content(role="model", parts=[Part(text="First answer with lots of text")]),
            Content(role="user", parts=[Part(text="<interruption>First answer</interruption>")]),
            Content(role="user", parts=[Part(text="Second question")]),
            Content(role="model", parts=[Part(text="Second answer with more details")]),
            Content(role="user", parts=[Part(text="<interruption>Second answer</interruption>")]),
        ]

        result = await self.plugin.before_model_callback(
            callback_context=self.callback_context,
            llm_request=llm_request
        )

        self.assertIsNone(result)
        # Should have 4 contents: user1, model1, user2, model2
        self.assertEqual(len(llm_request.contents), 4)
        self.assertEqual(llm_request.contents[0].parts[0].text, "First question")
        self.assertEqual(llm_request.contents[1].parts[0].text, "First answer")
        self.assertEqual(llm_request.contents[2].parts[0].text, "Second question")
        self.assertEqual(llm_request.contents[3].parts[0].text, "Second answer")

    async def test_interruption_with_assistant_role(self):
        """Test that plugin recognizes both 'model' and 'assistant' roles."""
        llm_request = MagicMock(spec=LlmRequest)
        llm_request.contents = [
            Content(role="assistant", parts=[Part(text="Hello, how can I help?")]),
            Content(role="user", parts=[Part(text="<interruption>Hello, how</interruption>")]),
        ]

        result = await self.plugin.before_model_callback(
            callback_context=self.callback_context,
            llm_request=llm_request
        )

        self.assertIsNone(result)
        self.assertEqual(len(llm_request.contents), 1)
        self.assertEqual(llm_request.contents[0].role, "assistant")
        self.assertEqual(llm_request.contents[0].parts[0].text, "Hello, how")

    async def test_interruption_with_multiline_text(self):
        """Test that interruption markers work with multiline text."""
        llm_request = MagicMock(spec=LlmRequest)
        llm_request.contents = [
            Content(role="model", parts=[Part(text="Line one\nLine two\nLine three")]),
            Content(role="user", parts=[Part(text="<interruption>Line one\nLine two</interruption>")]),
        ]

        result = await self.plugin.before_model_callback(
            callback_context=self.callback_context,
            llm_request=llm_request
        )

        self.assertIsNone(result)
        self.assertEqual(len(llm_request.contents), 1)
        self.assertEqual(llm_request.contents[0].parts[0].text, "Line one\nLine two")

    async def test_content_with_no_parts(self):
        """Test handling of content with no parts."""
        llm_request = MagicMock(spec=LlmRequest)
        llm_request.contents = [
            Content(role="user", parts=[]),
            Content(role="model", parts=[Part(text="Hello")]),
        ]

        result = await self.plugin.before_model_callback(
            callback_context=self.callback_context,
            llm_request=llm_request
        )

        self.assertIsNone(result)
        # Content with no parts should pass through
        self.assertEqual(len(llm_request.contents), 2)

    async def test_content_with_none_text_parts(self):
        """Test handling of parts with None text."""
        llm_request = MagicMock(spec=LlmRequest)
        part_with_none = Part()
        part_with_none.text = None
        llm_request.contents = [
            Content(role="model", parts=[part_with_none]),
            Content(role="user", parts=[Part(text="Hello")]),
        ]

        result = await self.plugin.before_model_callback(
            callback_context=self.callback_context,
            llm_request=llm_request
        )

        self.assertIsNone(result)
        # Should handle None text gracefully
        self.assertEqual(len(llm_request.contents), 2)

    async def test_multiple_parts_in_content(self):
        """Test handling of content with multiple parts."""
        llm_request = MagicMock(spec=LlmRequest)
        llm_request.contents = [
            Content(role="model", parts=[
                Part(text="First part. "),
                Part(text="Second part. "),
                Part(text="Third part.")
            ]),
            Content(role="user", parts=[Part(text="<interruption>First part. Second part.</interruption>")]),
        ]

        result = await self.plugin.before_model_callback(
            callback_context=self.callback_context,
            llm_request=llm_request
        )

        self.assertIsNone(result)
        self.assertEqual(len(llm_request.contents), 1)
        self.assertEqual(llm_request.contents[0].parts[0].text, "First part. Second part.")

    async def test_interruption_skips_user_messages(self):
        """Test that plugin skips user messages when looking for model response."""
        llm_request = MagicMock(spec=LlmRequest)
        llm_request.contents = [
            Content(role="model", parts=[Part(text="Hello there, friend!")]),
            Content(role="user", parts=[Part(text="Wait, let me think")]),
            Content(role="user", parts=[Part(text="Another user message")]),
            Content(role="user", parts=[Part(text="<interruption>Hello there</interruption>")]),
        ]

        result = await self.plugin.before_model_callback(
            callback_context=self.callback_context,
            llm_request=llm_request
        )

        self.assertIsNone(result)
        # Should find the first model response and replace it
        self.assertEqual(len(llm_request.contents), 3)
        self.assertEqual(llm_request.contents[0].parts[0].text, "Hello there")
        self.assertEqual(llm_request.contents[1].parts[0].text, "Wait, let me think")
        self.assertEqual(llm_request.contents[2].parts[0].text, "Another user message")

    async def test_normalize_text_helper(self):
        """Test the _normalize_text helper method directly."""
        # Test with various inputs
        self.assertEqual(
            self.plugin._normalize_text("Hello, World!"),
            "hello world"
        )
        self.assertEqual(
            self.plugin._normalize_text("  Multiple   spaces  "),
            "multiple spaces"
        )
        self.assertEqual(
            self.plugin._normalize_text("UPPERCASE"),
            "uppercase"
        )
        self.assertEqual(
            self.plugin._normalize_text("Punctuation!!! @#$"),
            "punctuation"
        )
        self.assertEqual(
            self.plugin._normalize_text("Line\nBreaks\rAnd\tTabs"),
            "line breaks and tabs"
        )

    async def test_is_fuzzy_substring_helper(self):
        """Test the _is_fuzzy_substring helper method directly."""
        # Should match with variations
        self.assertTrue(
            self.plugin._is_fuzzy_substring("hello", "Hello, World!")
        )
        self.assertTrue(
            self.plugin._is_fuzzy_substring("HELLO", "hello world")
        )
        self.assertTrue(
            self.plugin._is_fuzzy_substring("hello world", "Hello, World! How are you?")
        )

        # Should not match
        self.assertFalse(
            self.plugin._is_fuzzy_substring("goodbye", "Hello, World!")
        )
        self.assertFalse(
            self.plugin._is_fuzzy_substring("hello universe", "hello world")
        )

    async def test_get_content_text_helper(self):
        """Test the _get_content_text helper method directly."""
        # Multiple parts
        content = Content(role="model", parts=[
            Part(text="First "),
            Part(text="Second "),
            Part(text="Third")
        ])
        self.assertEqual(self.plugin._get_content_text(content), "First Second Third")

        # Single part
        content = Content(role="model", parts=[Part(text="Hello")])
        self.assertEqual(self.plugin._get_content_text(content), "Hello")

        # No parts
        content = Content(role="model", parts=[])
        self.assertEqual(self.plugin._get_content_text(content), "")

        # None text in parts
        part_with_none = Part()
        part_with_none.text = None
        content = Content(role="model", parts=[part_with_none, Part(text="Hello")])
        self.assertEqual(self.plugin._get_content_text(content), "Hello")

    async def test_find_previous_model_response_helper(self):
        """Test the _find_previous_model_response helper method directly."""
        contents = [
            Content(role="user", parts=[Part(text="First")]),
            Content(role="model", parts=[Part(text="Second")]),
            Content(role="user", parts=[Part(text="Third")]),
            Content(role="assistant", parts=[Part(text="Fourth")]),
            Content(role="user", parts=[Part(text="Fifth")]),
        ]

        # Find from end (index 5)
        idx = self.plugin._find_previous_model_response(contents, 5)
        self.assertEqual(idx, 3)  # Should find "assistant" at index 3

        # Find from index 4
        idx = self.plugin._find_previous_model_response(contents, 4)
        self.assertEqual(idx, 3)

        # Find from index 3
        idx = self.plugin._find_previous_model_response(contents, 3)
        self.assertEqual(idx, 1)  # Should find "model" at index 1

        # No model response before index 1
        idx = self.plugin._find_previous_model_response(contents, 1)
        self.assertIsNone(idx)

        # No model response before index 0
        idx = self.plugin._find_previous_model_response(contents, 0)
        self.assertIsNone(idx)

    async def test_extract_interruption_text_helper(self):
        """Test the _extract_interruption_text helper method directly."""
        # Valid interruption
        content = Content(role="user", parts=[Part(text="<interruption>Hello world</interruption>")])
        self.assertEqual(
            self.plugin._extract_interruption_text(content),
            "Hello world"
        )

        # Interruption with surrounding text
        content = Content(role="user", parts=[Part(text="Before <interruption>Middle</interruption> After")])
        self.assertEqual(
            self.plugin._extract_interruption_text(content),
            "Middle"
        )

        # No interruption marker
        content = Content(role="user", parts=[Part(text="Hello world")])
        self.assertIsNone(self.plugin._extract_interruption_text(content))

        # Empty interruption
        content = Content(role="user", parts=[Part(text="<interruption></interruption>")])
        self.assertEqual(
            self.plugin._extract_interruption_text(content),
            ""
        )

        # No parts
        content = Content(role="user", parts=[])
        self.assertIsNone(self.plugin._extract_interruption_text(content))

        # Multiple parts with interruption in one
        content = Content(role="user", parts=[
            Part(text="First part "),
            Part(text="<interruption>Interrupted</interruption>"),
            Part(text=" Last part")
        ])
        self.assertEqual(
            self.plugin._extract_interruption_text(content),
            "Interrupted"
        )


if __name__ == "__main__":
    unittest.main()
