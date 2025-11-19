"""Plugins for ADK integration.

This module provides plugins that modify LLM requests before they are sent
to the model, enabling features like interruption handling using ADK's
official plugin pattern with before_model_callback.
"""

import difflib
import re
from typing import Optional

from google.adk.agents.callback_context import CallbackContext
from google.adk.models.llm_request import LlmRequest
from google.adk.models.llm_response import LlmResponse
from google.adk.plugins.base_plugin import BasePlugin
from google.genai.types import Content, Part
from loguru import logger


class InterruptionHandlerPlugin(BasePlugin):
    """Plugin that filters interrupted agent responses before model calls.

    This plugin implements ADK's before_model_callback to modify the LLM
    request before it's sent to the model, ensuring only text that was
    actually heard by the user is included.

    How it works:
        1. On interruption, AdkAssistantContextAggregator adds synthetic event:
           <interruption>text_that_was_heard</interruption>
        2. Event is committed to ADK session immediately (natural flow)
        3. Before next LLM request, before_model_callback is invoked
        4. Plugin finds <interruption> markers in the LLM request contents
        5. Locates the previous agent response (ignoring user messages)
        6. Uses fuzzy matching to verify interruption text is a substring
        7. If match: replaces full agent response with just interruption text
        8. If no match: converts to fallback message explaining interruption
        9. LLM receives filtered conversation reflecting what user heard

    Example:
        Original agent response: "Have you worked with Java? What frameworks?"
        User interrupts after: "Have you worked with Java?"
        Marker added: <interruption>Have you worked with Java?</interruption>
        Result: Agent response replaced with "Have you worked with Java?"

    Example usage:
        plugin = InterruptionHandlerPlugin()
        runner = Runner(
            agent=agent,
            session_service=session_service,
            app_name="my_app",
            plugins=[plugin]
        )
    """

    def __init__(self) -> None:
        """Initialize the plugin."""
        super().__init__(name="interruption_handler")

    async def before_model_callback(
        self, *, callback_context: CallbackContext, llm_request: LlmRequest
    ) -> Optional[LlmResponse]:
        # Access the contents from llm_request
        if not llm_request or not hasattr(llm_request, 'contents'):
            return None

        contents = llm_request.contents
        if not contents:
            return None

        # Process interruptions in the request contents
        modified = self._process_interruptions(contents)

        if modified:
            logger.debug("Modified LLM request to handle interruptions")

        # Return None to proceed with the LLM request
        return None

    def _process_interruptions(self, contents: list[Content]) -> bool:
        # Capture original state for diff logging
        original_contents = [
            {
                'role': c.role,
                'text': self._get_content_text(c)
            }
            for c in contents
        ]

        modified = False
        new_contents = []

        for content in contents:
            # Check if this content has an interruption marker
            interruption_text = self._extract_interruption_text(content)

            # Handle empty or whitespace-only interruption text
            if interruption_text and interruption_text.strip():
                # Find the previous model response in the new list
                model_idx = self._find_previous_model_response(new_contents, len(new_contents))

                # Try to match if we found a model response
                if model_idx is not None:
                    model_text = self._get_content_text(new_contents[model_idx])
                    if self._is_fuzzy_substring(interruption_text, model_text):
                        # Replace model response with interruption text
                        new_contents[model_idx] = Content(
                            role=new_contents[model_idx].role,
                            parts=[Part(text=interruption_text)]
                        )
                        # Don't add the interruption marker content
                        logger.debug(
                            f"Replaced model response with interruption text: "
                            f"'{interruption_text[:50]}...'"
                        )
                        modified = True
                        continue  # Skip adding this content

                # If we get here, either no model response or fuzzy match failed
                # Determine why we're using fallback
                if model_idx is None:
                    reason = "No previous model response found"
                    logger.warning(f"{reason} for interruption. Using fallback message.")
                else:
                    reason = "Could not match interruption text to model response"
                    logger.warning(
                        f"{reason}.\n"
                        f"  Expected: '{interruption_text[:100]}'\n"
                        f"  Found: '{self._get_content_text(new_contents[model_idx])[:100]}'"
                    )

                # Add fallback message with reason
                fallback = (
                    f"User interrupted agent. The user only heard "
                    f'"{interruption_text}" before interrupting. '
                    f"({reason})"
                )
                new_contents.append(Content(
                    role=content.role,
                    parts=[Part(text=fallback)]
                ))
                modified = True
            else:
                # No interruption marker, copy content as-is
                new_contents.append(content)

        # Log summary of changes if modifications were made
        if modified:
            self._log_content_diff(original_contents, new_contents)

        # Replace the original contents with the new list
        contents[:] = new_contents
        return modified

    def _extract_interruption_text(self, content: Content) -> Optional[str]:
        if not content or not content.parts:
            return None

        text = self._get_content_text(content)

        match = re.search(r'<interruption>(.*?)</interruption>', text, re.DOTALL)
        if match:
            return match.group(1)

        return None

    def _find_previous_model_response(
        self, contents: list[Content], current_idx: int
    ) -> Optional[int]:
        for i in range(current_idx - 1, -1, -1):
            content = contents[i]
            if content.role == "model" or content.role == "assistant":
                return i

        return None

    def _get_content_text(self, content: Content) -> str:
        if not content or not content.parts:
            return ""

        return "".join([part.text for part in content.parts if part.text])

    def _normalize_text(self, text: str) -> str:
        text = text.lower()
        text = re.sub(r'[^\w\s]', '', text)
        text = re.sub(r'\s+', ' ', text)
        text = text.strip()
        return text

    def _is_fuzzy_substring(self, needle: str, haystack: str) -> bool:
        normalized_needle = self._normalize_text(needle)
        normalized_haystack = self._normalize_text(haystack)

        # Check if normalized needle is in normalized haystack
        is_match = normalized_needle in normalized_haystack

        if not is_match:
            logger.debug(
                f"Fuzzy match failed:\n"
                f"  Needle: '{normalized_needle[:100]}'\n"
                f"  Haystack: '{normalized_haystack[:100]}'"
            )

        return is_match

    def _log_content_diff(self, original: list[dict], new: list) -> None:
        """Log a concise summary of changes made to LLM request contents."""
        # Convert to string representations for diffing
        def format_content(c):
            if isinstance(c, dict):
                text = c['text'].replace('\n', ' ')
                return f"[{c['role']}] {text}"
            else:
                text = self._get_content_text(c).replace('\n', ' ')
                return f"[{c.role}] {text}"

        original_lines = [format_content(c) for c in original]
        new_lines = [format_content(c) for c in new]

        diff = '\n'.join(difflib.unified_diff(
            original_lines,
            new_lines,
            fromfile='before',
            tofile='after',
            lineterm='',
            n=3
        ))

        # Log the diff directly
        if diff:
            logger.info(f"Interruption handling modified LLM request:\n{diff}")
        else:
            logger.info("Interruption handling: No changes to LLM request")
