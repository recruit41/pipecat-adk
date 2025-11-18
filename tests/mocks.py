from typing import AsyncGenerator, List
from typing_extensions import override

from google.adk.models.base_llm import BaseLlm
from google.adk.models.llm_request import LlmRequest
from google.adk.models.llm_response import LlmResponse
from google.genai.types import Content, Part

class MockLLM(BaseLlm):
    """
    Mock LLM model that returns predefined responses with realistic streaming.

    Follows ADK's testing pattern from google.adk.testing_utils.MockModel.
    This allows tests to control what the agent says without calling real LLM APIs.

    Usage:
        # Simple single response
        mock_llm = MockLLM.single("Hello! Welcome to your interview.")

        # Multi-turn text conversation
        mock_llm = MockLLM.conversation([
            "Hello! Welcome.",
            "How are you today?",
            "Great! Let's begin."
        ])

        # Complex: function calls + text
        mock_llm = MockLLM.from_parts([
            [Part.from_function_call(name="change_section", args={"to_section_key": "technical"})],
            "Let's move to some technical topics now. Have you worked with Spring Boot?"
        ])

        # Complex: multiple function calls in one turn
        mock_llm = MockLLM.from_parts([
            [
                Part.from_function_call(name="change_section", args={...}),
                Part.from_function_call(name="update_notes", args={...})
            ],
            "Let's discuss your Java experience."
        ])
    """
    model: str = "mock"
    requests: List[LlmRequest] = []
    responses: List[List[Part]]  # Each turn is a list of Parts
    response_index: int = -1

    @classmethod
    def single(cls, text: str):
        """
        Create a MockLLM that returns a single text response.

        Args:
            text: The text response to return

        Returns:
            MockLLM instance

        Example:
            mock_llm = MockLLM.single("Hello! Welcome to your interview.")
        """
        return cls(responses=[[Part.from_text(text=text)]])

    @classmethod
    def conversation(cls, texts: List[str]):
        """
        Create a MockLLM that returns multiple text responses (multi-turn).

        Args:
            texts: List of text responses, one per turn

        Returns:
            MockLLM instance

        Example:
            mock_llm = MockLLM.conversation([
                "Hello! Welcome.",
                "How are you today?",
                "Great! Let's begin."
            ])
        """
        return cls(responses=[[Part.from_text(text=text)] for text in texts])

    @classmethod
    def from_parts(cls, turns):
        """
        Create a MockLLM with full control over Parts (text, function calls, etc.).

        Args:
            turns: List where each item is either:
                - str: Text response for that turn
                - List[Part]: Multiple Parts for that turn (function calls, text, etc.)

        Returns:
            MockLLM instance

        Examples:
            # Function call followed by text
            mock_llm = MockLLM.from_parts([
                [Part.from_function_call(name="change_section", args={...})],
                "Let's move to technical topics."
            ])

            # Multiple function calls in one turn
            mock_llm = MockLLM.from_parts([
                [
                    Part.from_function_call(name="change_section", args={...}),
                    Part.from_function_call(name="start_quiz", args={...})
                ],
                "Here's a quiz about Java."
            ])
        """
        normalized = cls._normalize_responses(turns)
        return cls(responses=normalized)

    @classmethod
    def _normalize_responses(cls, responses) -> List[List[Part]]:
        """Convert flexible input format to List[List[Part]]."""
        # Single string -> [[Part]]
        if isinstance(responses, str):
            return [[Part.from_text(text=responses)]]

        # List of items
        if isinstance(responses, list):
            normalized = []
            for item in responses:
                if isinstance(item, str):
                    # String -> [Part]
                    normalized.append([Part.from_text(text=item)])
                elif isinstance(item, list):
                    # List of Parts or strings -> [Part, Part, ...]
                    turn_parts = []
                    for part_or_str in item:
                        if isinstance(part_or_str, str):
                            turn_parts.append(Part.from_text(text=part_or_str))
                        else:
                            # Assume it's already a Part
                            turn_parts.append(part_or_str)
                    normalized.append(turn_parts)
                else:
                    # Single Part -> [Part]
                    normalized.append([item])
            return normalized

        # Fallback: treat as single Part
        return [[responses]]

    @classmethod
    def _split_text_for_streaming(cls, text: str, num_chunks: int = 2) -> List[str]:
        """
        Split text into chunks for realistic streaming simulation.

        Splits can occur mid-word to simulate real streaming behavior.
        Short texts (<20 chars) are not split.
        """
        if len(text) < 20:
            return [text]

        # Split into roughly equal chunks
        chunk_size = len(text) // num_chunks
        chunks = []

        for i in range(num_chunks - 1):
            start = i * chunk_size
            end = (i + 1) * chunk_size
            chunks.append(text[start:end])

        # Last chunk gets the remainder
        chunks.append(text[(num_chunks - 1) * chunk_size:])

        return chunks

    @classmethod
    @override
    def supported_models(cls) -> list[str]:
        return ["mock"]

    @override
    async def generate_content_async(
        self, llm_request: LlmRequest, stream: bool = True
    ) -> AsyncGenerator[LlmResponse, None]:
        """
        Generate mock responses asynchronously.

        Streaming behavior (stream=True):
        - For text: Yields 2-3 partial chunks with partial=True, then complete text with partial=None
        - For function calls: Yields once with partial=True, then with partial=None

        Non-streaming (stream=False):
        - Only yields final response with partial=None

        Function Call Handling:
        When ADK receives a function_call from LLM, it executes the function and calls
        generate_content_async again with the function_response in the request.
        MockLLM increments the response index normally - each call gets the next response.
        """
        # Increment index for each LLM call
        self.response_index += 1
        self.requests.append(llm_request)

        # Check if we have a response for this turn
        if self.response_index >= len(self.responses):
            return

        parts = self.responses[self.response_index]

        if stream:
            # Streaming mode: emit realistic chunks
            for part in parts:
                # Check if this is a text part that can be chunked
                if hasattr(part, 'text') and part.text:
                    # Split text into chunks for streaming
                    chunks = self._split_text_for_streaming(part.text)

                    # Yield each chunk as partial
                    for chunk in chunks:
                        chunk_content = Content(role="model", parts=[Part.from_text(text=chunk)])
                        yield LlmResponse(content=chunk_content, partial=True)
                    
                    # Now yield the entire text as final
                    final_content = Content(role="model", parts=[part])
                    yield LlmResponse(content=final_content, partial=None)
                else:
                    # Non-text parts (function calls, etc.) - yield as-is
                    part_content = Content(role="model", parts=[part])
                    yield LlmResponse(content=part_content)

        else:
            # Non-streaming mode: only yield final event
            final_content = Content(role="model", parts=parts)
            yield LlmResponse(content=final_content, partial=None)

