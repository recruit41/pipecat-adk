"""Test utilities for working with ADK events.

Adapted from Google ADK's internal test helpers to make event assertions easier.
"""
from typing import Union
from google.genai import types


def simplify_events(events: list) -> list[tuple[str, Union[str, types.Part, list[types.Part]]]]:
    """Extracts the contents from events and transforms them into a list of
    (author, simplified_content) tuples.

    Args:
        events: List of ADK Event objects

    Returns:
        List of (author, simplified_content) tuples where simplified_content
        is either a string (for simple text), a Part, or a list of Parts
    """
    return [
        (event.author, simplify_content(event.content))
        for event in events
        if event.content
    ]


def simplify_content(content: types.Content) -> Union[str, types.Part, list[types.Part]]:
    """Simplifies content to make it easier to assert.

    - If there is only one part, returns that part
    - If the only part is pure text, returns the stripped text
    - If there are multiple parts, returns the list of parts
    - Removes function_call_id and function_response_id if they exist

    Args:
        content: ADK Content object

    Returns:
        Simplified content as string, Part, or list of Parts
    """
    # Remove function call/response IDs as they're non-deterministic
    for part in content.parts:
        if part.function_call and part.function_call.id:
            part.function_call.id = None
        if part.function_response and part.function_response.id:
            part.function_response.id = None

    # If single part, simplify further
    if len(content.parts) == 1:
        if content.parts[0].text:
            return content.parts[0].text.strip()
        else:
            return content.parts[0]

    return content.parts
