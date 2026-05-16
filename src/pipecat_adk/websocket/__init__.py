"""WebSocket bridge: delegate LLM processing to an external (JavaScript) process.

``WebSocketLLMService`` is a black-box-equivalent replacement for
``AdkLLMService`` that forwards each turn to an external LLM component over a
persistent WebSocket connection.  See :mod:`.protocol` for the wire format.
"""

from . import protocol
from .client import WebSocketBridgeClient, WebSocketBridgeError
from .service import WebSocketLLMService

__all__ = [
    "WebSocketLLMService",
    "WebSocketBridgeClient",
    "WebSocketBridgeError",
    "protocol",
]
