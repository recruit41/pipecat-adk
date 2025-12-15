# Test Development Guide

Best practices for writing tests in pipecat-adk.

## Core Principles

### No Sleeps, Use Wait Methods
Never use `await asyncio.sleep()` to wait for pipeline events. Instead, use:
- `await runner.wait_for_response()` - wait for bot to finish speaking
- `await runner.wait_for(predicate)` - wait for custom conditions
- `await runner.stay_silent()` - ensure async messages are processed (not for timing)

The `wait_for()` method accepts a predicate `(bot_output, delta_messages) -> bool` and polls until true or timeout.

### Conversational DSL
Test methods should read like a conversation transcript:
```python
await runner.join()
await runner.speak_and_wait_for_response("Hello")
assert "welcome" in runner.last_bot_message.lower()

await runner.speak_and_wait_for_response("I need help")
assert runner.transcript == [
    Turn("user", "Hello"),
    Turn("bot", "Hello! How can I help?"),
    Turn("user", "I need help"),
    Turn("bot", "I'd be happy to assist."),
]
```

## Quick Start

```python
from google.adk.agents import Agent
from google.adk.apps import App
from pipecat_adk import InterruptionHandlerPlugin
from tests.mocks import MockLLM, TestRunner, Turn

class TestBasicFlow(unittest.IsolatedAsyncioTestCase):
    async def test_greeting(self):
        mock_llm = MockLLM.conversation([
            "Hello! Welcome.",
            "Thanks for sharing."
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
            await runner.join()
            await runner.speak_and_wait_for_response("Hi")
            assert runner.last_bot_message == "Hello! Welcome."

            await runner.speak_and_wait_for_response("I have a question")
            assert runner.transcript == [
                Turn("user", "Hi"),
                Turn("bot", "Hello! Welcome."),
                Turn("user", "I have a question"),
                Turn("bot", "Thanks for sharing."),
            ]
```

## MockLLM Responses

### Text Responses
```python
MockLLM.single("Hello!")
MockLLM.conversation(["Hello!", "How are you?", "Great!"])
```

**Important**: The number of MockLLM responses must match the number of conversation turns:
```python
# Bad - only 1 response but test expects 2 turns
mock_llm = MockLLM.single("Hello!")
await runner.speak_and_wait_for_response("hi")  # Uses response 1
await runner.speak_and_wait_for_response("test")  # Timeout - no response 2!

# Good - provide enough responses
mock_llm = MockLLM.conversation(["Hello!", "I see."])
```

### Function Calls
```python
from google.genai.types import Part

MockLLM.from_parts([
    # Turn with function call
    [Part.from_function_call(name="get_weather", args={"location": "NYC"})],
    # Turn with text
    "The weather is sunny.",
    # Turn with multiple function calls
    [
        Part.from_function_call(name="set_temp", args={"degrees": 72}),
        Part.from_function_call(name="set_lights", args={"brightness": 80}),
    ],
    "Done adjusting settings.",
])
```

## Runner Methods

| Method | Description |
|--------|-------------|
| `join()` | Simulate client connecting |
| `speak(text)` | Inject user speech |
| `speak_and_wait_for_response(text)` | Speak and wait for bot reply |
| `interrupt_bot(text)` | Wait for bot to start speaking, then interrupt |
| `push_message(type, data)` | Send client message |
| `queue_frame(frame)` | Queue a frame (e.g., AdkAppendEventFrame) into pipeline |
| `stay_silent()` | Push silence frames to process async messages |
| `wait_for_response()` | Wait for bot to finish speaking |
| `wait_for(predicate)` | Wait for custom condition |

## Runner Properties (Black-Box Assertions)

| Property | Type | Description |
|----------|------|-------------|
| `transcript` | `List[Turn]` | Chronological conversation history |
| `bot_messages` | `List[str]` | All bot utterances |
| `last_bot_message` | `str` | Most recent bot utterance |
| `messages` | `List[dict]` | Raw RTVI messages sent to client |
| `client_state` | `dict` | Merged state-sync deltas |

## Gray-Box Inspection

For testing ADK session state and events:
```python
# Get current session state
state = await runner.session_state()
assert state.get("current_section") == "screening"

# Get all session events
events = await runner.events()
```

## Testing Interruptions

Use `tts_delay` to slow TTS output so interruptions can occur mid-utterance:

```python
async with TestRunner(app=app, tts_delay=0.05) as runner:
    await runner.join()
    await runner.speak("Tell me a story")
    await runner.interrupt_bot("Wait, stop")
    # Bot was interrupted mid-speech
```

## Waiting for Async Message Processing

Messages sent via `push_message()` are processed asynchronously. If the message triggers an LLM response, use `wait_for_response()`:

```python
await runner.push_message("some-trigger", data)
await runner.wait_for_response()
```

If the message does NOT trigger an LLM response, use `stay_silent()` to ensure it's processed:

```python
await runner.push_message("state-update", data)
await runner.stay_silent()
state = await runner.session_state()  # Now safe to assert
```

## Queueing Frames Directly

Use `queue_frame()` to inject ADK frames directly into the pipeline:

```python
# Append an event to ADK session (no LLM response)
event = Event(author="user", actions=EventActions(state_delta={"key": "value"}))
await runner.queue_frame(AdkAppendEventFrame(event=event))
await runner.stay_silent()  # Process async frame

# Invoke agent with new content (triggers LLM response)
content = Content(role="user", parts=[Part(text="Please summarize.")])
await runner.queue_frame(AdkInvokeAgentFrame(new_content=content))
await runner.wait_for_response()  # Wait for LLM response
```

## Custom Wait Conditions

For complex assertions, use `wait_for()`:

```python
async def wait_for_state_sync():
    def _has_state(bot_output, delta):
        return any(
            m.get("type") == "state-sync" and
            m.get("state_delta", {}).get("myKey")
            for m in delta
        )
    await runner.wait_for(_has_state, timeout=2.0)
```

## Key Files

| File | Purpose |
|------|---------|
| `mocks.py` | `MockLLM`, `TestRunner`, `MockTTSService`, `MockSTTService` |
| `test_with_mocks.py` | Main integration tests |
| `test_state_sync.py` | State synchronization frame tests |
| `test_utils.py` | `simplify_events()` for readable event assertions |
