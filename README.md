# pipecat-adk

**Build powerful voice-enabled AI agents by combining Pipecat's real-time audio pipelines with Google ADK's agent framework.**

## The Problem

**Pipecat** excels at real-time voice applications. It handles audio streaming, VAD, STT, TTS, and transport protocols beautifully. But as your application grows more complex—managing conversation history, handling interruptions correctly, persisting sessions, calling tools—things get difficult. Pipecat's context management wasn't designed for sophisticated agent workflows.

**Google ADK** (Agent Development Kit) excels at building agents. It provides rich concepts for sessions, state management, tool definitions, multi-agent orchestration, evaluations, and much more. But ADK wasn't designed for real-time voice—it expects request/response patterns, not streaming audio.

**pipecat-adk** bridges these two worlds, letting you build voice applications with Pipecat's real-time capabilities while leveraging ADK's agent framework for everything else.

## Installation

```bash
pip install pipecat-adk

# Or install from source
pip install -e /path/to/pipecat-adk
```

## Getting Started

If you have an existing Pipecat application, here's what you need to change:

### Before (Standard Pipecat)

```python
from pipecat.services.google import GoogleLLMService
from pipecat.services.google.llm import GoogleLLMContext
from pipecat.pipeline.pipeline import Pipeline

# Standard LLM service
llm = GoogleLLMService(
    model="gemini-2.0-flash",
    api_key=os.getenv("GEMINI_API_KEY"),
)

# Standard context aggregator
context_aggregator = llm.create_context_aggregator(
    GoogleLLMContext(messages=[{"role": "system", "content": "You are helpful"}])
)

pipeline = Pipeline([
    transport.input(),
    stt_service,
    context_aggregator.user(),
    llm,
    tts_service,
    transport.output(),
    context_aggregator.assistant(),
])
```

### After (With pipecat-adk)

```python
from pipecat_adk import AdkBasedLLMService, SessionParams, InterruptionHandlerPlugin
from google.adk.agents import Agent
from google.adk.apps import App
from google.adk.sessions import InMemorySessionService
from pipecat.pipeline.pipeline import Pipeline

# 1. Define your ADK agent
agent = Agent(
    name="helpful_assistant",  # Note: use underscores, not hyphens
    model="gemini-2.0-flash",
    instruction="You are a helpful assistant.",
)

# 2. Create an ADK App with the InterruptionHandlerPlugin
app = App(
    name="my_app",
    root_agent=agent,
    plugins=[InterruptionHandlerPlugin()],  # Required for interruption handling
)

# 3. Set up session management
session_service = InMemorySessionService()
session_params = SessionParams(
    app_name=app.name,
    user_id="user_123",
    session_id="session_456",
)
await session_service.create_session(**session_params.model_dump())

# 4. Create the LLM service
llm = AdkBasedLLMService(
    session_service=session_service,
    session_params=session_params,
    app=app,
)

# 5. Create context aggregators and pipeline (structure stays the same!)
context_aggregator = llm.create_context_aggregator()

pipeline = Pipeline([
    transport.input(),
    stt_service,
    context_aggregator.user(),
    llm,
    tts_service,
    transport.output(),
    context_aggregator.assistant(),
])
```

The pipeline structure stays the same—you just swap the LLM service and context aggregator.

## Key Challenges Solved

### 1. Context Management

**The Problem**: Pipecat manages conversation history by accumulating messages in an `LLMContext`. This works for simple cases, but breaks down when you need sophisticated history management, multi-turn reasoning, or agent handoffs.

**Our Solution**: ADK manages the conversation history. When a user speaks, our `AdkUserContextAggregator`:
1. Clears Pipecat's context (since ADK owns the history)
2. Passes the message to ADK for processing

The user's transcription is sent directly to ADK without modification.

**Tradeoff**: You can't use Pipecat's context inspection tools. All history lives in ADK sessions, which you access via `session_service.get_session()`.

### 2. Persistence and Replayability

**The Problem**: Pipecat's context is ephemeral—restart the server and you lose everything. Building features like conversation replay, analytics, or multi-device continuity requires custom persistence logic.

**Our Solution**: Use any ADK session service. ADK provides:
- `InMemorySessionService` for development
- `DatabaseSessionService` for production persistence
- Custom implementations for your specific needs

Every event is persisted automatically. You get:
- Full conversation history across restarts
- Ability to replay and analyze sessions
- Audit trails for compliance
- Session handoff between agents

**Tradeoff**: You need to manage session IDs and ensure they're unique per conversation. You also need to handle session cleanup/expiration.

### 3. Interruption Handling

**The Problem**: When a user interrupts the AI mid-sentence, standard LLM integrations send the full planned response to the model on the next turn. The AI then responds as if the user heard everything—leading to confusing conversations.

**Our Solution**: The "accountant's approach"—we don't try to prevent events from being committed. Instead:

1. ADK commits the full response immediately (natural flow)
2. When interrupted, `AdkAssistantContextAggregator` adds a synthetic event: `<interruption>Hello, how can...</interruption>` containing only what was actually spoken
3. Before the next LLM call, `InterruptionHandlerPlugin` scans the history and replaces the full response with just the spoken portion
4. The LLM only sees what the user actually heard

**Tradeoff**: The session history contains extra synthetic events. If you're analyzing raw session data, you'll need to filter these out. The plugin uses fuzzy matching to handle formatting variations, which very rarely might match incorrectly.

### 4. State Management

**The Problem**: Voice applications often need to track state—form data, user preferences, game scores, etc. Coordinating this state between the client and the AI is tedious.

**Our Solution**: ADK's state management flows through to clients via three frame types:

- **`AdkStateDeltaFrame`**: Emitted when ADK produces state changes (from tools or events). Send this to your client to sync state.
- **`AdkAppendEventFrame`**: Client sends this to persist an event without triggering an LLM response. Use for form submissions, state updates, etc.
- **`AdkInvokeAgentFrame`**: Client sends this to trigger an LLM response after a state change. Use when you need the AI to react to client input.

Example: A quiz application where the client submits answers:

```python
from google.adk.events import Event, EventActions
from google.genai import types as genai_types
from pipecat_adk import AdkAppendEventFrame, AdkInvokeAgentFrame

# Client submits answer (persist without LLM response)
event = Event(
    author="user",
    content=genai_types.Content(
        role="user",
        parts=[genai_types.Part(text="<answer>Paris</answer>")]
    ),
    actions=EventActions(state_delta={"score": 10})
)
await task.queue_frame(AdkAppendEventFrame(event=event))

# Or if you want the AI to respond to the submission
content = genai_types.Content(
    role="user",
    parts=[genai_types.Part(text="Check my answer: Paris")]
)
await task.queue_frame(AdkInvokeAgentFrame(
    new_content=content,
    state_delta={"last_answer": "Paris"}
))
```

**Tradeoff**: You need to handle these custom frame types in your transport. The library doesn't automatically send state to clients—you need to capture `AdkStateDeltaFrame` and send it through your WebSocket/WebRTC data channel.

### 5. Function Call Lifecycle

**The Problem**: When an AI calls a tool, you often want to mute the microphone (to avoid picking up silence as speech) or show a loading indicator. Pipecat has `FunctionCallInProgressFrame` for this, but standard integrations don't always emit it correctly.

**Our Solution**: When ADK generates a function call, we push frames both upstream and downstream:

1. `FunctionCallsStartedFrame` → enables components like `STTMuteFilter` to mute the mic
2. `FunctionCallInProgressFrame` → lets your UI show "thinking..." or similar
3. ADK executes the function
4. `FunctionCallResultFrame` → lets your UI show results, unmutes mic

These frames flow upstream (to processors before the LLM) and downstream (to processors after), keeping everyone informed.

**Tradeoff**: Function calls are managed entirely by ADK. You define tools using ADK's `FunctionTool` or similar, not Pipecat's function calling mechanism. The frames inform Pipecat of the lifecycle but don't give you hooks to intercept or modify the calls.

### 6. Custom Context Injection

**The Problem**: You need to inject dynamic context into conversations—current time, user preferences, system warnings, etc.

**Our Solution**: Override `_aggregation_to_content()` in `AdkUserContextAggregator`:

```python
from pipecat_adk.context_aggregators import AdkUserContextAggregator
from google.genai.types import Content, Part

class MyUserAggregator(AdkUserContextAggregator):
    async def _aggregation_to_content(self, aggregation: str) -> Content:
        parts = []

        # Add current time
        parts.append(Part(text=f"<system>Current time: {datetime.now()}</system>"))

        # Add user preferences from database
        prefs = await self.get_user_preferences()
        parts.append(Part(text=f"<system>User prefers {prefs.language}</system>"))

        # Add the actual user message
        parts.append(Part(text=aggregation))

        return Content(role="user", parts=parts)
```

These parts are included with every user message.

**Tradeoff**: This runs on every user message. Keep it lightweight—avoid slow database queries or API calls here.

## Complete Example

See [`examples/assistant/`](examples/assistant/) for a complete working application:

- **`agent.py`**: Defines the ADK Agent, App, and includes `InterruptionHandlerPlugin`
- **`bot.py`**: Sets up the Pipecat pipeline with `AdkBasedLLMService`
- **`run.py`**: FastAPI server for WebRTC signaling

To run:

```bash
cd examples/assistant
pip install -r requirements.txt
pip install -e ../..  # Install pipecat-adk in development mode
export GEMINI_API_KEY=your_key
python run.py
```

Open http://localhost:7860 to interact with the voice assistant.

## Testing Your Application

pipecat-adk provides a comprehensive mock testing infrastructure so you can test your agents without calling real APIs.

**Note**: The testing utilities are located in `tests/mocks.py` and `tests/test_utils.py`. To use them in your own tests, you'll need to copy these files or add the tests directory to your path.

### MockLLM

Create mock LLM responses for testing:

```python
# Copy tests/mocks.py to your project, then:
from mocks import MockLLM, TestRunner, Say, WaitForResponse

# Simple text response
mock_llm = MockLLM.single("Hello! How can I help?")

# Multi-turn conversation
mock_llm = MockLLM.conversation([
    "Hello! How can I help?",
    "Sure, I can help with that.",
])

# With function calls
from google.genai.types import Part
mock_llm = MockLLM.from_parts([
    [Part.from_function_call(name="get_weather", args={"city": "NYC"})],
    "The weather in NYC is sunny and 72°F.",
])
```

### TestRunner

Run end-to-end tests with simulated user actions:

```python
import unittest
from mocks import MockLLM, TestRunner, Say, WaitForResponse
from pipecat_adk import InterruptionHandlerPlugin
from google.adk.agents import Agent
from google.adk.apps import App

class TestMyAgent(unittest.IsolatedAsyncioTestCase):
    async def test_basic_conversation(self):
        # Create agent with mock LLM
        mock_llm = MockLLM.conversation([
            "Hello! I'm your assistant.",
            "I'd be happy to help with that.",
        ])

        agent = Agent(name="test_agent", model=mock_llm)
        app = App(
            name="agents",  # Must match TestRunner's expected app name
            root_agent=agent,
            plugins=[InterruptionHandlerPlugin()],
        )

        async with TestRunner(app=app) as runner:
            # First turn
            response = await runner.simulate_user([
                Say("Hello"),
                WaitForResponse(),
            ])
            assert response.said("Hello! I'm your assistant.")

            # Second turn
            response = await runner.simulate_user([
                Say("Can you help me?"),
                WaitForResponse(),
            ])
            assert response.said("I'd be happy to help with that.")
```

### User Actions DSL

Simulate user behavior with these actions:

- `Say(text)`: User speaks text
- `WaitForResponse()`: Wait for bot to finish speaking
- `WaitTillBotSays(text)`: Wait until bot says specific text
- `InterruptAfter(wait_for, say)`: Interrupt after hearing specific text

### Testing Interruptions

```python
async def test_interruption_handling(self):
    mock_llm = MockLLM.conversation([
        "Let me tell you a very long story about many things...",
        "Okay, I'll keep it brief.",
    ])

    agent = Agent(name="test_agent", model=mock_llm)
    app = App(
        name="agents",
        root_agent=agent,
        plugins=[InterruptionHandlerPlugin()],
    )

    async with TestRunner(app=app) as runner:
        # User interrupts mid-response
        response = await runner.simulate_user([
            Say("Tell me a story"),
            InterruptAfter("very long story", "Stop"),
        ])

        # Bot acknowledges interruption
        response = await runner.simulate_user([
            WaitForResponse(),
        ])
        assert response.said("brief")
```

### Testing State and Events

```python
async def test_session_state(self):
    # ... setup agent with tools that modify state ...

    async with TestRunner(app=app) as runner:
        await runner.simulate_user([
            Say("Set my preference to dark mode"),
            WaitForResponse(),
        ])

        # Check session state
        state = await runner.session_state()
        assert state.get("theme") == "dark"

        # Check session events
        events = await runner.events()
        # events is a list of all ADK events in the session
```

### Test Utilities

```python
from test_utils import simplify_events

# Convert ADK events to simple tuples for assertions
events = await runner.events()
simplified = simplify_events(events)

# Returns: [("user", "Hello"), ("agent", "Hi there!"), ...]
```

## API Reference

### Core Classes

- **`AdkBasedLLMService`**: Main LLM service that replaces `GoogleLLMService`
- **`SessionParams`**: Dataclass for session identification (app_name, user_id, session_id)
- **`InterruptionHandlerPlugin`**: ADK plugin for handling interruptions (must be in App's plugins)

### Frame Types

- **`AdkStateDeltaFrame`**: Emitted when state changes occur (state_delta, source)
- **`AdkAppendEventFrame`**: Request to append event without LLM invocation (event)
- **`AdkInvokeAgentFrame`**: Request to invoke agent with optional state (new_content, state_delta)

### Context Aggregators

Created via `llm.create_context_aggregator()`:
- **User aggregator**: Packages speech for ADK, clears Pipecat context
- **Assistant aggregator**: Tracks spoken text, handles interruptions

## Requirements

- Python >= 3.12
- pipecat-ai >= 0.0.94
- google-adk >= 1.18.0
- google-genai >= 1.51.0

## License

MIT License

## Contributing

Contributions are welcome! Please open an issue or submit a pull request.
