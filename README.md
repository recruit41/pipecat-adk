# Pipecat-ADK

Integrate Google ADK agents with Pipecat pipelines for real-time voice AI applications.

## Features

- ✅ Use ADK agents instead of standard LLM services
- ✅ Automatic interruption handling with synthetic events
- ✅ Works with ANY ADK session service
- ✅ Keep Pipecat informed of function call lifecycle
- ✅ Extension points for custom frames and RTVI integration
- ✅ No TTS monkey-patching required
- ✅ No custom frame types needed

## How It Works

Unlike buffering approaches, pipecat-adk uses an "accountant's approach":

1. Let ADK commit events immediately (natural flow)
2. On interruption, add synthetic event: "User heard only: X"
3. Request processor filters conversation before each LLM call
4. LLM only sees what user actually heard

## Installation

```bash
# Install from local directory (development)
pip install -e /path/to/pipecat-adk

# Or with uv
uv pip install -e /path/to/pipecat-adk
```

## Quick Start

```python
from pipecat_adk import AdkBasedLLMService, SessionParams
from google.adk.agents import Agent
from google.adk.sessions import InMemorySessionService
from pipecat.pipeline.pipeline import Pipeline

# Create your ADK agent
agent = Agent(
    name="assistant",
    model="gemini-2.0-flash",
    instruction="You are a helpful assistant"
)

# Create session service (any ADK session service works)
session_service = InMemorySessionService()

# Create LLM service
llm = AdkBasedLLMService(
    session_service=session_service,
    session_params=SessionParams(
        app_name="my-app",
        user_id="user",
        session_id="session-123"
    ),
    agent=agent,
    api_key="your-gemini-api-key"
)

# Create pipeline with context aggregators
context_aggregator = llm.create_context_aggregator()

pipeline = Pipeline([
    transport.input(),
    stt_service,
    context_aggregator.user(),
    llm,
    tts_service,
    transport.output(),
    context_aggregator.assistant()
])
```

See [examples/](examples/) for complete working examples.

## Extension Points

### Custom Context

Add custom context (warnings, metadata) to user messages:

```python
from pipecat_adk import AdkUserContextAggregator
from google.genai.types import Part

class MyUserAggregator(AdkUserContextAggregator):
    async def get_custom_context_parts(self):
        parts = []

        # Add device warnings
        if warning := await self.check_device_status():
            parts.append(Part(text=f"<system>{warning}</system>"))

        # Add time tracking
        if elapsed_time := await self.get_elapsed_time():
            parts.append(Part(text=f"<system>Elapsed: {elapsed_time}s</system>"))

        return parts
```

### Custom Frames

Handle custom application frames:

```python
from pipecat_adk import AdkBasedLLMService

class MyLLMService(AdkBasedLLMService):
    async def process_frame(self, frame, direction):
        if isinstance(frame, MyCustomFrame):
            # Convert to ADK event
            await self.handle_custom_frame(frame)
        else:
            await super().process_frame(frame, direction)

    async def handle_custom_frame(self, frame):
        # Add event to ADK session
        session = await self.session_service.get_session(...)
        event = Event(
            author="user",
            content=Content(role="user", parts=[
                Part(text=f"<system>{frame.data}</system>")
            ])
        )
        await self.session_service.append_event(session, event)
```

### RTVI Integration

Handle RTVI protocol messages:

```python
from pipecat_adk import AdkBasedLLMService

class RTVIAwareLLMService(AdkBasedLLMService):
    async def handle_rtvi_frame(self, frame, direction):
        message_type = frame.data.get('type')

        if message_type == 'device-state-changed':
            await self._handle_device_state(frame.data)
        elif message_type == 'quiz-answer-submitted':
            await self._handle_quiz_answer(frame.data)
```

## Architecture

### Interruption Handling

When a user interrupts the AI:

1. **Pipecat** detects interruption and calls `AssistantContextAggregator._handle_interruptions()`
2. **Aggregator** collects the text that was actually spoken to the user
3. **Synthetic Event** is added to ADK session: "Previous response interrupted, user heard: 'Hello, how can I...'"
4. **Request Processor** filters conversation history before next LLM request
5. **LLM** only sees the portion the user actually heard

This approach:
- ✅ Works with any ADK session service (no custom wrappers)
- ✅ Session reflects reality (what actually happened)
- ✅ Uses ADK's built-in extension points
- ✅ No TTS monkey-patching required

### Function Call Lifecycle

When ADK agent calls a function:

1. **ADK** generates `function_call` event
2. **LLM Service** converts to Pipecat frames:
   - `FunctionCallsStartedFrame` (pushed upstream & downstream)
   - `FunctionCallInProgressFrame` (pushed upstream & downstream)
3. **STTMuteFilter** receives frames upstream, mutes microphone
4. **UserIdleProcessor** receives frames, pauses idle detection
5. **ADK** executes function and generates `function_response` event
6. **LLM Service** pushes `FunctionCallResultFrame` (upstream & downstream)
7. **STTMuteFilter** unmutes microphone
8. **UserIdleProcessor** resumes idle detection

This keeps Pipecat fully informed of the function call lifecycle without polluting the LLM context.

## Development

```bash
# Install in development mode with uv
uv sync

# Run tests
uv run python -m unittest discover -s tests -v

# Run tests with coverage
uv run python -m coverage run -m unittest discover -s tests
uv run python -m coverage report
uv run python -m coverage html  # Generate HTML report
```

## Examples

See [examples/basic_chatbot.py](examples/basic_chatbot.py) for a complete working example showing:
- Setting up ADK agents with Pipecat pipelines
- Automatic interruption handling
- Daily.co transport integration
- Voice input/output with Google STT/TTS

For advanced usage patterns, see the Extension Points section above for examples of:
- Adding custom context to user messages
- Handling custom application frames
- Integrating with RTVI protocol

## License

MIT License - see [LICENSE](LICENSE) for details.

## Contributing

Contributions are welcome! Please open an issue or submit a pull request.
