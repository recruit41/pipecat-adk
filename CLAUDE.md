# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

**pipecat-adk** is a Python library that integrates Google ADK (Agent Development Kit) agents with Pipecat pipelines for real-time voice AI applications. It uses an "accountant's approach" for interruption handling: letting ADK commit events immediately, then using synthetic events and ADK plugins to filter conversation history.

## Development Commands

### Setup
```bash
# Install in development mode with uv
uv sync
```

### Testing
```bash
# Run all tests (verbose)
uv run python -m unittest discover -s tests -v

# Run all tests (short form)
uv run python -m unittest

# Run specific test file
uv run python -m unittest tests.test_context_aggregator -v

# Run specific test
uv run python -m unittest tests.test_context_aggregator.TestContextAggregator.test_basic_interaction
```

### Installation
```bash
# Install from local directory (for development)
pip install -e /path/to/pipecat-adk

# Or with uv
uv pip install -e /path/to/pipecat-adk
```

## Architecture

### Core Components

The library has 5 main components that work together, plus 3 frame types for state synchronization:

1. **SessionParams** (`types.py`): Simple dataclass for identifying ADK sessions (app_name, user_id, session_id)

2. **AdkUserContextAggregator** (`context_aggregators.py`):
   - Packages user input for ADK processing
   - Wraps speech in `<candidate>` tags
   - Clears Pipecat's context (ADK manages conversation history)
   - Extension point: `get_custom_context_parts()` for adding warnings/metadata

3. **AdkAssistantContextAggregator** (`context_aggregators.py`):
   - Tracks what was actually spoken to user using Pipecat's built-in aggregation
   - On interruption: adds synthetic event to ADK session with spoken portion
   - Blocks function call frames from polluting Pipecat context (ADK manages these)

4. **InterruptionHandlerPlugin** (`plugin.py`):
   - Exported publicly for client developers to include in their ADK App
   - ADK plugin that implements `before_model_callback` (official ADK pattern)
   - Runs automatically before every LLM request
   - Detects synthetic interruption events in conversation history
   - Filters out full interrupted responses, replacing with only spoken portions
   - Ensures LLM only sees what user actually heard
   - **MUST be included in the App's plugins list for proper interruption handling**

5. **AdkBasedLLMService** (`llm_service.py`):
   - Main service that replaces GoogleLLMService
   - Accepts an ADK App (not individual agent and plugins)
   - Invokes ADK agents via `runner.run_async()`
   - Converts ADK events to Pipecat frames
   - Pushes function call frames UPSTREAM and DOWNSTREAM to keep all processors informed
   - Handles state synchronization frames (`AdkAppendEventFrame`, `AdkInvokeAgentFrame`)
   - Emits `AdkStateDeltaFrame` when state changes occur

### State Synchronization Frames

The library provides three frame types for bidirectional state synchronization between Pipecat and ADK:

1. **AdkStateDeltaFrame** (`frames.py`):
   - Emitted when ADK produces state_delta (from tools or events)
   - Flows downstream to transport for client synchronization
   - Contains `state_delta` (dict) and `source` (author of the change)
   - Example use: Transport sends state updates to connected clients

2. **AdkAppendEventFrame** (`frames.py`):
   - Request to append an event to ADK session without invoking the LLM
   - Used for persisting state changes from client commands
   - If event contains state_delta, an `AdkStateDeltaFrame` is emitted
   - Example use: Client submits form data that should be persisted to session

3. **AdkInvokeAgentFrame** (`frames.py`):
   - Request to invoke ADK agent outside of normal speech flow
   - Contains `new_content` (Content) and optional `state_delta` (dict)
   - If state_delta provided, an `AdkStateDeltaFrame` is emitted before invocation
   - Example use: Client command that requires LLM response after processing

### Interruption Handling Flow

This is the key innovation of the library - understanding this flow is critical:

1. User interrupts bot mid-response
2. `AdkAssistantContextAggregator._handle_interruptions()` is called by Pipecat
3. Aggregator uses Pipecat's built-in text tracking to get spoken portion
4. Synthetic event is added to ADK session: `"<interruption>Hello, how can I...</interruption>"`
5. Event is committed to ADK session immediately (natural flow)
6. On next user message, ADK invokes `InterruptionHandlerPlugin.before_model_callback()`
7. Plugin scans conversation for interruption markers
8. Full interrupted response is replaced with just the spoken portion
9. Filtered conversation is passed to LLM
10. LLM only sees what user actually heard

### Function Call Lifecycle

Function calls are handled specially to keep Pipecat informed without polluting context:

1. ADK generates `function_call` event
2. `AdkBasedLLMService._handle_function_call()` creates frames:
   - `FunctionCallsStartedFrame` (pushed upstream & downstream)
   - `FunctionCallInProgressFrame` (pushed upstream & downstream)
3. Upstream processors receive frames (e.g., STTMuteFilter mutes microphone)
4. Downstream processors receive frames (e.g., for tracking)
5. ADK executes function internally
6. ADK generates `function_response` event
7. `AdkBasedLLMService._handle_function_response()` pushes `FunctionCallResultFrame` (upstream & downstream)
8. STTMuteFilter unmutes microphone
9. Function call frames are blocked from entering Pipecat context by `AdkAssistantContextAggregator`

## Code Style & Patterns

### ADK Agent Names
ADK requires agent names to use underscores, not hyphens. Always use `name="my_agent"` not `name="my-agent"`.

### Frame Pushing
The library pushes function call frames both UPSTREAM and DOWNSTREAM using:
```python
await self.push_frame(frame, FrameDirection.UPSTREAM)
await self.push_frame(frame, FrameDirection.DOWNSTREAM)
```

This is intentional - it keeps all pipeline processors informed of the function call lifecycle without polluting Pipecat's context.

### Context Management
- ADK manages the conversation history, NOT Pipecat
- `AdkUserContextAggregator` clears Pipecat context with `self._context.set_messages([])`
- The library only uses Pipecat context to pass messages to the LLM service, not to store history

### Synthetic Event Format
Interruption events use a specific format that the plugin expects:
```python
f"<interruption>{spoken_text}</interruption>"
```

The regex pattern `r'<interruption>(.*?)</interruption>'` extracts the spoken portion.

## Testing Patterns

### Unittest Configuration
Tests use `unittest.IsolatedAsyncioTestCase` for async test support. Test files use absolute imports (e.g., `from tests.mocks import MockLLM`) to work properly with unittest discovery.

### Mock Patterns
When mocking ADK agents, ensure the flow structure is present:
```python
agent = MagicMock(spec=Agent)
agent._llm_flow = MagicMock()
agent._llm_flow.request_processors = []
```

When mocking sessions, initialize the `_events` attribute:
```python
session = MagicMock(spec=Session)
session.events = []
session._events = []
```

### Test Organization
- `test_context_aggregators.py`: User and assistant aggregators (10 tests)
- `test_request_processors.py`: Interruption filtering logic (13 tests)
- `test_llm_service.py`: LLM service and frame conversion (16 tests)
- `test_integration.py`: End-to-end flows (7 tests)

## Important Implementation Details

### Why Function Calls Don't Enter Context
The assistant aggregator intentionally blocks function call frames via no-op methods. This is because ADK manages function calls internally, and we don't want them polluting Pipecat's context. However, the frames are still pushed upstream/downstream to inform other processors.

### Fuzzy Matching for Interruptions
The plugin uses fuzzy substring matching to verify that interrupted text is part of the agent's response. Text is normalized (lowercased, punctuation removed, whitespace normalized) before matching to handle minor variations in formatting.

### Plugin Session Modification
`InterruptionHandlerPlugin.before_model_callback()` modifies `llm_request.contents` in-memory (`plugin.py`). This is intentional - the modification only applies to the current request, not the persisted session. The plugin uses ADK's official callback pattern which is invoked before each LLM request.

## Dependencies

Core dependencies (from `pyproject.toml:24-31`):
- `pipecat-ai>=0.0.94`: Real-time audio/video pipeline framework
- `google-adk>=1.18.0`: Google Agent Development Kit (with App support)
- `google-genai>=1.51.0`: Google Generative AI SDK
- `google-cloud-texttospeech>=2.33.0`: Text-to-speech
- `google-cloud-speech>=2.34.0`: Speech recognition
- `loguru>=0.7.3`: Logging

No additional dev dependencies required - uses Python's built-in `unittest` for testing.

## Common Tasks

### Adding a New Extension Point
1. Identify where custom behavior is needed (user aggregator, assistant aggregator, or LLM service)
2. Add a method with default no-op or base implementation
3. Document with docstring example showing override pattern
4. Add test coverage for both default and custom implementations

### Debugging Interruptions
1. Enable debug logging: `logger.debug()` statements in `context_aggregators.py` and `plugin.py`
2. Check synthetic event format matches expected pattern: `<interruption>text</interruption>`
3. Verify plugin is registered with runner in `AdkBasedLLMService.__init__`
4. Verify spoken text is being tracked by Pipecat aggregation

### Basic Usage Pattern

The library requires you to create an ADK App with InterruptionHandlerPlugin:

```python
from google.adk.agents import Agent
from google.adk.apps.app import App
from pipecat_adk import AdkBasedLLMService, SessionParams, InterruptionHandlerPlugin
from google.adk.sessions import InMemorySessionService

# 1. Create your ADK agent
agent = Agent(
    name="my_agent",
    model="gemini-2.5-flash",
    instruction="You are a helpful assistant.",
)

# 2. Create ADK App with InterruptionHandlerPlugin
# IMPORTANT: InterruptionHandlerPlugin is required for proper interruption handling
app = App(
    name="my_app",
    root_agent=agent,
    plugins=[InterruptionHandlerPlugin()],  # Required!
)

# 3. Create session service
session_service = InMemorySessionService()
session_params = SessionParams(
    app_name=app.name,
    user_id="user_123",
    session_id="session_456",
)
await session_service.create_session(**session_params.model_dump())

# 4. Create LLM service with the App
llm = AdkBasedLLMService(
    session_service=session_service,
    session_params=session_params,
    app=app,  # Pass the App, not the agent
)
```

### Example: See `examples/assistant/agent.py`

The `examples/assistant/` directory demonstrates the recommended pattern:

- `agent.py`: Defines the ADK `Agent`, `App`, and includes `InterruptionHandlerPlugin`
- `bot.py`: Imports the `app` from `agent.py` and uses it with `AdkBasedLLMService`
- `run.py`: Entry point that starts the WebRTC server

This separation keeps agent configuration separate from runtime logic.

### Adding Support for New ADK Session Services

The library works with ANY ADK session service. Just pass it when creating `AdkBasedLLMService`:

```python
from custom_session_service import MySessionService

llm = AdkBasedLLMService(
    session_service=MySessionService(),  # Any ADK session service
    session_params=SessionParams(...),
    app=app,  # App with InterruptionHandlerPlugin
)
```
