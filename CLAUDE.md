# CLAUDE.md

Development guide for internal contributors to pipecat-adk.

## Project Purpose

pipecat-adk integrates Google ADK agents with Pipecat pipelines for real-time voice AI. The core insight is that both frameworks excel at different things—Pipecat at real-time audio, ADK at agent logic—and combining them requires careful handling of the impedance mismatch between streaming audio and request/response patterns.

## Development Commands

```bash
# Setup
uv sync

# Run all tests
uv run python -m unittest discover -s tests -v

# Run specific test file
uv run python -m unittest tests.test_with_mocks -v

# Run specific test
uv run python -m unittest tests.test_with_mocks.TestIntegration.test_basic_interaction -v
```

## Architecture Overview

### Components

1. **SessionParams** (`types.py`): Session identification
2. **AdkUserContextAggregator** (`context_aggregators.py`): Packages user input for ADK
3. **AdkAssistantContextAggregator** (`context_aggregators.py`): Tracks spoken text, handles interruptions
4. **InterruptionHandlerPlugin** (`plugin.py`): Filters interrupted responses before LLM calls
5. **AdkBasedLLMService** (`llm_service.py`): Main service replacing GoogleLLMService
6. **Frame types** (`frames.py`): AdkStateDeltaFrame, AdkAppendEventFrame, AdkInvokeAgentFrame

### Data Flow

```
User speaks → STT → AdkUserContextAggregator → AdkBasedLLMService → ADK Runner
                                                       ↓
                    AdkAssistantContextAggregator ← TTS ← LLM response
```

## Design Decisions: The "Why"

### Why ADK Manages Context, Not Pipecat

**Decision**: Clear Pipecat's context after each message (`self._context.set_messages([])`).

**Why**: ADK provides sophisticated session management—persistence, state, multi-agent orchestration. Duplicating history in Pipecat would create synchronization bugs and prevent using ADK's features. The tradeoff is you can't use Pipecat's context inspection, but you get ADK's full session API.

### Why the "Accountant's Approach" to Interruptions

**Decision**: Let ADK commit the full response immediately, then filter it later.

**Why we rejected buffering**: We considered buffering responses in memory and only committing the spoken portion. This seemed cleaner but had fatal flaws:
- **Lost tool calls**: If the AI calls a tool mid-response and gets interrupted, we'd lose the tool execution entirely
- **Session inconsistency**: ADK's session wouldn't reflect what actually happened
- **Complexity**: Buffering in a streaming system creates subtle timing bugs

**Why the accountant's approach works**:
- Events are committed as they happen (audit trail preserved)
- Synthetic interruption events are just more events (natural extension)
- Plugin filters at read-time, not write-time (simpler, no race conditions)
- Tool calls are preserved even if the response is interrupted

The tradeoff is extra synthetic events in session history, but these are clearly marked and easy to filter for analytics.

### Why Fuzzy Matching for Interruptions

**Decision**: Use substring matching with normalized text (lowercase, no punctuation, collapsed whitespace).

**Why**: The AI's planned response and the actually-spoken text often differ slightly due to:
- TTS services adding/removing punctuation
- Streaming boundaries creating different chunk patterns
- Minor text normalization differences

We tried exact matching first—it failed ~20% of the time in testing. Fuzzy matching with normalization handles these variations. The risk of false positives is very low because we're matching within the same conversation turn.

### Why Push Function Call Frames Both Upstream and Downstream

**Decision**: Push `FunctionCallInProgressFrame` etc. in both directions.

**Why**: Different processors need to know about function calls:
- **Upstream**: `STTMuteFilter` needs to mute the mic during tool execution
- **Downstream**: UI needs to show "thinking..." indicators

Pushing only downstream would leave upstream processors uninformed. Pushing both ensures all processors can react appropriately. The frames don't enter the LLM context (blocked by assistant aggregator), so there's no pollution.

### Why Block Function Calls from Pipecat Context

**Decision**: Assistant aggregator blocks `FunctionCallInProgressFrame` and related frames from entering context.

**Why**: ADK manages tool calls internally. If we let these frames enter Pipecat's context, they'd create duplicate or malformed entries. The frames exist purely to inform other Pipecat processors of the lifecycle.

### Why Synthetic Events Use `<interruption>` Tags

**Decision**: Format is `<interruption>{spoken_text}</interruption>`.

**Why**: We need the plugin to reliably find and parse these events. XML-style tags are:
- Unambiguous (won't appear in natural speech)
- Easy to parse with regex
- Self-documenting in session history

The exact format doesn't matter much—consistency does.

## Testing Philosophy

### End-to-End Over Unit Tests

**Philosophy**: Test complete flows, not isolated components.

**Why**: The interesting bugs in this library are integration bugs:
- Interruption events not being created at the right time
- Function call frames not reaching the right processors
- State deltas not being emitted

Unit tests for individual methods miss these. Our tests simulate complete user interactions using `TestRunner` and verify the end result.

**Example of what we test**:
```python
async def test_interruption_handling(self):
    # Set up agent that will be interrupted
    mock_llm = MockLLM.conversation([...])

    async with TestRunner(app=app) as runner:
        # User speaks, bot starts responding
        # User interrupts mid-response
        # Bot receives next message
        # Verify bot only "remembers" what was actually spoken
```

This tests the full chain: user input → aggregator → service → ADK → plugin → response.

### Mock Infrastructure Design

**BotOutputTracker**: Shared state representing what the user sees/hears. Both input and output transports reference this to coordinate timing.

**MockLLM**: Simulates Gemini's streaming behavior:
- Streams text in chunks with `partial=True`
- Sends final chunk with `partial=None`
- Tracks requests for assertions
- Supports function calls via `Part.from_function_call()`

**MockInputTransport**: Executes a DSL of user actions:
- `Say(text)`: Simulates user speaking
- `WaitForResponse()`: Waits for bot to finish
- `InterruptAfter(text, say)`: Waits for specific text then interrupts

**MockSTTService/MockTTSService**: Convert frames symmetrically for testing. STT chunks input by words, TTS encodes text as fake audio.

**TestRunner**: Orchestrates everything:
- Creates pipeline with all mock services
- Manages session lifecycle
- Provides `simulate_user()` for running action sequences
- Provides `events()` and `session_state()` for assertions

### What Makes a Good Test

1. **Test complete scenarios**: Don't test "can the aggregator handle an interruption frame"—test "when a user interrupts, does the next response only reference what was spoken?"

2. **Use the DSL**: `Say()`, `WaitForResponse()`, `InterruptAfter()` make tests readable and express intent.

3. **Assert on outcomes, not internals**: Check `response.said("expected")` or session state, not internal method calls.

4. **One scenario per test**: Each test should have a clear "when X happens, Y should result" structure.

5. **Use `simplify_events()`**: Convert ADK events to `[("user", "Hello"), ("agent", "Hi")]` for readable assertions.

### Test Organization

- `test_with_mocks.py`: Main integration tests using TestRunner
- `test_state_sync.py`: State synchronization frame tests
- `test_plugin.py`: InterruptionHandlerPlugin edge cases
- `test_llm_service.py`: LLM service conversion tests
- `test_context_aggregator.py`: User aggregator tests
- `test_assistant_context_aggregator.py`: Assistant aggregator tests

## Manual Testing

### Running the Example Application

```bash
cd examples/assistant
pip install -r requirements.txt
pip install -e ../..  # Install pipecat-adk in development mode
export GEMINI_API_KEY=your_key
python run.py
```

Open http://localhost:7860 in your browser.

### What to Test Manually

1. **Basic conversation**: Say something, verify response makes sense
2. **Interruptions**: Start speaking while the bot is talking—verify the bot acknowledges being interrupted and doesn't reference unspoken content
3. **Multi-turn context**: Have a conversation, verify the bot remembers earlier turns
4. **Long responses**: Ask for something verbose, interrupt mid-sentence

### Debugging Tips

**Enable debug logging**: The library uses `loguru`. Set environment variable or configure in code:
```python
from loguru import logger
import sys
logger.add(sys.stderr, level="DEBUG")
```

**Check session history**: Add this to see what ADK recorded:
```python
session = await session_service.get_session(
    app_name=session_params.app_name,
    user_id=session_params.user_id,
    session_id=session_params.session_id
)
for event in session.events:
    print(f"{event.author}: {event.content}")
```

**Check interruption events**: Look for `<interruption>` tags in session history to verify the aggregator is creating them correctly.

**Check plugin filtering**: Add logging to `InterruptionHandlerPlugin.before_model_callback()` to see what it's filtering.

## Code Patterns

### Agent Names

ADK requires underscores: `name="my_agent"` not `name="my-agent"`.

### Frame Direction

Always push function call frames both ways:
```python
await self.push_frame(frame, FrameDirection.UPSTREAM)
await self.push_frame(frame, FrameDirection.DOWNSTREAM)
```

### Session Mocking

When mocking ADK sessions in tests:
```python
session = MagicMock(spec=Session)
session.events = []
session._events = []  # Internal attribute ADK uses
```

### Test Isolation

Use unique session IDs per test:
```python
session_id=f"test_session_{uuid.uuid4().hex[:8]}"
```

## Common Tasks

### Adding a New Frame Type

1. Define in `frames.py`:
```python
@dataclass
class MyNewFrame(DataFrame):
    my_field: str
```

2. Export in `__init__.py`

3. Handle in `AdkBasedLLMService.process_frame()`:
```python
if isinstance(frame, MyNewFrame):
    await self._handle_my_frame(frame)
```

4. Add tests in `test_state_sync.py` or new test file

### Adding an Extension Point

1. Add method with default implementation:
```python
async def my_extension_point(self):
    """Override this to customize behavior.

    Example:
        class MyAggregator(AdkUserContextAggregator):
            async def my_extension_point(self):
                return custom_value
    """
    return default_value
```

2. Call it where needed

3. Add tests for both default and overridden behavior

### Debugging Interruptions

1. Check that `AdkAssistantContextAggregator._handle_interruptions()` is being called
2. Verify the aggregator has accumulated text (it uses Pipecat's built-in tracking)
3. Check that the synthetic event is in session history with correct format
4. Verify `InterruptionHandlerPlugin` is in the App's plugins list
5. Add logging to `before_model_callback()` to see filtering

## Dependencies

- `pipecat-ai>=0.0.94`: Pipeline framework
- `google-adk>=1.18.0`: Agent Development Kit
- `google-genai>=1.51.0`: Generative AI SDK
- `google-cloud-texttospeech>=2.33.0`: TTS
- `google-cloud-speech>=2.34.0`: STT
- `loguru>=0.7.3`: Logging

No dev dependencies—uses Python's built-in `unittest`.
