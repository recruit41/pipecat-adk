# WebSocket LLM Bridge

`WebSocketLLMService` delegates LLM processing to an external process over a
persistent WebSocket. It is a **black-box-equivalent replacement for
`AdkLLMService`**: same inbound/outbound frames, same interruption semantics,
same matched start/end pairing — only the backend differs. Where
`AdkLLMService` drives `runner.run_async`, `WebSocketLLMService` forwards each
turn to a JavaScript/Bun process and streams the response frames back.

See [architecture.md](architecture.md) for the `AdkLLMService` contract this
mirrors, and [`js-bridge/README.md`](../js-bridge/README.md) for the external
component.

## Why

ADK owns agent logic in `AdkLLMService`. Some teams want that logic in
TypeScript instead. The bridge keeps Pipecat (audio, turns, interruption
correctness) in Python and moves only the LLM brain out of process, behind a
typed protocol — without the rest of the pipeline noticing.

## Components

| File | Role |
|------|------|
| `websocket/service.py` | `WebSocketLLMService` — the `LLMService`; same `process_frame` shape as `AdkLLMService`. `_run_adk` becomes `_run_remote`. |
| `websocket/client.py` | `WebSocketBridgeClient` — hardened connection: built on Pipecat's `WebsocketService` (auto-reconnect, exponential backoff, quick-failure detection), plus connect/turn timeouts, an application-level heartbeat, and fail-fast turn termination on connection loss. |
| `websocket/protocol.py` | The wire protocol — encoders (Python→JS), decoders, and Vql-frame reconstruction. Single source of truth, mirrored in `js-bridge/src/protocol.ts`. |
| `js-bridge/` | The external Bun component: `BridgeConnection` (protocol + turn lifecycle) and a pluggable `LLMHandler`. |

## Data flow

```
VqlContextFrame(turn_id, text)
  → WebSocketLLMService._run_remote
      client.run_turn → send  turn.run
      ◀ frame VqlLLMFullResponseStartFrame   → push downstream
      ◀ frame VqlLLMTextFrame                → push downstream
      ◀ frame VqlFunctionCall* / result      → push UPSTREAM + DOWNSTREAM
      ◀ turn.usage                           → LLM usage metrics
      ◀ frame VqlLLMFullResponseEndFrame     → stream terminator
  → push VqlLLMFullResponseEndFrame downstream  (always — finally block)

VqlTurnCompletedFrame(turn_id, text, interrupted)  (UPSTREAM)
  → consumed; send turn.completed   (the remote records its own [HEARD])
```

## Interruption

`VqlContextFrame` is a plain `Frame`, so `_run_remote` runs in Pipecat's
cancellable `__process_frame_task`. `VqlInterruptionFrame` cancels that task;
`CancelledError` propagates into `_run_remote`, which sends `turn.cancel` so
the remote process aborts generation. This mirrors how `AdkLLMService`'s ADK
runner is cancelled mid-stream.

## Hardening

- **Connect timeout** — `connect_timeout` bounds the WebSocket handshake.
- **Turn idle timeout** — `turn_timeout` bounds the gap between two messages
  of a turn; on expiry the turn fails with `on_completion_timeout` + an
  `ErrorFrame`, exactly like `AdkLLMService`'s `asyncio.TimeoutError` path.
- **Keepalive** — protocol-level `ping_interval` / `ping_timeout` detect a
  dead socket; an application-level heartbeat additionally detects a wedged
  remote event loop and forces a reconnect.
- **Reconnection** — automatic, with exponential backoff and quick-failure
  detection (inherited from Pipecat's `WebsocketService`).
- **Fail-fast turns** — if the socket drops mid-turn, the in-flight turn fails
  immediately instead of waiting for the idle timeout.
- **Malformed messages** — dropped with a warning; the connection survives.

## Usage

```python
from pipecat_adk import WebSocketLLMService

llm = WebSocketLLMService(uri="ws://localhost:8787")
aggregators = llm.create_context_aggregator()

pipeline = Pipeline([
    transport.input(),
    stt,
    aggregators.user(),
    llm,
    tts,                      # wrapped with VqlTTSMixin
    transport.output(),
    aggregators.assistant(),
])
```

The Vql aggregators and `VqlTTSMixin` are backend-agnostic — the exact same
pieces used with `AdkLLMService`.

## Testing

- `tests/test_websocket_bridge.py` — protocol unit tests, client hardening
  tests (against a controllable `FakeBridgeServer`), and full end-to-end tests
  against the real Bun bridge.
- `tests/ws_fake_server.py` — a deliberately-misbehaving Python bridge server
  for the hardening tests.
- `TestRunner(llm_service=...)` drives a full mock pipeline with any LLM
  service, not just `AdkLLMService`.
