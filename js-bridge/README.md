# pipecat-llm-bridge

The JavaScript / Bun half of the `pipecat-adk` WebSocket LLM bridge.

`WebSocketLLMService` (Python, in `pipecat_adk.websocket`) is a drop-in
replacement for `AdkLLMService`: it has the exact same black-box behaviour in
a Pipecat pipeline, but instead of running an LLM in-process it forwards every
turn to **this** process over a persistent WebSocket. That lets the LLM logic
live in TypeScript/Bun while Pipecat keeps owning real-time audio.

```
┌─────────── Pipecat (Python) ───────────┐         ┌──── this process (Bun) ────┐
 STT → VqlUserContextAggregator           │  WS      │  BridgeConnection           │
       → WebSocketLLMService  ───turn.run──────────▶ │   → LLMHandler.run()        │
       ← VqlLLMTextFrame, …    ◀──frame────────────  │   ← emit.text(), …          │
 TTS ← …                                  │         │                             │
└──────────────────────────────────────────┘        └─────────────────────────────┘
```

## Run

Requires [Bun](https://bun.sh) >= 1.1. No dependencies to install — the bridge
uses only the Bun runtime.

```bash
bun run src/index.ts          # listens on ws://127.0.0.1:8787 (EchoHandler)
PORT=9000 bun run src/index.ts
```

On startup it prints one line to stdout:

```
BRIDGE_LISTENING 8787
```

### Configuration (environment variables)

| Variable        | Default     | Meaning                                              |
|-----------------|-------------|------------------------------------------------------|
| `PORT`          | `8787`      | TCP port (`0` picks a free port).                    |
| `HOST`          | `127.0.0.1` | Interface to bind.                                   |
| `BRIDGE_SCRIPT` | _(unset)_   | JSON script → deterministic `ScriptedHandler`.       |

When `BRIDGE_SCRIPT` is unset the trivial `EchoHandler` is used.

## Implementing a real LLM

Implement `LLMHandler` (see `src/handler.ts`). The bridge owns the protocol
and the turn lifecycle (response-start / response-end framing, cancellation,
heartbeats); a handler only produces a turn's *content*.

```ts
import type { LLMHandler, TurnContext, TurnEmitter } from "./handler";

class MyHandler implements LLMHandler {
  async run(ctx: TurnContext, emit: TurnEmitter): Promise<void> {
    // ctx.text       — the user utterance
    // ctx.signal     — aborted when the pipeline interrupts; check it while streaming
    // ctx.turnId     — correlates the turn across the whole pipeline
    for await (const chunk of myModel.stream(ctx.text, { signal: ctx.signal })) {
      emit.text(chunk);
    }
    emit.usage({ prompt_tokens: 12, completion_tokens: 34, total_tokens: 46 });
  }

  // The bridge's [HEARD] provenance hook — record what the user actually heard.
  onTurnCompleted(turnId: string, text: string, interrupted: boolean): void {
    if (interrupted) myModel.recordHeard(turnId, text);
  }
}
```

Wire it up in `src/index.ts` (replace `createHandler`).

Function/tool calls are emitted within a single turn — the JS side owns the
whole tool loop:

```ts
emit.functionCall({ tool_call_id: "c1", function_name: "get_weather", arguments: { city } });
const result = await getWeather(city);
emit.functionResult({ tool_call_id: "c1", function_name: "get_weather", arguments: { city } }, result);
emit.text("It is sunny.");
```

## Wire protocol

JSON text frames over WebSocket. The contract is defined once in
`src/protocol.ts` (and mirrored in `pipecat_adk/websocket/protocol.py`).

**Python → JS:** `session.start`, `turn.run`, `turn.cancel`, `turn.completed`,
`session.end`, `ping`.

**JS → Python:** `session.ready`, `frame` (the `VqlLLM*` / `VqlFunctionCall*`
frames), `turn.usage`, `state.delta`, `error`, `pong`.

A turn is one `turn.run` answered by a stream of `frame` messages that begins
with `VqlLLMFullResponseStartFrame` and ends with `VqlLLMFullResponseEndFrame`
(the stream terminator). `turn.cancel` aborts an in-flight turn.

## Type-checking

```bash
bun run typecheck   # tsc --noEmit, no installed deps needed
```

`src/bun.d.ts` provides minimal ambient types for the Bun APIs used here so
`tsc` works offline. For full types, `bun add -d @types/bun` and delete that
file.
