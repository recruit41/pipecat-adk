/**
 * LLM handler abstraction for the bridge.
 *
 * The bridge owns the protocol and the turn lifecycle (it emits the
 * response-start / response-end frames). An `LLMHandler` only has to produce
 * the *content* of a turn: text chunks, function calls, state deltas, usage.
 *
 * Plug in a real model by implementing `LLMHandler`. Two reference handlers
 * are provided: `EchoHandler` (trivial) and `ScriptedHandler` (deterministic,
 * used by the end-to-end tests).
 */

import type { FunctionCall, TokenUsage } from "./protocol";

/** Everything a handler needs to know about the turn it is running. */
export interface TurnContext {
  /** Pipecat turn id — correlates this turn across the whole pipeline. */
  readonly turnId: string;
  /** Bridge-generated id for this LLM invocation (the ADK-`invocation_id` analogue). */
  readonly invocationId: string;
  /** The user utterance for this turn. */
  readonly text: string;
  /** Optional state delta forwarded from the pipeline. */
  readonly stateDelta: Record<string, unknown> | null;
  /** Aborted when the pipeline interrupts the turn (`turn.cancel`). */
  readonly signal: AbortSignal;
}

/** Sink a handler uses to stream a turn's content back to the pipeline. */
export interface TurnEmitter {
  /** Emit a chunk of streamed assistant text. */
  text(chunk: string): void;
  /** Announce a function/tool call. */
  functionCall(call: FunctionCall): void;
  /** Report the result of a previously announced function call. */
  functionResult(call: FunctionCall, result: unknown): void;
  /** Emit a state delta for the client (RTVI state-sync etc.). */
  stateDelta(delta: Record<string, unknown>): void;
  /** Report token usage for the turn. */
  usage(usage: Partial<TokenUsage>): void;
}

/** The LLM logic plugged into the bridge. */
export interface LLMHandler {
  /** Generate one turn. Resolves when the turn is complete. */
  run(ctx: TurnContext, emit: TurnEmitter): Promise<void>;
  /** Record what the user actually heard — the `[HEARD]` provenance hook. */
  onTurnCompleted?(turnId: string, text: string, interrupted: boolean): void | Promise<void>;
  /** Called once when the pipeline session opens. */
  onSessionStart?(
    session: Record<string, unknown>,
    metadata: Record<string, unknown>,
  ): void | Promise<void>;
  /** Called once when the pipeline session ends. */
  onSessionEnd?(): void | Promise<void>;
}

/** Raised by a handler to abort a turn cooperatively when `signal` fires. */
export class TurnAbortedError extends Error {
  constructor() {
    super("turn aborted");
    this.name = "TurnAbortedError";
  }
}

function splitWords(text: string): string[] {
  return text.match(/\S+\s*/g) ?? [];
}

/** Trivial handler: echoes the user's utterance back. */
export class EchoHandler implements LLMHandler {
  async run(ctx: TurnContext, emit: TurnEmitter): Promise<void> {
    const reply = `You said: ${ctx.text}`;
    for (const word of splitWords(reply)) {
      if (ctx.signal.aborted) throw new TurnAbortedError();
      emit.text(word);
    }
    emit.usage({
      prompt_tokens: ctx.text.length,
      completion_tokens: reply.length,
      total_tokens: ctx.text.length + reply.length,
    });
  }
}

// ---------------------------------------------------------------------------
// ScriptedHandler — deterministic responses for tests and demos
// ---------------------------------------------------------------------------

/** A single op within a scripted turn. */
export type ScriptOp =
  | { text: string; chunks?: number; delayMs?: number }
  | { call: { name: string; args?: Record<string, unknown>; result?: unknown } };

/** A scripted turn is an ordered list of ops emitted within one response. */
export type ScriptTurn = ScriptOp[];

/** A script is one entry per user turn, consumed in order. */
export type Script = ScriptTurn[];

/** Normalise loose script input (a bare string becomes a one-text-op turn). */
export function normaliseScript(raw: unknown): Script {
  if (!Array.isArray(raw)) {
    throw new Error("script must be an array of turns");
  }
  return raw.map((turn): ScriptTurn => {
    if (typeof turn === "string") return [{ text: turn }];
    if (!Array.isArray(turn)) {
      throw new Error("each script turn must be a string or an array of ops");
    }
    return turn.map((op): ScriptOp => {
      if (typeof op === "string") return { text: op };
      return op as ScriptOp;
    });
  });
}

function chunkText(text: string, chunks: number): string[] {
  if (chunks <= 1 || text.length <= chunks) return [text];
  const size = Math.ceil(text.length / chunks);
  const out: string[] = [];
  for (let i = 0; i < text.length; i += size) out.push(text.slice(i, i + size));
  return out;
}

const sleep = (ms: number): Promise<void> =>
  new Promise((resolve) => setTimeout(resolve, ms));

/**
 * Replays a fixed script — the JavaScript analogue of the Python `MockLLM`.
 *
 * Each call to `run` consumes the next scripted turn. A `{ text }` op is
 * streamed in chunks (optionally with a per-chunk delay so interruption tests
 * can cut in mid-utterance); a `{ call }` op emits the function-call frames
 * followed by its result.
 */
export class ScriptedHandler implements LLMHandler {
  private readonly script: Script;
  private turnIndex = -1;
  private readonly heard: Array<{ turnId: string; text: string; interrupted: boolean }> = [];

  constructor(script: Script) {
    this.script = script;
  }

  /** Heard-provenance records collected from `turn.completed` messages. */
  get heardLog(): ReadonlyArray<{ turnId: string; text: string; interrupted: boolean }> {
    return this.heard;
  }

  async run(ctx: TurnContext, emit: TurnEmitter): Promise<void> {
    this.turnIndex += 1;
    const turn = this.script[this.turnIndex];
    if (!turn) {
      throw new Error(
        `scripted handler exhausted: no response for turn #${this.turnIndex}`,
      );
    }

    let completionChars = 0;
    for (const op of turn) {
      if (ctx.signal.aborted) throw new TurnAbortedError();

      if ("text" in op) {
        const pieces = chunkText(op.text, op.chunks ?? 2);
        for (const piece of pieces) {
          if (ctx.signal.aborted) throw new TurnAbortedError();
          if (op.delayMs) await sleep(op.delayMs);
          if (ctx.signal.aborted) throw new TurnAbortedError();
          emit.text(piece);
          completionChars += piece.length;
        }
      } else {
        const call: FunctionCall = {
          tool_call_id: `${ctx.invocationId}-call-${this.turnIndex}-${op.call.name}`,
          function_name: op.call.name,
          arguments: op.call.args ?? {},
        };
        emit.functionCall(call);
        emit.functionResult(call, op.call.result ?? { status: "ok" });
      }
    }

    emit.usage({
      prompt_tokens: ctx.text.length,
      completion_tokens: completionChars,
      total_tokens: ctx.text.length + completionChars,
    });
  }

  onTurnCompleted(turnId: string, text: string, interrupted: boolean): void {
    this.heard.push({ turnId, text, interrupted });
    // Emitted on stdout so end-to-end tests can assert the [HEARD] round-trip.
    console.log(
      `HEARD turn_id=${turnId} interrupted=${interrupted} text=${JSON.stringify(text)}`,
    );
  }
}
