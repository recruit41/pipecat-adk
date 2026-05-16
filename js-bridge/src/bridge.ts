/**
 * Bridge connection logic — protocol handling, turn lifecycle, cancellation.
 *
 * `BridgeConnection` is transport-agnostic: it is handed a `send` callback and
 * a `LLMHandler`, and it owns everything else (decoding messages, the
 * response-start / response-end framing, abort handling, heartbeat replies).
 * `index.ts` wires it to a Bun WebSocket; tests can wire it to anything.
 */

import {
  type LLMHandler,
  type TurnContext,
  type TurnEmitter,
  TurnAbortedError,
} from "./handler";
import {
  type FunctionCall,
  type InboundMessage,
  type TurnRunMessage,
  MSG,
  ProtocolError,
  decodeMessage,
  encodeError,
  encodeFunctionCallInProgress,
  encodeFunctionCallResult,
  encodeFunctionCallsStarted,
  encodePong,
  encodeResponseEnd,
  encodeResponseStart,
  encodeSessionReady,
  encodeStateDelta,
  encodeText,
  encodeTurnUsage,
} from "./protocol";

/** Sends a UTF-8 text frame over the underlying transport. */
export type SendFn = (data: string) => void;

function newInvocationId(): string {
  return `inv-${crypto.randomUUID()}`;
}

/**
 * One Pipecat <-> LLM session over a single WebSocket connection.
 *
 * A connection multiplexes turns: `turn.run` starts a turn, `turn.cancel`
 * aborts it. At most one turn is active at a time in normal pipeline use, but
 * the implementation keeps a map so a late cancel can never hit the wrong turn.
 */
export class BridgeConnection {
  private readonly handler: LLMHandler;
  private readonly send: SendFn;
  private readonly turns = new Map<string, AbortController>();
  private closed = false;

  constructor(handler: LLMHandler, send: SendFn) {
    this.handler = handler;
    this.send = send;
  }

  /** Number of turns currently in flight (used by tests). */
  get activeTurns(): number {
    return this.turns.size;
  }

  /** Decode and dispatch one inbound WebSocket message. */
  async handleMessage(raw: string | Uint8Array): Promise<void> {
    if (this.closed) return;

    let msg: InboundMessage;
    try {
      msg = decodeMessage(raw);
    } catch (e) {
      const reason = e instanceof ProtocolError ? e.message : String(e);
      this.safeSend(encodeError(`malformed message: ${reason}`));
      return;
    }

    switch (msg.type) {
      case MSG.SESSION_START:
        await this.handler.onSessionStart?.(msg.session, msg.metadata);
        this.safeSend(encodeSessionReady({ protocol: "ok" }));
        break;
      case MSG.TURN_RUN:
        // Not awaited: the message loop must stay responsive to turn.cancel.
        void this.runTurn(msg);
        break;
      case MSG.TURN_CANCEL:
        this.cancelTurn(msg.turn_id);
        break;
      case MSG.TURN_COMPLETED:
        await this.handler.onTurnCompleted?.(
          msg.turn_id,
          msg.text,
          msg.interrupted,
        );
        break;
      case MSG.SESSION_END:
        await this.handler.onSessionEnd?.();
        break;
      case MSG.PING:
        this.safeSend(encodePong(msg.ts));
        break;
      default:
        this.safeSend(
          encodeError(`unsupported message type: ${(msg as { type: string }).type}`),
        );
    }
  }

  /** Abort every in-flight turn and stop accepting messages. */
  close(): void {
    this.closed = true;
    for (const controller of this.turns.values()) controller.abort();
    this.turns.clear();
  }

  // -------------------------------------------------------------------------

  private cancelTurn(turnId: string): void {
    this.turns.get(turnId)?.abort();
  }

  private async runTurn(msg: TurnRunMessage): Promise<void> {
    const turnId = msg.turn_id;
    if (this.turns.has(turnId)) {
      this.safeSend(encodeError(`turn ${turnId} is already running`, { turnId }));
      return;
    }

    const controller = new AbortController();
    this.turns.set(turnId, controller);
    const invocationId = newInvocationId();
    const signal = controller.signal;

    // The bridge owns the start/end framing; the handler only emits content.
    this.safeSend(encodeResponseStart(turnId, invocationId));

    const emit = this.makeEmitter(turnId, invocationId, signal);
    const ctx: TurnContext = {
      turnId,
      invocationId,
      text: msg.payload?.text ?? "",
      stateDelta: msg.state_delta,
      signal,
    };

    try {
      await this.handler.run(ctx, emit);
      if (!signal.aborted) {
        this.safeSend(encodeResponseEnd(turnId, invocationId));
      }
    } catch (e) {
      if (signal.aborted || e instanceof TurnAbortedError) {
        // Interrupted: the Python side cancelled and is no longer listening.
        return;
      }
      this.safeSend(encodeError(`turn failed: ${String(e)}`, { turnId }));
    } finally {
      this.turns.delete(turnId);
    }
  }

  private makeEmitter(
    turnId: string,
    invocationId: string,
    signal: AbortSignal,
  ): TurnEmitter {
    const guard = (encode: () => string): void => {
      if (signal.aborted || this.closed) return;
      this.safeSend(encode());
    };
    return {
      text: (chunk) => guard(() => encodeText(turnId, invocationId, chunk)),
      functionCall: (call: FunctionCall) => {
        guard(() => encodeFunctionCallsStarted(turnId, invocationId, [call]));
        guard(() => encodeFunctionCallInProgress(turnId, invocationId, call));
      },
      functionResult: (call: FunctionCall, result: unknown) =>
        guard(() => encodeFunctionCallResult(turnId, invocationId, call, result)),
      stateDelta: (delta) => guard(() => encodeStateDelta(turnId, delta)),
      usage: (usage) => guard(() => encodeTurnUsage(turnId, usage)),
    };
  }

  private safeSend(data: string): void {
    if (this.closed) return;
    try {
      this.send(data);
    } catch {
      // Socket already gone — nothing we can do; the turn will be aborted on close.
    }
  }
}
