/**
 * Wire protocol for the Pipecat <-> external LLM WebSocket bridge.
 *
 * This is the TypeScript mirror of `src/pipecat_adk/websocket/protocol.py`.
 * Keep the two in sync — they are the two halves of one contract.
 *
 * Messages are UTF-8 JSON text frames. Every message is an envelope with a
 * `v` (protocol version) and a `type`.
 */

export const PROTOCOL_VERSION = 1;

/** Message types. */
export const MSG = {
  // Python -> JS
  SESSION_START: "session.start",
  TURN_RUN: "turn.run",
  TURN_CANCEL: "turn.cancel",
  TURN_COMPLETED: "turn.completed",
  SESSION_END: "session.end",
  PING: "ping",
  // JS -> Python
  SESSION_READY: "session.ready",
  FRAME: "frame",
  TURN_USAGE: "turn.usage",
  STATE_DELTA: "state.delta",
  ERROR: "error",
  PONG: "pong",
} as const;

/** Frame names — the `frame` field of an MSG.FRAME envelope (JS -> Python). */
export const FRAME = {
  RESPONSE_START: "VqlLLMFullResponseStartFrame",
  TEXT: "VqlLLMTextFrame",
  FUNCTION_CALLS_STARTED: "VqlFunctionCallsStartedFrame",
  FUNCTION_CALL_IN_PROGRESS: "VqlFunctionCallInProgressFrame",
  FUNCTION_CALL_RESULT: "VqlFunctionCallResultFrame",
  RESPONSE_END: "VqlLLMFullResponseEndFrame",
} as const;

// ---------------------------------------------------------------------------
// Inbound message shapes (Python -> JS)
// ---------------------------------------------------------------------------

export interface SessionStartMessage {
  v: number;
  type: typeof MSG.SESSION_START;
  session: Record<string, unknown>;
  metadata: Record<string, unknown>;
}

export interface TurnRunMessage {
  v: number;
  type: typeof MSG.TURN_RUN;
  turn_id: string;
  payload: { text?: string } & Record<string, unknown>;
  state_delta: Record<string, unknown> | null;
}

export interface TurnCancelMessage {
  v: number;
  type: typeof MSG.TURN_CANCEL;
  turn_id: string;
}

export interface TurnCompletedMessage {
  v: number;
  type: typeof MSG.TURN_COMPLETED;
  turn_id: string;
  text: string;
  interrupted: boolean;
}

export interface SessionEndMessage {
  v: number;
  type: typeof MSG.SESSION_END;
}

export interface PingMessage {
  v: number;
  type: typeof MSG.PING;
  ts: number;
}

export type InboundMessage =
  | SessionStartMessage
  | TurnRunMessage
  | TurnCancelMessage
  | TurnCompletedMessage
  | SessionEndMessage
  | PingMessage;

// ---------------------------------------------------------------------------
// Outbound payloads (JS -> Python)
// ---------------------------------------------------------------------------

export interface FunctionCall {
  tool_call_id: string;
  function_name: string;
  arguments: Record<string, unknown>;
}

export interface TokenUsage {
  prompt_tokens: number;
  completion_tokens: number;
  total_tokens: number;
  cache_read_input_tokens: number;
  reasoning_tokens: number;
}

export class ProtocolError extends Error {}

/** Parse and validate a raw WebSocket message into an inbound envelope. */
export function decodeMessage(raw: string | Uint8Array): InboundMessage {
  const text =
    typeof raw === "string" ? raw : new TextDecoder().decode(raw);

  let envelope: unknown;
  try {
    envelope = JSON.parse(text);
  } catch (e) {
    throw new ProtocolError(`message is not valid JSON: ${(e as Error).message}`);
  }
  if (typeof envelope !== "object" || envelope === null) {
    throw new ProtocolError("message must be a JSON object");
  }
  const env = envelope as Record<string, unknown>;
  if (env.v !== PROTOCOL_VERSION) {
    throw new ProtocolError(
      `unsupported protocol version ${String(env.v)} (expected ${PROTOCOL_VERSION})`,
    );
  }
  if (typeof env.type !== "string" || env.type.length === 0) {
    throw new ProtocolError("message is missing 'type'");
  }
  return env as unknown as InboundMessage;
}

// ---------------------------------------------------------------------------
// Encoders (JS -> Python)
// ---------------------------------------------------------------------------

function encode(obj: Record<string, unknown>): string {
  return JSON.stringify({ v: PROTOCOL_VERSION, ...obj });
}

export function encodeSessionReady(info: Record<string, unknown> = {}): string {
  return encode({ type: MSG.SESSION_READY, info });
}

export function encodePong(ts: number): string {
  return encode({ type: MSG.PONG, ts });
}

function encodeFrame(
  frame: string,
  turnId: string,
  data: Record<string, unknown>,
): string {
  return encode({ type: MSG.FRAME, frame, turn_id: turnId, data });
}

export function encodeResponseStart(turnId: string, invocationId: string): string {
  return encodeFrame(FRAME.RESPONSE_START, turnId, { invocation_id: invocationId });
}

export function encodeResponseEnd(turnId: string, invocationId: string): string {
  return encodeFrame(FRAME.RESPONSE_END, turnId, { invocation_id: invocationId });
}

export function encodeText(
  turnId: string,
  invocationId: string,
  text: string,
): string {
  return encodeFrame(FRAME.TEXT, turnId, { invocation_id: invocationId, text });
}

export function encodeFunctionCallsStarted(
  turnId: string,
  invocationId: string,
  calls: FunctionCall[],
): string {
  return encodeFrame(FRAME.FUNCTION_CALLS_STARTED, turnId, {
    invocation_id: invocationId,
    function_calls: calls,
  });
}

export function encodeFunctionCallInProgress(
  turnId: string,
  invocationId: string,
  call: FunctionCall,
): string {
  return encodeFrame(FRAME.FUNCTION_CALL_IN_PROGRESS, turnId, {
    invocation_id: invocationId,
    tool_call_id: call.tool_call_id,
    function_name: call.function_name,
    arguments: call.arguments,
  });
}

export function encodeFunctionCallResult(
  turnId: string,
  invocationId: string,
  call: FunctionCall,
  result: unknown,
): string {
  return encodeFrame(FRAME.FUNCTION_CALL_RESULT, turnId, {
    invocation_id: invocationId,
    tool_call_id: call.tool_call_id,
    function_name: call.function_name,
    result,
  });
}

export function encodeTurnUsage(
  turnId: string,
  usage: Partial<TokenUsage>,
): string {
  return encode({ type: MSG.TURN_USAGE, turn_id: turnId, data: usage });
}

export function encodeStateDelta(
  turnId: string,
  delta: Record<string, unknown>,
): string {
  return encode({ type: MSG.STATE_DELTA, turn_id: turnId, data: delta });
}

export function encodeError(
  message: string,
  opts: { turnId?: string; fatal?: boolean } = {},
): string {
  return encode({
    type: MSG.ERROR,
    turn_id: opts.turnId ?? null,
    data: { message, fatal: opts.fatal ?? false },
  });
}
