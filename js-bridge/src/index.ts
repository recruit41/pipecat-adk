/**
 * Bun entrypoint for the Pipecat LLM bridge.
 *
 * Starts a WebSocket server that `WebSocketLLMService` (the Python side)
 * connects to. Each connection gets its own `BridgeConnection` and a fresh
 * `LLMHandler`.
 *
 * Configuration (environment variables):
 *   PORT           TCP port to listen on (default: 8787).
 *   HOST           Interface to bind (default: 127.0.0.1).
 *   BRIDGE_SCRIPT  Optional JSON script -> deterministic `ScriptedHandler`.
 *                  When unset, the trivial `EchoHandler` is used.
 *
 * Run:  bun run src/index.ts
 */

import { BridgeConnection } from "./bridge";
import {
  EchoHandler,
  ScriptedHandler,
  normaliseScript,
  type LLMHandler,
} from "./handler";

interface SocketData {
  conn: BridgeConnection | null;
}

function createHandler(): LLMHandler {
  const script = process.env.BRIDGE_SCRIPT;
  if (script && script.trim().length > 0) {
    return new ScriptedHandler(normaliseScript(JSON.parse(script)));
  }
  return new EchoHandler();
}

const port = Number(process.env.PORT ?? "8787");
const hostname = process.env.HOST ?? "127.0.0.1";

const server = Bun.serve<SocketData>({
  port,
  hostname,
  fetch(req, srv): Response | undefined {
    if (srv.upgrade(req, { data: { conn: null } })) {
      return undefined;
    }
    return new Response("pipecat LLM bridge — connect via WebSocket\n", {
      status: 426,
    });
  },
  websocket: {
    open(ws): void {
      ws.data.conn = new BridgeConnection(createHandler(), (data) => {
        ws.send(data);
      });
    },
    message(ws, message): void {
      void ws.data.conn?.handleMessage(message);
    },
    close(ws): void {
      ws.data.conn?.close();
      ws.data.conn = null;
    },
  },
});

// Stable, parseable readiness line — test harnesses wait for this on stdout.
console.log(`BRIDGE_LISTENING ${server.port}`);
