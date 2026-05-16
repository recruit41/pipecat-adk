/**
 * Minimal ambient declarations for the subset of the Bun runtime API used by
 * this bridge. This lets `tsc` type-check the project with no installed
 * dependencies. For full types, `bun add -d @types/bun` and delete this file.
 */

declare const process: {
  env: Record<string, string | undefined>;
};

declare namespace Bun {
  interface ServerWebSocket<T = undefined> {
    data: T;
    send(data: string | Uint8Array): number;
    close(code?: number, reason?: string): void;
    readonly readyState: number;
  }

  interface WebSocketHandler<T = undefined> {
    open?(ws: ServerWebSocket<T>): void | Promise<void>;
    message?(
      ws: ServerWebSocket<T>,
      message: string | Uint8Array,
    ): void | Promise<void>;
    close?(
      ws: ServerWebSocket<T>,
      code: number,
      reason: string,
    ): void | Promise<void>;
    drain?(ws: ServerWebSocket<T>): void | Promise<void>;
  }

  interface Server {
    readonly port: number;
    readonly hostname: string;
    upgrade<T>(req: Request, options?: { data: T }): boolean;
    stop(closeActiveConnections?: boolean): void;
  }

  interface ServeOptions<T> {
    port?: number | string;
    hostname?: string;
    fetch(
      req: Request,
      server: Server,
    ): Response | undefined | Promise<Response | undefined>;
    websocket: WebSocketHandler<T>;
  }

  function serve<T = undefined>(options: ServeOptions<T>): Server;
}
