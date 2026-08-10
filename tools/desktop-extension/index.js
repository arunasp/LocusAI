#!/usr/bin/env node
// Entry point for the LocusAI Local Bash Desktop extension.
//
// Connects Desktop directly to the local-bash MCP service over
// Streamable HTTP using @modelcontextprotocol/sdk's own transports,
// in-process. There is no subprocess and no mcp-remote.
//
// This replaces an mcp-remote-based design. That design worked, but it
// could not recover from a stale session: mcp-remote owns both
// transports internally and its CLI entry exports no hooks, so nothing
// above it can observe a dead session, let alone reconnect. The failure
// is routine rather than exotic -- rebuilding the server container
// (tools/pipeline.sh server) recreates it, and every subsequent request
// then hangs until Desktop's own ~4-minute client timeout, presenting
// as an unexplained freeze. The only remedy was toggling the extension
// off and on by hand.
//
// Holding both transports here makes that recoverable, and the same
// approach is already proven in the cicd-runner extension.
const fs = require('fs');
const os = require('os');
const path = require('path');
// Loaded through require() rather than at the top of the file so a
// missing or broken dependency fails with a stated reason instead of a
// bare MODULE_NOT_FOUND stack. This is the only dependency, and the
// extension is useless without it, so failing loudly here is correct.
let StdioServerTransport;
let StreamableHTTPClientTransport;
let StreamableHTTPError;
try {
  ({ StdioServerTransport } = require('@modelcontextprotocol/sdk/server/stdio.js'));
  ({ StreamableHTTPClientTransport, StreamableHTTPError } = require('@modelcontextprotocol/sdk/client/streamableHttp.js'));
} catch (err) {
  console.error('Failed to load bundled @modelcontextprotocol/sdk dependency:', err);
  process.exit(1);
}

// Logs alongside Desktop's own MCP server logs: %APPDATA%\Claude\logs on
// Windows, per the official MCP debugging documentation. Falls back to
// the OS temp directory where APPDATA is unset.
const debugLogDir = process.env.APPDATA ? path.join(process.env.APPDATA, 'Claude', 'logs') : os.tmpdir();
const debugLogPath = path.join(debugLogDir, 'locusai-local-bash-debug.log');
function debugLog(line) {
  try {
    fs.appendFileSync(debugLogPath, `${new Date().toISOString()} ${line}\n`);
  } catch {
    // Debug logging must never be why the extension fails to start.
  }
}

debugLog('--- launch (sdk-direct) ---');
debugLog(`process.execPath=${process.execPath}`);
debugLog(`process.version=${process.version}`);
debugLog(`process.platform=${process.platform}`);
debugLog(`__dirname=${__dirname}`);

// The server mounts exactly one directory (the LocusAI checkout, as
// /workspace) and decides what may run there from its own allowlist, so
// this extension passes no directory configuration of any kind. That is
// a deliberate difference from the cicd-runner extension, which must
// forward a user-selected directory set because it is project-agnostic.
const SERVER_URL = new URL('http://localhost:1443/mcp');

// Deliberately shorter than Desktop's own ~4-minute client-side timeout,
// so a lost connection is detected and acted on here rather than
// surfacing to the user as a bare hang. Well above realistic call
// durations: a stale session fails via onerror almost immediately, so
// this fallback only covers the case where no error arrives at all.
const REQUEST_TIMEOUT_MS = 90_000;
const TIMEOUT_CHECK_INTERVAL_MS = 5_000;

// A request against a session the server no longer recognises fails
// with a real HTTP 404, surfaced by the SDK as StreamableHTTPError with
// a structured .code. Checking the code is precise; the message text is
// kept only as a fallback for a differently-shaped error carrying the
// same meaning.
function isStaleSessionError(err) {
  if (!err) return false;
  if (err instanceof StreamableHTTPError && err.code === 404) return true;
  return /Session not found/i.test(err.message || '');
}

const serverTransport = new StdioServerTransport();

let clientTransport = null;
const pending = new Map(); // id -> { message, originalSentAt }
let reconnecting = false;

// A fresh transport must complete its own initialize handshake before it
// will accept anything else -- it has no session ID until then, and any
// other message fails with "Missing session ID". Desktop's initialize
// and the notifications/initialized that follows are remembered so a new
// transport can replay them before retrying anything.
let lastInitializeMessage = null;
let lastInitializedNotification = null;

function createClientTransport() {
  return new StreamableHTTPClientTransport(SERVER_URL);
}

function sendErrorResponse(id, message) {
  debugLog(`sending final error response for id=${id}: ${message}`);
  serverTransport.send({
    jsonrpc: '2.0',
    id,
    error: { code: -32000, message },
  }).catch((err) => debugLog(`failed to send error response: ${err && err.stack ? err.stack : err}`));
}

function wireClientTransport(transport) {
  transport.onmessage = (message) => {
    if (message.id !== undefined && pending.has(message.id)) {
      pending.delete(message.id);
    }
    serverTransport.send(message).catch((err) => {
      debugLog(`error sending to Desktop: ${err && err.stack ? err.stack : err}`);
    });
  };

  transport.onerror = (err) => {
    debugLog(`client transport error: ${err && err.stack ? err.stack : err}`);
    if (isStaleSessionError(err)) {
      debugLog('detected stale-session signal -- reconnecting (client transport only)');
      handleStaleConnection();
    }
  };

  transport.onclose = () => {
    debugLog('client transport closed');
  };
}

// Budget-based rather than attempt-count-based: a request may legitimately
// need more than one retry inside its own window, for instance if the
// container is rebuilt twice in quick succession. Each request's
// originalSentAt is set once, at first send, and never touched again, so
// only a request that has genuinely exhausted its own budget is failed.
function handleStaleConnection() {
  if (reconnecting) return; // already in progress; avoid a duplicate teardown
  reconnecting = true;
  debugLog('handling stale connection');

  const old = clientTransport;
  if (old) {
    old.onmessage = undefined;
    old.onerror = undefined;
    old.onclose = undefined;
    old.close().catch(() => {});
  }

  const now = Date.now();
  const toRetry = [];
  for (const [id, info] of pending.entries()) {
    // Handshake messages are replayed explicitly below, never queued
    // into the ordinary retry list.
    if (info.message.method === 'initialize' || info.message.method === 'notifications/initialized') {
      continue;
    }
    if (now - info.originalSentAt >= REQUEST_TIMEOUT_MS) {
      pending.delete(id);
      sendErrorResponse(id, 'LocusAI local-bash connection was lost and could not be recovered in time');
    } else {
      toRetry.push([id, info.message]);
    }
  }

  clientTransport = createClientTransport();
  wireClientTransport(clientTransport);
  clientTransport.start()
    .then(async () => {
      if (lastInitializeMessage) {
        debugLog('replaying initialize on fresh client transport');
        await clientTransport.send(lastInitializeMessage);
      }
      if (lastInitializedNotification) {
        debugLog('replaying notifications/initialized on fresh client transport');
        await clientTransport.send(lastInitializedNotification);
      }

      reconnecting = false;
      for (const [id, message] of toRetry) {
        debugLog(`retrying request id=${id} on fresh client transport`);
        clientTransport.send(message).catch((err) => {
          debugLog(`error retrying request id=${id}: ${err && err.stack ? err.stack : err}`);
        });
      }
    })
    .catch((err) => {
      reconnecting = false;
      debugLog(`failed to start/re-initialize fresh client transport: ${err && err.stack ? err.stack : err}`);
    });
}

function checkPendingTimeouts() {
  if (reconnecting) return;
  const now = Date.now();
  for (const [, info] of pending.entries()) {
    if (now - info.originalSentAt > REQUEST_TIMEOUT_MS) {
      debugLog(`request pending >${REQUEST_TIMEOUT_MS}ms since ORIGINAL send with no response -- treating as stale (timeout fallback)`);
      handleStaleConnection();
      break; // handleStaleConnection reconnects; the rest are covered next tick
    }
  }
}

serverTransport.onmessage = (message) => {
  if (message.method === 'initialize') {
    lastInitializeMessage = message;
  } else if (message.method === 'notifications/initialized') {
    lastInitializedNotification = message;
  }
  if (message.id !== undefined) {
    pending.set(message.id, { message, originalSentAt: Date.now() });
  }
  clientTransport.send(message).catch((err) => {
    debugLog(`error sending to local-bash: ${err && err.stack ? err.stack : err}`);
  });
};

serverTransport.onerror = (err) => {
  debugLog(`server transport (Desktop-facing) error: ${err && err.stack ? err.stack : err}`);
};

async function main() {
  clientTransport = createClientTransport();
  wireClientTransport(clientTransport);
  await clientTransport.start();
  debugLog('client transport started');

  await serverTransport.start();
  debugLog('server transport started -- relaying stdio to local-bash directly, no subprocess');

  setInterval(checkPendingTimeouts, TIMEOUT_CHECK_INTERVAL_MS);
}

main().catch((err) => {
  debugLog(`fatal error during startup: ${err && err.stack ? err.stack : err}`);
  console.error('Failed to start LocusAI Local Bash extension:', err);
  process.exit(1);
});

function cleanup() {
  debugLog('shutting down');
  serverTransport.close().catch(() => {});
  if (clientTransport) clientTransport.close().catch(() => {});
  process.exit(0);
}
process.on('SIGINT', cleanup);
process.on('SIGTERM', cleanup);
