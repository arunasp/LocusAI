'use strict';
// Automated test suite for ../index.js.
//
// Same approach as the suite it replaces: fake the *dependency*, run
// the *real* shipped index.js (copied in byte-for-byte, never
// duplicated as inline text) in a real child process. What changed is
// the dependency -- the extension no longer wraps mcp-remote, it drives
// @modelcontextprotocol/sdk's transports itself, so that is what gets
// faked here.
//
// These are mechanics/wiring tests: nothing talks to localhost:1443 and
// nothing needs Docker, so they run anywhere Node runs. The point is to
// cover the behaviour the rewrite exists for -- recovering from a stale
// session -- which is otherwise only observable by rebuilding a
// container and watching what happens.
//
// What this deliberately does NOT verify: that the real SDK behaves
// like these stand-ins, or that Claude Desktop can launch the packed
// extension. Both need the real machine -- see tools/pipeline.sh's
// `server` stage and tools/README.md.
const test = require('node:test');
const assert = require('node:assert/strict');
const fs = require('node:fs');
const os = require('node:os');
const path = require('node:path');
const { spawnSync } = require('node:child_process');

const INDEX_JS = path.join(__dirname, '..', 'index.js');

// A fake @modelcontextprotocol/sdk exposing only what index.js imports.
// `clientSend` is the body of the client transport's send(), supplied
// per test so each can drive a different outcome; `serverScript` is
// what the Desktop-facing transport feeds in after start().
function makeFakeSdk(nodeModulesDir, { clientSend, serverScript }) {
  const pkgDir = path.join(nodeModulesDir, '@modelcontextprotocol', 'sdk');
  fs.mkdirSync(path.join(pkgDir, 'server'), { recursive: true });
  fs.mkdirSync(path.join(pkgDir, 'client'), { recursive: true });

  fs.writeFileSync(
    path.join(pkgDir, 'package.json'),
    JSON.stringify({ name: '@modelcontextprotocol/sdk', version: '0.0.0-test' }, null, 2),
  );

  fs.writeFileSync(
    path.join(pkgDir, 'server', 'stdio.js'),
    `'use strict';
class StdioServerTransport {
  async start() {
    ${serverScript}
  }
  async send(message) {
    process.stdout.write('TO_DESKTOP ' + JSON.stringify(message) + '\\n');
  }
  async close() {}
}
module.exports = { StdioServerTransport };
`,
  );

  fs.writeFileSync(
    path.join(pkgDir, 'client', 'streamableHttp.js'),
    `'use strict';
class StreamableHTTPError extends Error {
  constructor(code, message) {
    super(message);
    this.code = code;
  }
}
globalThis.__clientInstances = 0;
class StreamableHTTPClientTransport {
  constructor(url) {
    this.url = url;
    globalThis.__clientInstances += 1;
    this.instance = globalThis.__clientInstances;
  }
  async start() {}
  async send(message) {
    ${clientSend}
  }
  async close() {}
}
module.exports = { StreamableHTTPClientTransport, StreamableHTTPError };
`,
  );
}

function makeWorkDir(prefix) {
  const workDir = fs.mkdtempSync(path.join(os.tmpdir(), prefix));
  fs.copyFileSync(INDEX_JS, path.join(workDir, 'index.js'));
  const appDataDir = path.join(workDir, 'appdata');
  fs.mkdirSync(path.join(appDataDir, 'Claude', 'logs'), { recursive: true });
  return { workDir, appDataDir };
}

function runIndex(workDir, appDataDir) {
  return spawnSync(process.execPath, [path.join(workDir, 'index.js')], {
    cwd: workDir,
    env: { ...process.env, APPDATA: appDataDir },
    input: '',
    encoding: 'utf8',
    timeout: 10_000,
  });
}

function readDebugLog(appDataDir) {
  return fs.readFileSync(
    path.join(appDataDir, 'Claude', 'logs', 'locusai-local-bash-debug.log'),
    'utf8',
  );
}

test('starts both transports and relays a request to the server', () => {
  const { workDir, appDataDir } = makeWorkDir('locusai-index-test-start-');
  makeFakeSdk(path.join(workDir, 'node_modules'), {
    clientSend: `
      process.stdout.write('TO_SERVER ' + JSON.stringify(message) + '\\n');
      if (message.id === 1) process.exit(0);
    `,
    serverScript: `
      setImmediate(() => {
        this.onmessage({ jsonrpc: '2.0', id: 0, method: 'initialize' });
        this.onmessage({ jsonrpc: '2.0', method: 'notifications/initialized' });
        this.onmessage({ jsonrpc: '2.0', id: 1, method: 'tools/call' });
      });
    `,
  });

  const result = runIndex(workDir, appDataDir);

  assert.equal(result.status, 0, `expected exit 0, got ${result.status}\nstderr:\n${result.stderr}`);
  assert.match(result.stdout, /TO_SERVER .*"method":"tools\/call"/, 'a Desktop request must reach the server transport');

  const log = readDebugLog(appDataDir);
  assert.match(log, /--- launch \(sdk-direct\) ---/);
  assert.match(log, /client transport started/);
  assert.match(log, /server transport started/);
});

test('a stale session reconnects, replays the handshake, and retries the request', () => {
  const { workDir, appDataDir } = makeWorkDir('locusai-index-test-stale-');
  makeFakeSdk(path.join(workDir, 'node_modules'), {
    // First transport rejects the real request the way a server that has
    // forgotten the session does: HTTP 404 surfaced through onerror.
    // The replacement transport records what it receives and exits once
    // the retried request arrives, which is the behaviour under test.
    clientSend: `
      process.stdout.write('TO_SERVER#' + this.instance + ' ' + JSON.stringify(message) + '\\n');
      if (this.instance === 1 && message.id === 1) {
        const { StreamableHTTPError } = module.exports;
        setImmediate(() => this.onerror(new StreamableHTTPError(404, 'Session not found')));
        return;
      }
      if (this.instance === 2 && message.id === 1) {
        setImmediate(() => process.exit(0));
      }
    `,
    serverScript: `
      setImmediate(() => {
        this.onmessage({ jsonrpc: '2.0', id: 0, method: 'initialize' });
        this.onmessage({ jsonrpc: '2.0', method: 'notifications/initialized' });
        this.onmessage({ jsonrpc: '2.0', id: 1, method: 'tools/call' });
      });
    `,
  });

  const result = runIndex(workDir, appDataDir);

  assert.equal(result.status, 0, `expected exit 0, got ${result.status}\nstderr:\n${result.stderr}`);

  const log = readDebugLog(appDataDir);
  assert.match(log, /detected stale-session signal -- reconnecting/, 'a 404 must be recognised as staleness');
  assert.match(
    log,
    /replaying initialize on fresh client transport/,
    'the fresh transport must re-establish its session BEFORE anything else, or the retry fails with "Missing session ID"',
  );
  assert.match(log, /retrying request id=1 on fresh client transport/);

  // The retry must land on the SECOND transport, not the dead first one.
  assert.match(result.stdout, /TO_SERVER#2 .*"method":"tools\/call"/);
  // And the handshake must be replayed on it before that retry.
  const order = result.stdout.split('\n').filter((l) => l.startsWith('TO_SERVER#2'));
  assert.match(order[0] || '', /"method":"initialize"/, 'initialize must be the first message on a fresh transport');
});

test('an unrecognised error does not trigger a reconnect', () => {
  const { workDir, appDataDir } = makeWorkDir('locusai-index-test-other-err-');
  makeFakeSdk(path.join(workDir, 'node_modules'), {
    clientSend: `
      process.stdout.write('TO_SERVER#' + this.instance + ' ' + JSON.stringify(message) + '\\n');
      if (this.instance === 1 && message.id === 1) {
        setImmediate(() => {
          this.onerror(new Error('some unrelated transport hiccup'));
          setImmediate(() => process.exit(0));
        });
      }
    `,
    serverScript: `
      setImmediate(() => {
        this.onmessage({ jsonrpc: '2.0', id: 0, method: 'initialize' });
        this.onmessage({ jsonrpc: '2.0', id: 1, method: 'tools/call' });
      });
    `,
  });

  const result = runIndex(workDir, appDataDir);

  assert.equal(result.status, 0, `expected exit 0, got ${result.status}\nstderr:\n${result.stderr}`);
  const log = readDebugLog(appDataDir);
  assert.match(log, /client transport error/, 'the error must still be logged');
  assert.doesNotMatch(
    log,
    /detected stale-session signal/,
    'only a stale-session signal may trigger a reconnect -- reconnecting on any error would mask real faults',
  );
  assert.doesNotMatch(result.stdout, /TO_SERVER#2/, 'no second transport should be created');
});

test('missing SDK dependency fails loudly with a stated reason, not a bare stack', () => {
  const { workDir, appDataDir } = makeWorkDir('locusai-index-test-missing-');
  // Deliberately no node_modules at all.

  const result = runIndex(workDir, appDataDir);

  assert.equal(result.status, 1, `expected exit 1, got ${result.status}`);
  assert.match(result.stderr, /Failed to load bundled @modelcontextprotocol\/sdk dependency/);
});
