'use strict';
// Automated test suite for ../index.js -- formalizes the ad hoc
// mechanics check performed by hand while building the in-process
// (no-subprocess) architecture: a fake local mcp-remote package that
// mimics the real one's shape ("type": "module", top-level
// process.argv.slice(2) read, console.error-based logging), run
// against the REAL, shipped index.js (copied in byte-for-byte, never
// duplicated as inline text) via a real child process. This is a
// mechanics/wiring test, not a network test -- it does not talk to
// localhost:1443 or require Docker, so it runs anywhere Node runs
// (the sandbox, CI, this machine), matching this project's
// dependency-mock-verification approach: fake the *dependency*, run
// the *real* code under test.
//
// What this deliberately does NOT verify: that the real, compiled
// mcp-remote package (node_modules/mcp-remote/dist/proxy.js) behaves
// the same way as this test's fake stand-in, or that a real Claude
// Desktop install can actually launch and talk to this file. Both of
// those need the real machine -- see tools/pipeline.sh's `server`/
// `verify` stages and tools/README.md's "What was verified" section.
const test = require('node:test');
const assert = require('node:assert/strict');
const fs = require('node:fs');
const os = require('node:os');
const path = require('node:path');
const { spawnSync } = require('node:child_process');

const INDEX_JS = path.join(__dirname, '..', 'index.js');

/**
 * Builds a fake mcp-remote package under `nodeModulesDir/mcp-remote`,
 * matching the real package's shape closely enough for index.js's own
 * resolveMcpRemoteEntry()/import() logic to exercise for real:
 * "type": "module" (so index.js's dynamic import() path is genuinely
 * taken, not require()), a "bin" field pointing at dist/proxy.js, and
 * a compiled entry whose body is supplied by the caller so each test
 * can simulate a different mcp-remote outcome.
 */
function makeFakeMcpRemote(nodeModulesDir, proxyBody) {
  const pkgDir = path.join(nodeModulesDir, 'mcp-remote');
  const distDir = path.join(pkgDir, 'dist');
  fs.mkdirSync(distDir, { recursive: true });
  fs.writeFileSync(
    path.join(pkgDir, 'package.json'),
    JSON.stringify(
      {
        name: 'mcp-remote',
        version: '0.0.0-test',
        type: 'module',
        bin: { 'mcp-remote': 'dist/proxy.js' },
      },
      null,
      2,
    ),
  );
  fs.writeFileSync(path.join(distDir, 'proxy.js'), proxyBody);
}

/**
 * Sets up an isolated working directory containing a copy of the real
 * index.js (so require.resolve('mcp-remote/...') inside it walks up
 * from THIS directory, not the real repo's node_modules) plus an
 * APPDATA/Claude/logs directory for debugLog() to write into.
 */
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

test('resolves mcp-remote, threads process.argv, tees console.error, exits 0 on success', () => {
  const { workDir, appDataDir } = makeWorkDir('locusai-index-test-ok-');
  makeFakeMcpRemote(
    path.join(workDir, 'node_modules'),
    [
      "console.error('argv:', JSON.stringify(process.argv.slice(2)));",
      "process.stdout.write('READY\\n');",
    ].join('\n'),
  );

  const result = runIndex(workDir, appDataDir);

  assert.equal(result.status, 0, `expected exit 0, got ${result.status}\nstderr:\n${result.stderr}`);
  assert.match(result.stdout, /READY/, 'the fake module\'s stdout must reach the real process.stdout unmodified');
  assert.match(
    result.stderr,
    /argv: \["http:\/\/localhost:1443\/mcp","--allow-http"\]/,
    'process.argv must be set to [.., serverUrl, --allow-http] before the dynamic import() resolves',
  );

  const debugLog = fs.readFileSync(
    path.join(appDataDir, 'Claude', 'logs', 'locusai-local-bash-debug.log'),
    'utf8',
  );
  assert.match(debugLog, /--- launch ---/);
  assert.match(debugLog, /resolved mcp-remote entry:/);
  assert.match(
    debugLog,
    /\[mcp-remote\] argv:/,
    'the imported module\'s own console.error calls must be tee\'d into debugLog()',
  );
});

test('missing mcp-remote dependency fails loudly (exit 1, logged reason) instead of hanging', () => {
  const { workDir, appDataDir } = makeWorkDir('locusai-index-test-missing-');
  // Deliberately no node_modules/mcp-remote here.

  const result = runIndex(workDir, appDataDir);

  assert.equal(result.status, 1, `expected exit 1, got ${result.status}`);
  assert.match(result.stderr, /Failed to resolve bundled mcp-remote dependency/);
});

test('an mcp-remote entry that throws during import is caught and exits 1, not an uncaught crash', () => {
  const { workDir, appDataDir } = makeWorkDir('locusai-index-test-throw-');
  makeFakeMcpRemote(path.join(workDir, 'node_modules'), "throw new Error('boom from fake mcp-remote');\n");

  const result = runIndex(workDir, appDataDir);

  assert.equal(result.status, 1, `expected exit 1, got ${result.status}`);
  assert.match(result.stderr, /Failed to run bundled mcp-remote dependency/);
});
