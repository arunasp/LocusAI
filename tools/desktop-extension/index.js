#!/usr/bin/env node
// Entry point for the LocusAI Local Bash Desktop extension.
//
// Runs mcp-remote's own compiled CLI entry (bundled as a real dependency
// of this extension, see package.json) IN-PROCESS via a dynamic
// import(), rather than spawning it as a child process. This mirrors the
// official reference "Filesystem" MCP extension
// (github.com/modelcontextprotocol/servers, src/filesystem/index.ts),
// which Desktop is confirmed to bundle verbatim: its entire stdio setup
// is two lines -- `new StdioServerTransport()` + `server.connect(transport)`
// -- operating directly on the real process.stdin/process.stdout that
// Desktop's own Electron utilityProcess.fork() is driving. No subprocess,
// no stdio relay, no second process whose own stdin lifecycle could
// diverge from the first.
//
// This supersedes the previous two-hop spawn-and-relay design (spawn() +
// manual stdio piping + a findSystemNode() workaround for the
// ELECTRON_RUN_AS_NODE/runAsNode-fuse problem). Both problems that design
// was fighting are structural consequences of spawning a second process
// at all:
//   - ELECTRON_RUN_AS_NODE / runAsNode fuse: only matters when spawning
//     Claude.exe itself as if it were node.exe. Running in-process means
//     there is no second `node` invocation to launch, so the fuse being
//     disabled (as it apparently is in this packaged build) is now
//     irrelevant.
//   - mcp-remote shutting itself down ~1ms after connecting: confirmed
//     (via the user's own main.log) to be mcp-remote's own
//     process.stdin.on('end', ...) handler reacting to something about
//     how UtilityProcess's stdin behaved when relayed through a second
//     spawned process's piped stdin. Running mcp-remote directly against
//     the real process.stdin removes the relay entirely -- there is
//     nothing left to diverge from the Filesystem extension's own proven
//     stdin handling, because it is now the exact same stdin object.
const fs = require('fs');
const os = require('os');
const path = require('path');
const { pathToFileURL } = require('url');

// Logs to the same directory Desktop itself already uses for MCP server
// logs -- confirmed against the official MCP debugging docs
// (modelcontextprotocol.io/docs/tools/debugging): Windows is
// `%APPDATA%\Claude\logs`. Falls back to the OS temp dir when `APPDATA`
// isn't set (Linux -- manifest.json declares that as a compatible
// platform, but Claude Desktop itself only ships for macOS/Windows per
// the same docs, so `APPDATA` is never expected there).
const debugLogDir = process.env.APPDATA ? path.join(process.env.APPDATA, 'Claude', 'logs') : os.tmpdir();
const debugLogPath = path.join(debugLogDir, 'locusai-local-bash-debug.log');
function debugLog(line) {
  try {
    fs.appendFileSync(debugLogPath, `${new Date().toISOString()} ${line}\n`);
  } catch {
    // Debug logging must never be why the extension fails to start.
  }
}

// Resolves mcp-remote's actual compiled entry script from its own
// package.json `bin` field, rather than hardcoding a path like
// "dist/proxy.js" -- that field is what mcp-remote itself publishes as
// its true bin target, so this stays correct across mcp-remote version
// bumps.
function resolveMcpRemoteEntry() {
  const pkgJsonPath = require.resolve('mcp-remote/package.json');
  const pkgDir = path.dirname(pkgJsonPath);
  const pkg = JSON.parse(fs.readFileSync(pkgJsonPath, 'utf8'));
  const bin = typeof pkg.bin === 'string' ? pkg.bin : pkg.bin && pkg.bin['mcp-remote'];
  if (!bin) {
    throw new Error('mcp-remote package.json has no usable "bin" entry');
  }
  return path.join(pkgDir, bin);
}

debugLog('--- launch ---');
debugLog(`process.execPath=${process.execPath}`);
debugLog(`process.version=${process.version}`);
debugLog(`process.platform=${process.platform}`);
debugLog(`process.cwd()=${process.cwd()}`);
debugLog(`__dirname=${__dirname}`);

let mcpRemoteEntry;
try {
  mcpRemoteEntry = resolveMcpRemoteEntry();
  debugLog(`resolved mcp-remote entry: ${mcpRemoteEntry}`);
} catch (err) {
  debugLog(`failed to resolve mcp-remote entry: ${err && err.stack ? err.stack : err}`);
  console.error('Failed to resolve bundled mcp-remote dependency:', err);
  process.exit(1);
}

// Tee console.error into debugLog before importing mcp-remote, since its
// own log()/debugLog() helpers (src/lib/utils.ts) write via
// console.error -- this keeps every line mcp-remote logs visible in our
// own debug file too, alongside Desktop's normal stderr capture (which
// still happens unmodified, since the original console.error is still
// called).
const originalConsoleError = console.error.bind(console);
console.error = (...args) => {
  try {
    debugLog(`[mcp-remote] ${args.map((a) => (typeof a === 'string' ? a : JSON.stringify(a))).join(' ')}`);
  } catch {
    // Never let debug logging break the actual log call.
  }
  originalConsoleError(...args);
};

// mcp-remote's compiled CLI (dist/proxy.js, from src/proxy.ts) reads its
// arguments from `process.argv.slice(2)` at module-evaluation time (top
// level, via parseCommandLineArgs(...)) -- so process.argv must be set to
// what it expects *before* the dynamic import() below resolves and runs
// that top-level code. Indices 0 and 1 are never read (slice(2) discards
// them), so their exact values don't matter.
//
// --allow-http: the target is a plain (non-TLS) HTTP endpoint on
// localhost by design (see tools/server/bash_mcp_server.py) -- this tells
// mcp-remote that's intentional rather than something to warn about or
// refuse, per its own README ("trusted private networks").
process.argv = [process.execPath, mcpRemoteEntry, 'http://localhost:1443/mcp', '--allow-http'];
debugLog(`process.argv set to: ${JSON.stringify(process.argv)}`);

// mcp-remote is published as an ES module ("type": "module" in its own
// package.json), so it's loaded with a dynamic import() (which works
// from this CommonJS file without needing to convert this file itself to
// ESM) rather than require(). pathToFileURL() is required on Windows to
// turn the absolute filesystem path into a valid file:// URL string --
// import() does not accept raw Windows paths (backslashes / missing
// scheme) the way require() does.
//
// Once loaded, mcp-remote's own top-level code runs its
// StdioServerTransport against the real process.stdin/process.stdout of
// *this* process -- the same process Desktop's UtilityProcess launched
// and is directly driving -- exactly matching the Filesystem reference
// extension's single-hop pattern. There is no child process and nothing
// left to relay.
debugLog('importing mcp-remote entry in-process...');
import(pathToFileURL(mcpRemoteEntry).href).catch((err) => {
  debugLog(`failed to import/run mcp-remote: ${err && err.stack ? err.stack : err}`);
  originalConsoleError('Failed to run bundled mcp-remote dependency:', err);
  process.exit(1);
});
