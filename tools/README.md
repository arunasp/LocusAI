# Local git/bash MCP tools

Two Claude skills exist for working on this repo, delivered as
`.skill` files during development for the user to save: the
project-specific `locusai-mcp-pipeline` (the repo layout, the WSL2
checkout path, and `tools/pipeline.sh`'s stages -- read this first
before running ad hoc lint/build/install commands here) and the
general-purpose `sandbox-staging-pipeline` (the underlying
sandbox-develops/real-device-stages discipline, applicable to any
project, not specific to this repo). If either is installed in a
session working on this repo, consult it before re-deriving a
procedure this README or the skill itself already documents.

## Architecture — the actual thing that was asked for

Claude Desktop (Windows) never executes anything cross-OS. The only
thing that crosses the Windows/WSL2 boundary is a plain HTTP request:

```
Claude Desktop (Windows)
  -> spawns tools/desktop-extension/index.js via Electron's
     utilityProcess.fork() (confirmed in Desktop's own main.log, not
     assumed -- native Windows, no WSL2 involved)
  -> index.js resolves its bundled mcp-remote dependency
     (node_modules/mcp-remote, via its own package.json "bin" field)
     and runs it IN-PROCESS with a dynamic import() -- no child
     process, no npx, no shell. mcp-remote's own StdioServerTransport
     binds directly to this process's real process.stdin/stdout, the
     same object Desktop's UtilityProcess is driving -- mirroring the
     official reference "Filesystem" extension's single-hop pattern
     (see "What was NOT verified" for why this replaced an earlier
     spawn-and-relay design)
  -> HTTP request to localhost:1443
  -> WSL2's automatic port forwarding (confirmed real: any port a
     container publishes inside WSL2 is reachable from Windows as
     localhost:<port>, zero extra config) delivers it to...
  -> the actual MCP server, running as a persistent Docker container
     inside WSL2, giving you the real tools (run_command, etc.)
```

mcp-remote used to be fetched via `npx -y mcp-remote ...` at every
launch instead of bundled. Changed because that violates MCPB's own
guidance to bundle all dependencies and work offline, and because it
depends on `npx` resolving via shell `PATH` -- something that can
differ between an interactive shell and a process Desktop spawns from
its own sandboxed Node runtime. See `index.js`'s own comments and
`package.json`.

The bundled dependency is loaded in-process via `import()`, not
spawned as a subprocess at all -- see "What was NOT verified" for the
real debug.log/main.log evidence (an `ELECTRON_RUN_AS_NODE`/runAsNode
fuse problem, then a stdin-relay problem) that led to dropping the
subprocess entirely in favor of this single-hop design.

**Verified end-to-end in a sandbox, not just designed:** the server
handled a real MCP `initialize` handshake over HTTP (200 OK, correct
JSON-RPC response). The pre-bundling version of `index.js` (the
actual bundled entry point, not just a bare `npx` command) was run
directly and logged "Connected to remote server using
StreamableHTTPClientTransport" / "Proxy established successfully
between local STDIO and remote StreamableHTTPClientTransport" --
genuinely working, in that sandbox. The packed `.mcpb` round-trips
identically through unpack. **Confirmed against a real Claude Desktop
install after the switch to the in-process (no-subprocess) design:**
the extension now shows as connected in Settings > Extensions (`Run
command` listed under Tool permissions, no "Unable to connect" error),
and a real `run_command` call (`ls -la`) was executed end-to-end
through the full chain -- Desktop's UtilityProcess, the in-process
mcp-remote proxy, HTTP over the WSL2 port forward, into the Docker
container -- and returned real `/workspace` directory output with
`exit_code: 0`. See "What was NOT verified" for the full history of
what this fixed.

## Start everything — one script, from the project root

```bash
./start-mcp-server.sh
```

**Never run `docker compose`/`docker-compose` directly against
`tools/server/docker-compose.yml`.** Confirmed recurring (not
hypothetical): doing so skips `start.sh`'s `compose rm -sf` step and
hits docker-compose v1's `KeyError: 'ContainerConfig'` bug
(docker/compose#11742) when recreating an existing container after a
rebuild -- the exact failure this section's own `start.sh` script
exists to prevent. Always go through `start-mcp-server.sh` or
`tools/pipeline.sh server`.

This lives at the repo root and delegates to
`tools/server/start.sh` (which does the actual work: resolve
`PROJECT_DIR` via git, `docker compose build` then `docker compose up
-d`, removing any stale container first to avoid docker-compose v1's
`KeyError: 'ContainerConfig'` bug) -- one place the real logic lives,
one place you actually run it from. Works from any `$PWD`, not just
the repo root.

Both `start-mcp-server.sh` and `tools/server/start.sh` need their
execute bit set once after a fresh checkout (`chmod +x`) -- git
preserves the bit once it's committed, but any tool that rewrites
the file wholesale (including remote file-edit tools) can silently
drop it, so re-check after out-of-band edits.

Check it's actually up:
```bash
docker compose -f tools/server/docker-compose.yml ps
curl http://localhost:1443/mcp   # should respond, not connection-refused
```
A plain `curl`/browser GET gets back a JSON-RPC error --
`"Not Acceptable: Client must accept text/event-stream"` -- because
the streamable-HTTP transport requires clients to declare SSE
support via an `Accept` header. That response is the success signal
here: it means the server is up and enforcing the protocol correctly,
not that something is broken. For an actual protocol-level test, use
[MCP Inspector](https://modelcontextprotocol.io/docs/2026-07-28/tools/inspector)
rather than hand-rolling a curl request against the SSE/session
negotiation.

## Call a tool directly, without Desktop

```bash
bash tools/server/call.sh chmod +x tools/server/start.sh
```

Wraps MCP Inspector's CLI client (`npx @modelcontextprotocol/inspector
--cli`) around `run_command`, so testing an allowlist change or any
other tool call against the already-running container doesn't mean
hand-building a `--tool-args-json` payload each time. Point it at a
different port with `MCP_URL=http://localhost:PORT/mcp`. Deliberately
callable as `bash tools/server/call.sh ...` without `chmod +x` first --
it exists partly to test that `chmod` works *through* the proxy, so it
can't depend on `chmod` having already been run outside it.

## Build the Desktop extension

Source (`index.js`, `manifest.json`) and build output live together
under `tools/desktop-extension/`, output specifically into `dist/` so
it's never ambiguous which files are source and which are the
built artifact:

```bash
tools/desktop-extension/build.sh
```

Scripted rather than manual `npm install` + `npx --yes
@anthropic-ai/mcpb pack . dist/locusai-local-bash.mcpb` (which is all
it runs, plus `npm`/`npx` presence checks) so neither step can drift
from what's documented here. `npm install` runs every time (fast
no-op if `package.json` hasn't changed) because `mcp-remote` must
actually be on disk in `node_modules/` for `mcpb pack` to bundle it --
see `index.js` for why it's bundled instead of `npx`-fetched at
launch. Needs `chmod +x` once after a fresh checkout, same caveat as
the start scripts above.

Confirmed safe to re-run repeatedly -- the CLI automatically excludes
its own prior output in `dist/` from being swept into a new pack
(verified directly, not assumed). `dist/` is gitignored; it's a build
artifact, regenerate it rather than committing it.

Drag the resulting `.mcpb` into Claude Desktop's Settings > Extensions
panel.

## Run the pipeline (lint/test/build/server/verify)

```bash
tools/pipeline.sh lint      # shellcheck + node --check + py_compile, no side effects
tools/pipeline.sh test      # node --test tools/desktop-extension/test -- no Docker/Desktop needed
tools/pipeline.sh build     # tools/desktop-extension/build.sh -> dist/locusai-local-bash.mcpb
tools/pipeline.sh server    # tools/server/start.sh, then confirms the /mcp endpoint responds
tools/pipeline.sh verify    # confirms the endpoint, then one real run_command (ls) via call.sh
tools/pipeline.sh all       # all of the above, in order
```

Deliberately **local stage scripts, not a cloud CI/CD runner**: the
stages that matter most (packing the real `.mcpb`, starting a real
Docker container, confirming a real Claude Desktop install actually
connects) only mean anything on the machine that has Claude Desktop +
WSL2 + Docker running -- a cloud runner (GitHub Actions or similar)
has no way to reach any of that, so there isn't one. `lint` and `test`
are Docker/Desktop-agnostic and are meant to run wherever development
is actually happening -- run them there before anything is pushed to
this machine. `build`, `server`, and `verify` need this real machine
and are meant to run here, as the staging step after that development.

Each stage reports `[PASS]`, `[FAIL]`, or `[SKIP]` (a required tool
missing from `PATH`, e.g. no `docker` -- not treated as a failure).
`all` exits non-zero only if something actually `[FAIL]`ed. Verified
directly against a disposable mock repo before ever touching this one
(clean-pass, deliberate-lint-failure, and missing-tool-SKIP cases all
produced the expected `[PASS]`/`[FAIL]`/`[SKIP]` and exit code) --
this repo's own `tools/desktop-extension/test/index.test.js` (see
below) is the permanent, real-repo counterpart of that same
mock-first verification discipline.

`tools/desktop-extension/test/index.test.js` uses Node's built-in
`node:test` to exercise the real, shipped `index.js` against a fake
local `mcp-remote` package (same shape as the real one -- `"type":
"module"`, a top-level `process.argv.slice(2)` read, `console.error`
logging) covering three cases: clean success (argv threaded through,
`console.error` tee'd into the debug log, real stdout passed through
unmodified), the dependency missing entirely, and the dependency
throwing during import. This is a mechanics/wiring test -- it needs no
network access and no Docker, so it runs anywhere Node runs, but it
deliberately does NOT verify that the real, compiled
`node_modules/mcp-remote/dist/proxy.js` behaves identically to the
fake stand-in, or that a real Claude Desktop install can actually
launch this file -- those still need the `server`/`verify` stages
against the real machine. One real gotcha surfaced while building this
suite, worth not rediscovering: `node --test <directory>` does not
reliably recurse into a non-default-named directory passed explicitly
on the command line in Node 22.22.2 (it tried to `require()` the
directory itself and failed with `MODULE_NOT_FOUND`, even though bare
`node --test` with no path correctly auto-discovers the same files
from cwd) -- `pipeline.sh`'s `test` stage works around this by
enumerating `*.test.js` files explicitly rather than passing a bare
directory.

## Wire into Claude Desktop manually instead, if preferred

`%APPDATA%\Claude\claude_desktop_config.json`:

```json
{
  "mcpServers": {
    "local-bash": {
      "command": "node",
      "args": ["C:\\path\\to\\LocusAI\\tools\\desktop-extension\\index.js"]
    }
  }
}
```

## `local-git` — not yet redone for this architecture, flagged not assumed

`mcp-server-git` (the official reference server) hasn't been verified
to support `streamable-http` transport the way `bash_mcp_server.py`
was just confirmed to. Until that's checked, `local-git` needs either
the old per-request model or the same HTTP-transport treatment once
verified -- don't assume parity here.

## Security note, explicit rather than implied

`docker-compose.yml` binds the published port to `127.0.0.1` only,
not `0.0.0.0` -- belt-and-braces on top of WSL2's own default
localhost-only forwarding. This is a deliberate trade against the
earlier `--network none` per-request design (full isolation, but
incompatible with a reachable port), not an accidental loss of it.

## What was verified, and how
- The HTTP-transport server handled a real MCP `initialize` handshake
  (200 OK, correct JSON-RPC response) -- actually run, not assumed.
- `index.js` was actually run against the live server and completed
  the full stdio<->HTTP proxy handshake, in the pre-bundling version
  that shelled out to `npx -y mcp-remote`.
- An earlier, now-superseded rewrite of `index.js` (bundled
  `mcp-remote`, resolved via its own `package.json` "bin" field and
  spawned as a subprocess) was exercised against a fake local
  `mcp-remote` package covering three paths: clean success with
  byte-for-byte stdout/stderr forwarding, `mcp-remote` missing
  entirely from `node_modules`, and the spawn target itself missing
  -- all three produced a clear, logged failure rather than a silent
  hang or a cryptic crash. That subprocess design has since been
  replaced by the in-process `import()` design described above; see
  "What was NOT verified" for why.
- The current in-process `import()` version of `index.js` was
  exercised in a sandbox against a fake local `mcp-remote` package
  that mimics the real one's shape (`"type": "module"`, top-level
  `process.argv.slice(2)` read, `console.error`-based logging): argv
  was correctly threaded through before the dynamic `import()`
  resolved, the fake module's own `console.error` calls were
  correctly tee'd into `debugLog()` while still reaching real stderr,
  and the process exited cleanly with the fake module's simulated
  stdout output intact. Confirms the mechanics (argv timing,
  `pathToFileURL()` usage, console.error tee) work as designed. **Now
  also confirmed against the real, compiled `mcp-remote` package on
  the user's actual machine** -- see the "Fixed" entry at the end of
  "What was NOT verified" below.
- The packed `.mcpb` round-trips identically through unpack.
- Packing into `dist/` repeatedly, with a stale `.mcpb` already
  present, does not bundle the old archive into the new one --
  tested directly.
- The top-level `start-mcp-server.sh` correctly delegates to
  `tools/server/start.sh` from the repo root and from an unrelated
  `$PWD` -- both tested.
- All shell scripts (`start-mcp-server.sh`, `tools/server/start.sh`,
  `tools/server/resolve-project-dir.sh`, `tools/desktop-extension/build.sh`,
  `tools/pipeline.sh`) pass ShellCheck with zero warnings -- confirmed
  live 2026-08-05 against `tools/pipeline.sh`'s own `stage_lint`
  invocation, not just an ad hoc run: plain `shellcheck` on all repo
  `.sh` files reported `SC1091` on `start.sh`'s dynamic `source` of
  `resolve-project-dir.sh`, because `-x` alone resolves
  `# shellcheck source=` relative to shellcheck's own cwd, not each
  script's own directory. Fixed by adding `-P SCRIPTDIR` alongside
  `-x` in `stage_lint`; re-verified clean (exit 0) afterward, both
  standalone and via `tools/pipeline.sh lint`.
- `docker compose build` then `up -d` actually succeeding against a
  real Docker daemon -- confirmed on the user's own machine: build
  completed, the `compose rm -sf` step stopped and removed the prior
  `server_local-bash_1` container, and `up -d` recreated it cleanly
  with no `KeyError: 'ContainerConfig'`.
- The running server correctly rejects a plain GET (no
  `Accept: text/event-stream`) with a JSON-RPC `-32600` error rather
  than failing to respond at all -- confirmed live, and this is the
  expected streamable-HTTP behavior, not a bug.
- `tools/pipeline.sh` itself was verified against a disposable mock
  repo (fake `.sh`/`.js`/`.py` files standing in for the real ones)
  before ever touching this repo: a clean run reports `[PASS]` on
  every stage with exit 0, a deliberately broken shell file makes
  `lint` report `[FAIL]` with exit 1, and a stage whose required tool
  (`docker`) is absent reports `[SKIP]` rather than `[FAIL]` -- all
  three outcomes confirmed directly, not assumed from reading the
  script. `tools/desktop-extension/test/index.test.js` was likewise
  run against the real, shipped `index.js` (copied byte-for-byte, not
  duplicated) before being committed here, and all three cases passed.

## What was NOT verified, deliberately flagged rather than assumed:
- The `.mcpb` bundle's actual install/run experience inside Claude
  Desktop's UI. As of this writing, installing it produces "Unable to
  connect to extension server" -- under active debugging. Ruled out
  so far: the server itself (a direct `curl`/Inspector `initialize`
  against `localhost:1443` from the same machine works cleanly),
  capability negotiation (Inspector's `initialize` response shows an
  unremarkable, complete capabilities object), and an AppContainer
  loopback block (added via `CheckNetIsolation.exe LoopbackExempt`, no
  observed change in behavior).

  A `debug.log` captured from a real install (bundled-`mcp-remote`
  version, logging to a file next to `index.js`) showed three separate
  launches within 9 seconds, each spawning `mcp-remote` successfully
  (correct `__dirname`, correct resolved entry path,
  `node_modules/mcp-remote` present) and each child then exiting
  cleanly with `code=0` after nothing but a stray Node `DEP0169`
  deprecation warning on stderr. That does not match mcp-remote's own
  connection-failure path (`src/proxy.ts` logs an error and calls
  `process.exit(1)`); a clean `exit(0)` with no error only happens via
  mcp-remote's `stdin` `'end'` handler -- i.e. something is closing
  this process's stdin, and Desktop is respawning the extension
  moments later. This matches a reported Claude Desktop bug
  ([anthropics/claude-code#61052](https://github.com/anthropics/claude-code/issues/61052)):
  Desktop silently closing an MCP child's stdin pipe with no log entry
  and no logged auto-relaunch.

  Untested hypothesis, not confirmed by any doc: `debug.log` was being
  written into `__dirname`, which for an "Install Unpacked Extension"
  is the *live* extension directory Desktop just launched from -- if
  Desktop watches that directory for dev-mode hot-reload, our own log
  writes could be self-inflicting the restart loop. `index.js` now
  logs to `%APPDATA%\Claude\logs\locusai-local-bash-debug.log`
  instead -- the same directory Desktop's own `mcp-server-*.log` files
  already live in, confirmed against the official
  [MCP debugging docs](https://modelcontextprotocol.io/docs/tools/debugging)
  rather than assumed, so both logs sit next to each other instead of
  being one more path to remember. Falls back to `os.tmpdir()` if
  `APPDATA` isn't set (Linux, which `manifest.json` declares as
  compatible even though Desktop itself only ships for macOS/Windows).

  Source-verified lead (checked against a real clone of
  [geelen/mcp-remote](https://github.com/geelen/mcp-remote), not
  guessed): `process.execPath` in the captured `debug.log` is
  `Claude.exe` itself (Electron's own binary), not a standalone
  `node.exe`. Electron only runs a spawned script as plain Node when
  `ELECTRON_RUN_AS_NODE=1` is set in its environment -- see
  [Electron's environment-variables docs](https://www.electronjs.org/docs/latest/api/environment-variables)
  -- otherwise it boots as the full GUI app. Claude Desktop is a
  single-instance app, so a second Electron instance launched without
  that variable would detect the already-running primary instance and
  quit almost immediately: a clean `exit code=0`, no error, and no
  output from any of mcp-remote's own logging (every `log()` call in
  `src/proxy.ts`/`src/lib/utils.ts` is prefixed with the process's pid
  and would have shown up in the captured stderr if `runProxy()` had
  ever actually started -- it never did). That matches the debug.log
  evidence exactly: repeated launches, each ending in `exit code=0`
  with nothing but a stray Node deprecation warning on stderr, never
  mcp-remote's own `Fatal error:` + `exit(1)` path.

  **Tested and only half-confirmed.** `index.js` was changed to
  explicitly pass `env: { ...process.env, ELECTRON_RUN_AS_NODE: '1' }`
  to the spawned child and to log the variable's own value on the
  *parent* (this script's) side before spawning. A real install run
  confirmed part of the theory: the parent process logged
  `ELECTRON_RUN_AS_NODE=(unset)`, proving Desktop does *not* use that
  env var to launch `index.js` itself (so it must use something else,
  e.g. Electron's `utilityProcess` API, to run this script as plain
  Node). But explicitly setting it for the *child* spawn made no
  observable difference whatsoever -- identical timing, identical
  lone `DEP0169` warning, identical clean `exit code=0`. If the
  env var were the whole story, setting it should have changed the
  outcome one way or another. Getting byte-for-byte the same failure
  is what a disabled
  [`runAsNode` fuse](https://www.electronjs.org/docs/latest/tutorial/fuses)
  looks like: packaged/Microsoft-Store Electron builds can turn this
  off entirely, in which case Electron ignores the env var
  unconditionally and always boots as itself.

  Given that, `index.js` (in an intermediate version, since replaced
  -- see the next entry) stopped spawning `process.execPath` for
  mcp-remote and instead scanned `PATH` directly for a real
  `node.exe` (`findSystemNode()`, no `where`/`which` shell-out, same
  no-shell/no-network spirit as avoiding `npx`), falling back to
  `process.execPath` + `ELECTRON_RUN_AS_NODE` only if no standalone
  Node was found.

  **Retested -- fixed the fuse issue, surfaced a second one.** A real
  install run with the real `node.exe` fix showed mcp-remote actually
  connect: `Connected to remote server using StreamableHTTPClientTransport`
  / `Local STDIO server running` / `Proxy established successfully`.
  Then, ~1ms later, mcp-remote logged its own `Shutting down...` and
  exited cleanly. Checking Desktop's own `main.log` (not just this
  extension's `debug.log`) settled which side caused it: Desktop's
  `[LocalMcpServerManager] Failed to connect to LocusAI Local Bash:
  MCP error -32000: Connection closed` is logged *after* mcp-remote's
  `Shutting down...` line, with a stack trace through
  `ForkUtilityProcess`'s `onclose` handler -- i.e. Desktop is reacting
  to mcp-remote's own exit, not causing it. `main.log` also directly
  confirms Desktop launches this extension via Electron's
  `utilityProcess.fork()` (`"Using UtilityProcess for extension
  LocusAI Local Bash: appConfig.isUsingBuiltInNodeForMcp is true"`).

  At that point `index.js` still spawned mcp-remote as a subprocess
  and had been changed from `stdio: ['inherit', ...]` to `stdio:
  ['pipe', ...]` plus `process.stdin.pipe(child.stdin, { end: false
  })`, on the theory that a `UtilityProcess`-forked process's own
  stdin isn't necessarily wired the same way a normally-spawned
  process's is, and that decoupling this process's stdin `'end'` event
  from the child's stdin would stop mcp-remote's own stdin-`'end'`
  shutdown handler from firing. That fix was never retested, because
  a more direct read of the same evidence pointed at removing the
  subprocess (and the relay) entirely instead -- see the next entry.

  **Superseded the whole subprocess design -- clone the Filesystem
  extension's architecture instead.** The official reference
  "Filesystem" MCP extension
  ([modelcontextprotocol/servers](https://github.com/modelcontextprotocol/servers),
  `src/filesystem/index.ts`, cloned locally and confirmed to be the
  literal code Desktop bundles -- its log strings match `main.log`
  verbatim) has no subprocess at all: its entire stdio setup is `new
  StdioServerTransport()` + `server.connect(transport)`, operating
  directly on the real process's own `process.stdin`/`process.stdout`
  -- the same object Desktop's `UtilityProcess` is driving. `index.js`
  was rewritten to match: it no longer spawns mcp-remote as a child
  process at all. Instead it resolves `node_modules/mcp-remote`'s
  compiled entry (`dist/proxy.js`, via `package.json`'s `bin` field,
  same as before), sets `process.argv` to what that entry point's own
  top-level `parseCommandLineArgs(process.argv.slice(2))` call expects
  (confirmed by reading `src/proxy.ts` directly -- this runs
  synchronously at module-evaluation time, so `process.argv` must be
  set *before* the import resolves), and loads it in-process with a
  dynamic `import(pathToFileURL(entry).href)` (mcp-remote publishes
  itself as `"type": "module"`, so `import()` rather than `require()`;
  `pathToFileURL()` because `import()` doesn't accept raw Windows
  paths). `console.error` is monkey-patched to also tee into
  `debugLog()`, since mcp-remote's own logging helpers
  (`src/lib/utils.ts`) log via `console.error`, while still calling
  through to the original so Desktop's normal stderr capture is
  unaffected.

  This removes `findSystemNode()`, the `spawn()` call, all `child.*`
  event handlers, the `stdin.pipe(..., { end: false })` relay, and the
  `ELECTRON_RUN_AS_NODE` env-passing entirely -- not by fixing any of
  those mechanisms further, but by deleting the second process they
  all existed to manage. There is now exactly one process, exactly one
  real `process.stdin`, and no relay for its lifecycle to diverge
  through.

  **Fixed -- confirmed against a real Claude Desktop install.** After
  rebuild + reinstall, Settings > Extensions shows `locusai-local-bash`
  as `Enabled` with `Run command` listed under Tool permissions --
  the "Unable to connect to extension server" error is gone. A real
  `run_command` call (`ls -la` against the mounted `/workspace`) was
  executed through the complete real chain -- Desktop's
  `UtilityProcess`, the in-process mcp-remote proxy inside `index.js`,
  HTTP over the WSL2 port forward, the Docker container's
  `bash_mcp_server.py` -- and returned real directory output with
  `exit_code: 0`. This is the first fix in the whole debugging session
  confirmed by an actual successful tool call rather than just a
  connection log line, closing out the "Unable to connect to extension
  server" issue this section was originally opened to track.
- A real MCP client (`local-bash` wired into Desktop, exercised live)
  completing a full `initialize` handshake and a real tool call
  against the container running on the user's actual machine via
  `index.js` -- confirmed above (the `run_command` / `ls -la` call).
- The allowlist now lives in `server/allowlist.txt`, not hardcoded in
  `bash_mcp_server.py` -- one binary name per line, `#` comments
  allowed. It currently permits: ls, cat, grep, find, wc, head, tail,
  python3, pip, pytest, git. Because the file sits under the
  `/workspace` bind mount rather than the image's `COPY`'d `/app`
  layer, editing it on the host takes effect on the server's next
  request (mtime-checked, see `FileAllowlist` in `server/lib/common.py`)
  with no container rebuild or restart. Extend it only after reviewing
  each addition. Currently permits: ls, cat, grep, find, wc, head,
  tail, python3, pip, pytest, git, gh, chmod.
- **GitHub auth split (added 2026-08-06):** `git push`/pull go over
  ssh using a read-only-mounted deploy key
  (`~/.ssh/github` on the host → `/root/.ssh/github_deploy_key` in the
  container, per `docker-compose.yml`); everything else (PRs, issues,
  releases) goes through `gh`'s own token via `GH_TOKEN`. Set
  `GH_TOKEN` in the launching shell or in a gitignored `.env` next to
  `docker-compose.yml` -- never commit a real token there. GitHub's
  published ed25519 host key is pinned in the image's
  `/root/.ssh/known_hosts` (verified against GitHub's own docs
  2026-08-06) rather than trusted on first use -- deliberately only
  the ed25519 entry, with `HostKeyAlgorithms ssh-ed25519` in
  `/root/.ssh/config` to match. Never mount the whole `~/.ssh`
  directory into this container -- it also exposes the allowlisted
  `run_command` execution surface above, so only the one key meant
  for this purpose goes in.
- `tools/pipeline.sh` run for real, end-to-end, against this repo's own
  checkout on 2026-08-05 (`tools/pipeline.sh all`): `lint`, `test`,
  `build`, `server`, and `verify` all `[PASS]`. This surfaced and fixed
  two real defects that only existed because this had never actually
  been run here before: (1) `stage_lint`'s ShellCheck call needed
  `-P SCRIPTDIR` alongside `-x` for `start.sh`'s dynamic `source` to
  resolve regardless of invocation cwd; (2) `stage_verify_endpoint`
  used `curl --fail`, which treats the server's correct HTTP 406
  response (see "Check it's actually up" above) as an error --
  replaced with an explicit status/body check, verified first against
  a mock server reproducing the exact real response before being
  pushed. Also added a ~10s readiness-poll retry loop, since
  `docker compose up -d` returns before the app inside has finished
  binding the socket.
