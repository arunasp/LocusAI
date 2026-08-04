# Local git/bash MCP tools

## Structure — two clean parts, only touching at one point

- **`server/`** — the actual MCP server implementation (Dockerfile,
  docker-compose.yml, bash_mcp_server.py, lib/, requirements.txt).
  Fully standalone: buildable and testable with plain `docker compose`,
  no dependency on Claude Desktop or MCPB packaging at all.
- **`desktop-extension/`** — the proxy Claude Desktop actually spawns
  (manifest.json, mcp-run.sh, resolve-project-dir.sh). Deliberately
  minimal — nothing in this directory except what MCPB needs, so
  packing it can never accidentally sweep in unrelated files (this
  split exists specifically because that happened once).

The only coupling point: `desktop-extension/mcp-run.sh` resolves
`../server` to an absolute path and passes it to `docker compose
--project-directory`. Nothing else crosses the boundary.

## 0. No path configuration needed — it's resolved, not hardcoded

`PROJECT_DIR` is no longer a value you set. `resolve-project-dir.sh`
(in `desktop-extension/`) defines `resolve_project_dir()`, which asks
git for this repo's root (`git rev-parse --show-toplevel`, run from
the script's own location, not your shell's `$PWD`) — so it's correct
regardless of where LocusAI is cloned or which directory you invoke
from. `mcp-run.sh` calls that function and exports the result fresh on
every launch, unless `PROJECT_DIR` is already set in the environment
(the MCPB-packaged-extension case).

## 1. Build and verify the server (independent of Desktop entirely)

```bash
cd tools/server
docker compose build
docker compose run --rm local-bash   # should start cleanly against /workspace
docker compose run --rm local-git    # should start cleanly against the mounted repo
```

Note: `uvx` isn't installed by this Dockerfile (only `pip`
requirements are). Either add `pip install uv` to the Dockerfile, or
switch `local-git`'s command in docker-compose.yml to
`["python3", "-m", "mcp_server_git", "--repository", "/workspace"]`
(mcp-server-git is already a pip dependency here) — whichever you
verify actually works.

## 2. Verify the proxy end-to-end

```bash
cd tools/desktop-extension
./mcp-run.sh local-bash   # should reach the same point as step 1
```

If it prints `resolve_project_dir: ... is not inside a git
repository`, you're running it from outside a cloned copy of this repo
— that's the function correctly refusing to guess, not a bug.

## 3. Wire into Claude Desktop

Windows: `%APPDATA%\Claude\claude_desktop_config.json`

```json
{
  "mcpServers": {
    "local-bash": {
      "command": "C:\\path\\to\\LocusAI\\tools\\desktop-extension\\mcp-run.sh",
      "args": ["local-bash"]
    },
    "local-git": {
      "command": "C:\\path\\to\\LocusAI\\tools\\desktop-extension\\mcp-run.sh",
      "args": ["local-git"]
    }
  }
}
```

## 4. Or install as a proper MCPB bundle instead of editing config

```bash
cd tools/desktop-extension
npx --yes @anthropic-ai/mcpb pack . ../../locusai-local-bash.mcpb
```

Run this from inside `desktop-extension/`, not the repo root — packing
from the wrong directory pulls in `.git` and everything else in the
repo, and the manifest won't be found at the archive root where MCPB
expects it. This directory being minimal by construction is exactly
what makes that mistake structurally hard to make again.

Drag the resulting `.mcpb` into Claude Desktop's Settings > Extensions
panel. It should prompt for the "LocusAI project directory" via a
picker (`user_config.workspace_directory`) rather than relying on git
auto-detection, since a packaged extension runs from Claude's own
extension directory, not from inside this repo.

## What was verified, and how
- `resolve_project_dir()` returns the correct git repo root both from
  within the repo and from a completely unrelated `$PWD`.
- `mcp-run.sh` correctly resolves `../server` to an absolute path and
  reaches the docker compose invocation in both PROJECT_DIR modes
  (pre-set, and git-resolved) -- tested directly.
- Both scripts pass ShellCheck with zero warnings.
- Packing `desktop-extension/` with the real published
  `@anthropic-ai/mcpb` CLI produces exactly 3 files (manifest.json,
  mcp-run.sh, resolve-project-dir.sh), 2.1kB -- confirmed the
  directory split makes accidental whole-repo packing structurally
  impossible, not just avoided by remembering the right command.

## What was NOT verified, deliberately flagged rather than assumed:
- `docker compose build`/`run` actually succeeding against a real
  Docker daemon.
- The `.mcpb` bundle's actual install experience inside Claude
  Desktop's UI (the picker prompt, etc.) -- only CLI-level packaging
  was verified.
- The allowlist in `server/bash_mcp_server.py` currently permits: ls,
  cat, grep, find, wc, head, tail, python3, pip, pytest. Extend it
  only after reviewing each addition.
