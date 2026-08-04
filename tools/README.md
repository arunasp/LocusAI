# Local git/bash MCP tools

## 0. No path configuration needed — it's resolved, not hardcoded

`PROJECT_DIR` is no longer a value you set. `resolve-project-dir.sh`
defines `resolve_project_dir()`, which asks git for this repo's root
(`git rev-parse --show-toplevel`, run from the script's own location,
not your shell's `$PWD`) — so it's correct regardless of where LocusAI
is cloned, what it's renamed to, or which directory you invoke from.
`mcp-run.sh` calls that function and exports the result fresh on every
launch before handing off to compose, unless `PROJECT_DIR` is already
set in the environment (that's the MCPB-packaged-extension case — see
below). There is nothing to edit before building.

## 1. Build (inside WSL2 Ubuntu, in this directory)

```bash
docker compose build
```

Compose reads `PROJECT_DIR` from the shell environment `mcp-run.sh`
sets — `docker compose build` alone doesn't need it (the build stage
doesn't mount anything), only `run` does.

## 2. Verify before trusting it

```bash
./mcp-run.sh local-bash   # should start cleanly against /workspace
./mcp-run.sh local-git    # should start cleanly against the mounted repo
```

Ctrl-C each after confirming it starts without an import/missing-binary
error — these are long-lived stdio processes, not one-shot commands.
If either prints `resolve_project_dir: ... is not inside a git
repository`, you're running it from outside a cloned copy of this repo
— that's the function correctly refusing to guess, not a bug to work
around.

Note: `uvx` isn't installed by this Dockerfile (only `pip` requirements
are). Either add `pip install uv` to the Dockerfile, or switch
`local-git`'s command in docker-compose.yml to
`["python3", "-m", "mcp_server_git", "--repository", "/workspace"]`
(mcp-server-git is already a pip dependency here) — whichever you
verify actually works in this step.

## 3. Wire into Claude Desktop

Windows: `%APPDATA%\Claude\claude_desktop_config.json`

```json
{
  "mcpServers": {
    "local-bash": {
      "command": "C:\\path\\to\\LocusAI\\tools\\mcp-run.sh",
      "args": ["local-bash"]
    },
    "local-git": {
      "command": "C:\\path\\to\\LocusAI\\tools\\mcp-run.sh",
      "args": ["local-git"]
    }
  }
}
```

Desktop calls the wrapper directly now, not `docker compose` — the
wrapper resolves the path and sets `--project-directory` itself, so
there's no `cwd` field to get wrong and no `.env` file whose contents
could go stale if the repo is ever moved. `--network none` on
local-bash and no network restriction on local-git are set in
docker-compose.yml, not passed here.

If Desktop is launched from Windows but this script lives in WSL2, the
`command` path needs to resolve through whatever WSL2-bridging
approach you're using (the `\\wsl.localhost\...` UNC path, or an SSH
invocation).

## 4. Alternative: install as an MCPB bundle instead of editing config

`local-bash` is also packaged as a proper one-click `.mcpb` extension
(drag into Claude Desktop's Settings > Extensions panel, same drop
zone as the Filesystem extension). That version prompts for the
project directory via a picker (`user_config.workspace_directory`)
rather than relying on git auto-detection, since a packaged extension
runs from Claude's own extension directory, not from inside this repo
-- `mcp-run.sh`'s environment-variable-first PROJECT_DIR check (see
section 0) is what makes both distribution modes work from the same
script.

## What was verified, and how
- `resolve_project_dir()` correctly returns the git repo root when
  sourced from within the repo, AND when invoked from a completely
  unrelated `$PWD`.
- `mcp-run.sh` validates its argument and correctly skips git
  resolution when `PROJECT_DIR` is already set (the packaged-extension
  path), and correctly falls back to git resolution when it isn't (the
  manual-terminal path) -- both modes tested directly.
- Both scripts pass ShellCheck with zero warnings.
- The `local-bash` `.mcpb` bundle was built and packed with the real
  `mcpb` CLI (not hand-assembled), validated against the current
  manifest schema, and round-tripped through unpack to confirm
  identical content.

## What was NOT verified, deliberately flagged rather than assumed:
- `docker compose build` and `docker compose run` actually succeeding
  against a real Docker daemon.
- `uvx` availability inside the image (see note in step 2).
- The `.mcpb` bundle's actual install experience inside Claude
  Desktop's UI (the picker prompt, etc.) -- only the CLI-level
  packaging was verified.
- The allowlist in `bash_mcp_server.py` currently permits: ls, cat,
  grep, find, wc, head, tail, python3, pip, pytest. Extend it only
  after reviewing each addition -- this is the whole point of using an
  allowlist instead of raw shell access.
