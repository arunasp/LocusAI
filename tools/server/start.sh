#!/usr/bin/env bash
# Starts the persistent MCP server: resolves PROJECT_DIR via git, then
# builds and launches as two explicit, separately-failing steps
# (rather than relying on `up --build` to do both implicitly) --
# build errors and launch errors are then distinguishable at a glance.
#
# Detects docker compose (v2, plugin) vs docker-compose (v1, standalone)
# at runtime rather than assuming one -- confirmed real gap: this
# machine has only v1 installed, not the newer v2 plugin syntax every
# command here originally assumed.
set -euo pipefail

if docker compose version &>/dev/null; then
    compose() { docker compose "$@"; }
elif command -v docker-compose &>/dev/null; then
    compose() { docker-compose "$@"; }
else
    echo "error: neither 'docker compose' (v2) nor 'docker-compose' (v1) found" >&2
    exit 1
fi

script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=resolve-project-dir.sh
source "${script_dir}/resolve-project-dir.sh"

PROJECT_DIR="$(resolve_project_dir)"
export PROJECT_DIR

cd "${script_dir}"

# Compose has no precondition hook (confirmed for this same v1 CLI in
# opencode-model-eval), so this pre-flight -- phantom-mount-dir defense
# for the ssh key + auto-extracting GH_TOKEN if missing -- runs here
# explicitly rather than relying on compose to trigger it.
"${script_dir}/ensure-auth-data.sh"

echo "Building..." >&2
compose build

# Drop any existing container before recreating it. This isn't just
# cleanup: `compose up -d` recreating an existing container walks a
# path that reads the *old* container's image metadata to migrate its
# anonymous volumes, and on docker-compose v1 that crashes with
# `KeyError: 'ContainerConfig'` when the new image was built by
# BuildKit (BuildKit output omits the legacy ContainerConfig field
# v1 expects -- confirmed via docker/compose#11742, closed upstream
# as "not planned" for v1). There's nothing to lose here: the only
# volume in docker-compose.yml is the ${PROJECT_DIR} bind mount, not
# an anonymous one, so removing the container first is a safe no-op
# on v2 and the actual fix on v1. `-s` stops it first since the
# service runs with `restart: unless-stopped`.
echo "Removing any stale container..." >&2
compose rm -sf
echo "Launching..." >&2
compose up -d
