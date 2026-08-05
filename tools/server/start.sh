#!/usr/bin/env bash
# Starts the persistent MCP server: resolves PROJECT_DIR via git, then
# builds and launches as two explicit, separately-failing steps
# (rather than relying on `up --build` to do both implicitly) --
# build errors and launch errors are then distinguishable at a glance.
set -euo pipefail

script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=resolve-project-dir.sh
source "${script_dir}/resolve-project-dir.sh"

PROJECT_DIR="$(resolve_project_dir)"
export PROJECT_DIR

cd "${script_dir}"
echo "Building..." >&2
docker compose build
echo "Launching..." >&2
docker compose up -d
