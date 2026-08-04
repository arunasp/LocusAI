#!/usr/bin/env bash
# Usage: ./mcp-run.sh <service-name>
# e.g.:  ./mcp-run.sh local-bash
#        ./mcp-run.sh local-git
#
# PROJECT_DIR resolution, in priority order:
#   1. Already set in the environment (MCPB packaging sets this from
#      user_config.workspace_directory -- once installed as a Desktop
#      extension, this script lives in Claude's private extension
#      directory, not inside the LocusAI repo, so git-based detection
#      cannot work and must not run).
#   2. Resolved via git (see resolve-project-dir.sh) -- the manual,
#      run-from-inside-the-repo case this was originally built for.
set -euo pipefail

if [[ $# -ne 1 ]]; then
    echo "usage: $(basename "$0") <service-name>" >&2
    exit 1
fi

script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

if [[ -z "${PROJECT_DIR:-}" ]]; then
    # shellcheck source=resolve-project-dir.sh
    source "${script_dir}/resolve-project-dir.sh"
    PROJECT_DIR="$(resolve_project_dir)"
fi
export PROJECT_DIR

exec docker compose --project-directory "${script_dir}" run --rm "$1"
