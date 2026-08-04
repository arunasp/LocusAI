#!/usr/bin/env bash
# Usage: ./mcp-run.sh <service-name>
# e.g.:  ./mcp-run.sh local-bash
#        ./mcp-run.sh local-git
#
# Resolves PROJECT_DIR via git (see resolve-project-dir.sh) and execs
# docker compose with it set, so nothing is hardcoded in .env or in
# Claude Desktop's config -- the path is computed fresh on every launch.
set -euo pipefail

if [[ $# -ne 1 ]]; then
    echo "usage: $(basename "$0") <service-name>" >&2
    exit 1
fi

script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=resolve-project-dir.sh
source "${script_dir}/resolve-project-dir.sh"

PROJECT_DIR="$(resolve_project_dir)"
export PROJECT_DIR

exec docker compose --project-directory "${script_dir}" run --rm "$1"
