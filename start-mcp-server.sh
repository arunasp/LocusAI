#!/usr/bin/env bash
# Single entry point at the project root for the MCP tooling
# specifically -- named start-mcp-server.sh, not start.sh, because
# LocusAI itself (the actual bio-inspired system, not yet
# implemented) will need its own separate top-level launcher once it
# exists. Delegates to tools/server/start.sh, which does the real
# work -- one place the actual logic lives.
#
# Usage: ./start-mcp-server.sh     (from anywhere; resolves its own location)
set -euo pipefail

script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
exec "${script_dir}/tools/server/start.sh"
