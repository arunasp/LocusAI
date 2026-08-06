#!/usr/bin/env bash
# Calls the local-bash MCP server's run_command tool directly via MCP
# Inspector's CLI client, bypassing Desktop entirely -- lets you
# smoke-test an allowlist change (or anything else run_command
# exposes) against the already-running container with one command,
# instead of hand-building the --tool-args-json payload each time.
#
# Usage: tools/server/call.sh <binary> [args...]
# Example: tools/server/call.sh chmod +x tools/server/start.sh
#
# No +x needed on this file itself to use it -- `bash tools/server/call.sh
# ...` works even before chmod does, which matters since chmod wasn't
# runnable through the proxy until the allowlist change this script
# was written to help test.
set -euo pipefail

if [[ $# -lt 1 ]]; then
    echo "usage: $0 <binary> [args...]" >&2
    exit 1
fi

if ! command -v npx &>/dev/null; then
    echo "error: 'npx' not found -- install Node.js/npm first" >&2
    exit 1
fi

binary="$1"
shift

# Build the JSON-RPC tool-call payload with python3's json module
# rather than string-interpolating args into JSON by hand -- avoids
# breaking on quotes/spaces/backslashes in an argument.
payload="$(python3 -c '
import json
import sys

binary = sys.argv[1]
args = sys.argv[2:]
print(json.dumps({"binary": binary, "args": args}))
' "${binary}" "$@")"

mcp_url="${MCP_URL:-http://localhost:1443/mcp}"

npx --yes @modelcontextprotocol/inspector --cli "${mcp_url}" --transport http \
    --method tools/call --tool-name run_command \
    --tool-args-json "${payload}" \
    --format json
