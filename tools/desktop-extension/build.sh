#!/usr/bin/env bash
# Packs tools/desktop-extension/ (index.js, manifest.json,
# node_modules/) into dist/locusai-local-bash.mcpb via the published
# @anthropic-ai/mcpb CLI. This was a one-off manual command (see
# tools/README.md) until now -- scripted for the same reason
# tools/server/start.sh replaced its own manual steps: it can't drift
# from what's documented, and isn't retyped from memory each time.
#
# Installs node_modules before packing (once npm-installed, npm skips
# reinstalling on subsequent runs unless package.json/lockfile
# changed) -- mcp-remote must actually be present on disk here for
# `mcpb pack` to bundle it, matching MCPB's own guidance to ship
# dependencies rather than fetch them at launch time. See index.js
# for why that matters.
#
# Safe to re-run repeatedly: the CLI excludes dist/'s own prior output
# from being swept into a new pack (confirmed in the session that
# built this extension, not assumed).
set -euo pipefail

if ! command -v npm &>/dev/null; then
    echo "error: 'npm' not found -- install Node.js/npm first" >&2
    exit 1
fi
if ! command -v npx &>/dev/null; then
    echo "error: 'npx' not found -- install Node.js/npm first" >&2
    exit 1
fi

script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "${script_dir}"

echo "Installing dependencies..." >&2
npm install

output="dist/locusai-local-bash.mcpb"
mkdir -p "$(dirname "${output}")"

echo "Packing ${output}..." >&2
npx --yes @anthropic-ai/mcpb pack . "${output}"
echo "Built ${output}" >&2
