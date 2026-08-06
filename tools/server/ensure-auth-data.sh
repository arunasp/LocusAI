#!/usr/bin/env bash
# Compose-path pre-flight, mirroring opencode-model-eval's
# ensure-auth-data.sh -- docker-compose has no precondition hook, so
# this runs before `build`/`up` instead (wired into start.sh). Two
# responsibilities:
#   1. Defend against Docker's phantom-mount bug: if a bind-mount
#      source path doesn't exist yet, Docker creates an empty
#      directory there (as root) rather than erroring -- confirmed as
#      a real, recurring issue on this same host in
#      opencode-model-eval, for the exact same class of bind mount.
#      Applies here to ~/.ssh/github (the ssh deploy-key mount
#      source).
#   2. If tools/server/.env doesn't exist yet (GH_TOKEN unset), run
#      extract-github-token.sh automatically rather than failing
#      opaquely inside the container later.
#
# Usage:
#   bash tools/server/ensure-auth-data.sh
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
readonly SCRIPT_DIR
readonly SSH_KEY_SOURCE="${HOME}/.ssh/github"
readonly ENV_FILE="${SCRIPT_DIR}/.env"
readonly EXTRACT_SCRIPT="${SCRIPT_DIR}/extract-github-token.sh"

if [ -d "${SSH_KEY_SOURCE}" ]; then
    echo "ensure-auth-data.sh: ${SSH_KEY_SOURCE} is a directory -- this is Docker's" >&2
    echo "  phantom-mount bug (a bind-mount source that didn't exist yet), not a" >&2
    echo "  real ssh key. Removing it via rmdir (refuses if non-empty, so this" >&2
    echo "  only ever removes an empty phantom, never real content)." >&2

    if ! rmdir_err="$(rmdir "${SSH_KEY_SOURCE}" 2>&1)"; then
        if echo "${rmdir_err}" | grep -qi "permission denied"; then
            echo "ensure-auth-data.sh: permission denied -- created by the Docker" >&2
            echo "  daemon (runs as root), so removing it needs root too. Retrying" >&2
            echo "  with sudo (will prompt for your password)..." >&2
            if ! sudo rmdir "${SSH_KEY_SOURCE}"; then
                echo "ensure-auth-data.sh: rmdir still failed even with sudo. Inspect" >&2
                echo "  yourself: sudo ls -la ${SSH_KEY_SOURCE}" >&2
                exit 1
            fi
        else
            echo "ensure-auth-data.sh: rmdir failed for a reason other than permissions:" >&2
            echo "  ${rmdir_err}" >&2
            echo "  This probably ISN'T the phantom-mount case. Not deleting anything" >&2
            echo "  automatically. Inspect yourself: ls -la ${SSH_KEY_SOURCE}" >&2
            exit 1
        fi
    fi
    echo "ensure-auth-data.sh: phantom directory removed -- it was empty, so no" >&2
    echo "  real key was ever there. Put your real deploy key at" >&2
    echo "  ${SSH_KEY_SOURCE} before continuing." >&2
    exit 1
fi

if [ ! -f "${SSH_KEY_SOURCE}" ]; then
    echo "ensure-auth-data.sh: no ssh key found at ${SSH_KEY_SOURCE} --" >&2
    echo "  git push/pull over ssh won't work until it's there. Continuing" >&2
    echo "  anyway (gh/GH_TOKEN below is independent of this)." >&2
fi

if [ -f "${ENV_FILE}" ]; then
    echo "ensure-auth-data.sh: ${ENV_FILE} already exists -- leaving it as-is." >&2
    echo "  (delete it yourself first to re-extract a fresh GH_TOKEN.)" >&2
else
    echo "ensure-auth-data.sh: ${ENV_FILE} not found -- extracting GH_TOKEN now..." >&2
    "${EXTRACT_SCRIPT}"
fi
