#!/usr/bin/env bash
# Sourceable function: resolve_project_dir
#
# Resolves the project directory to mount at /workspace by asking git,
# not by reading a hardcoded path from a config file. This is
# location-independent: it works regardless of where this repo is
# cloned, what it's renamed to, or which directory the caller invokes
# from, because it starts from THIS script's own location rather than
# the caller's $PWD.
#
# Deliberately does not `set -e`/`set -u` here: this file is sourced,
# and changing shell options on the caller's behalf is a footgun for
# whatever sources it. The caller sets its own options.

resolve_project_dir() {
    local script_dir
    script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

    local repo_root
    if ! repo_root="$(git -C "${script_dir}" rev-parse --show-toplevel 2>/dev/null)"; then
        echo "resolve_project_dir: ${script_dir} is not inside a git repository" >&2
        return 1
    fi

    printf '%s\n' "${repo_root}"
}
