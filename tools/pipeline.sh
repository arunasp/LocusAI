#!/usr/bin/env bash
# Named-stage pipeline runner for LocusAI's local-bash MCP tooling.
#
# This gives the repo CI/CD-style *terminology* (independently
# runnable lint/test/build/server/verify stages, each idempotent, each
# reporting its own pass/fail/skip) without an actual CI/CD *runner*:
# the stages that matter most (packing the real .mcpb, starting the
# real Docker container, confirming a real Claude Desktop install
# actually connects) only mean anything on the machine that has
# Claude Desktop + WSL2 + Docker running -- see tools/README.md's
# Architecture section. A cloud CI runner can't reach any of that.
#
# `lint` and `test` are environment-agnostic (no Docker/Desktop
# needed) and are meant to run wherever development is actually
# happening -- run them there before anything is pushed to this
# machine. `build`, `server`, and `verify` need this real local
# machine and are meant to run here, as the staging step after that
# development.
#
# UPDATE 2026-08-08: `build`/`server`/`verify` no longer strictly
# require running this by hand on the host. ~/stuff/cicd_runner is a
# separate, project-agnostic container with real docker-socket access
# (deliberately NOT this project's own container -- a project can't
# safely rebuild itself, see cicd_runner's own README for why) that
# can run `run_command(project="LocusAI", binary="bash", args=["tools/
# pipeline.sh", ...])` directly. `bash` would need adding to
# cicd_runner's own allowlist.txt for that exact form. Still true:
# this file itself has no docker/npm/npx on ITS OWN allowlist, and
# that's staying that way -- cicd_runner is the answer to "how do
# these stages run without a human", not a reason to widen this
# project's own container.
#
# Usage: tools/pipeline.sh <stage> [stage...]
#   lint    - shellcheck every .sh, `node --check` every .js
#             (excluding node_modules/dist), `python3 -m py_compile`
#             every .py (excluding node_modules) under tools/. No
#             side effects. Skips a check silently-as-SKIP (not a
#             failure) if the relevant tool isn't on PATH.
#   test    - runs tools/desktop-extension/test's automated suite
#             (`node --test`). No Docker/Desktop needed.
#   build   - delegates to tools/desktop-extension/build.sh (npm
#             install + mcpb pack -> dist/locusai-local-bash.mcpb).
#             Needs npm/npx.
#   server  - delegates to tools/server/start.sh (docker compose
#             build + up -d), then confirms the endpoint responds.
#             Needs a real Docker daemon.
#   verify  - curl's the running server's /mcp endpoint for the
#             expected "must accept text/event-stream" JSON-RPC error
#             (see tools/README.md -- that response IS the success
#             signal), and if npx is available, makes one real
#             run_command call (`ls`) through tools/server/call.sh.
#             Needs the server stage to have already succeeded.
#   all     - runs every stage above in order. A stage that can't run
#             in this environment (missing docker/npm/npx) is reported
#             SKIPPED with the reason, not silently dropped and not a
#             hard failure -- lint/test failures ARE hard failures.
#
# Each stage prints its own [PASS]/[FAIL]/[SKIP] line to stderr.
# Exits 0 only if nothing FAILED (SKIPPED stages don't count against
# that); exits 1 if any requested stage failed.
set -euo pipefail

script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
repo_root="$(cd "${script_dir}/.." && pwd)"

# Stage functions return 0 (pass), 1 (fail), or 2 (skip -- a
# prerequisite tool is missing, not a defect in the code being
# checked). run_stage() turns that into a labeled log line and
# tracks failures for the final summary.
declare -a FAILED_STAGES=()

run_stage() {
    local name="$1"
    shift
    local rc=0
    "$@" || rc=$?
    case "${rc}" in
        0) echo "[PASS] ${name}" >&2 ;;
        2) echo "[SKIP] ${name}" >&2 ;;
        *) echo "[FAIL] ${name}" >&2; FAILED_STAGES+=("${name}") ;;
    esac
    return 0
}

stage_lint() {
    local had_failure=0
    local ran_any=0

    if command -v shellcheck &>/dev/null; then
        local sh_files=()
        while IFS= read -r -d '' f; do
            sh_files+=("${f}")
        done < <(find "${repo_root}" -name '*.sh' -not -path '*/node_modules/*' -not -path '*/.git/*' -print0)
        if [[ ${#sh_files[@]} -gt 0 ]]; then
            ran_any=1
            shellcheck -x -P SCRIPTDIR "${sh_files[@]}" || had_failure=1
        fi
    else
        echo "  (shellcheck not found on PATH -- shell scripts not linted)" >&2
    fi

    if command -v node &>/dev/null; then
        local js_files=()
        while IFS= read -r -d '' f; do
            js_files+=("${f}")
        done < <(find "${repo_root}/tools" -name '*.js' -not -path '*/node_modules/*' -not -path '*/dist/*' -print0)
        for f in "${js_files[@]}"; do
            ran_any=1
            node --check "${f}" || had_failure=1
        done
    else
        echo "  (node not found on PATH -- .js files not syntax-checked)" >&2
    fi

    if command -v python3 &>/dev/null; then
        local py_files=()
        while IFS= read -r -d '' f; do
            py_files+=("${f}")
        done < <(find "${repo_root}/tools/server" -name '*.py' -not -path '*/node_modules/*' -print0)
        for f in "${py_files[@]}"; do
            ran_any=1
            python3 -m py_compile "${f}" || had_failure=1
        done
    else
        echo "  (python3 not found on PATH -- .py files not syntax-checked)" >&2
    fi

    if [[ "${had_failure}" -eq 1 ]]; then
        return 1
    fi
    if [[ "${ran_any}" -eq 0 ]]; then
        return 2
    fi
    return 0
}

stage_test() {
    if ! command -v node &>/dev/null; then
        echo "  (node not found on PATH -- cannot run the test suite)" >&2
        return 2
    fi
    local test_dir="${repo_root}/tools/desktop-extension/test"
    if [[ ! -d "${test_dir}" ]]; then
        echo "  (${test_dir} does not exist -- nothing to run)" >&2
        return 2
    fi
    # `node --test <directory>` does NOT reliably recurse into a
    # non-default-named directory passed explicitly on the command
    # line (confirmed by direct testing against Node 22.22.2 during
    # this pipeline's own construction: it tried to require() the
    # directory itself as a single test file and failed with
    # MODULE_NOT_FOUND, even though `node --test` with no path
    # correctly auto-discovers the same files from cwd). Passing the
    # matched *.test.js files explicitly sidesteps that entirely.
    local test_files=()
    while IFS= read -r -d '' f; do
        test_files+=("${f}")
    done < <(find "${test_dir}" -name '*.test.js' -print0)
    if [[ ${#test_files[@]} -eq 0 ]]; then
        echo "  (no *.test.js files under ${test_dir})" >&2
        return 2
    fi
    node --test "${test_files[@]}"
}

stage_build() {
    if ! command -v npm &>/dev/null || ! command -v npx &>/dev/null; then
        echo "  (npm/npx not found on PATH -- cannot pack the .mcpb)" >&2
        return 2
    fi
    "${repo_root}/tools/desktop-extension/build.sh"
}

stage_server() {
    if ! command -v docker &>/dev/null; then
        echo "  (docker not found on PATH -- cannot start the server)" >&2
        return 2
    fi
    "${repo_root}/tools/server/start.sh"
    stage_verify_endpoint
}

stage_verify_endpoint() {
    if ! command -v curl &>/dev/null; then
        echo "  (curl not found on PATH -- cannot confirm the endpoint)" >&2
        return 2
    fi
    # A plain GET with no `Accept: text/event-stream` deliberately gets
    # back HTTP 406 with a JSON-RPC "Not Acceptable" error body (see
    # tools/README.md) -- that response IS the success signal: the
    # server is up and enforcing the streamable-HTTP protocol
    # correctly. `curl --fail` was wrong here -- confirmed live
    # 2026-08-05: it treats any HTTP >=400 as failure, so it flagged
    # this exact documented-correct response as an error
    # (`curl: (22) ... returned error: 406`) on every real run against
    # this repo's own checkout (the first time pipeline.sh all was
    # actually run here -- see README's former "What was NOT verified"
    # note). Check the real status/body instead of relying on --fail.
    #
    # `docker compose up -d` also returns before the app inside has
    # finished binding the socket, so this polls for up to ~10s rather
    # than firing once.
    local attempt response http_code body
    for attempt in $(seq 1 10); do
        response=""
        response="$(curl --silent --max-time 2 --write-out '\n%{http_code}' http://localhost:1443/mcp 2>/dev/null)" || response=""
        http_code="${response##*$'\n'}"
        body="${response%$'\n'*}"
        if [[ "${http_code}" == "406" ]] && grep -q "text/event-stream" <<< "${body}"; then
            return 0
        fi
        echo "  (endpoint not ready yet, attempt ${attempt}/10)" >&2
        sleep 1
    done
    echo "  (final response: http_code=${http_code:-<none>} body=${body:-<none>})" >&2
    return 1
}

stage_verify() {
    stage_verify_endpoint || return "$?"
    if ! command -v npx &>/dev/null; then
        echo "  (npx not found on PATH -- endpoint confirmed, but skipped a real tool call)" >&2
        return 2
    fi
    "${repo_root}/tools/server/call.sh" ls -la >/dev/null
}

usage() {
    echo "Usage: $0 <stage> [stage...]" >&2
    echo "Stages: lint test build server verify all" >&2
}

if [[ $# -eq 0 ]]; then
    usage
    exit 1
fi

stages=("$@")
if [[ "${stages[0]}" == "all" ]]; then
    stages=(lint test build server verify)
fi

# Log every run to logs/ (gitignored) so it can be fetched directly --
# via LocusAI Local Bash's cat/find/tail, or by hand -- instead of
# requiring output to be pasted back.
#
# Deliberately NOT wrapping the stage loop below in a function piped
# through tee (`run_pipeline() { ... }; run_pipeline | tee ...`) --
# tried that first, but it broke ShellCheck's reachability tracing for
# every single stage_* function (SC2317 "unreachable", confirmed via a
# minimal repro: the exact same `run_stage lint stage_lint` indirect
# dispatch is traced fine by ShellCheck at top level, but not once
# it's nested inside another function or brace-group that's the left
# side of a pipe). Keeping the loop at top level avoids that
# regression entirely.
#
# Using `exec > >(tee -a ...)` here instead needs one extra step to be
# correct: that form is normally racy on its own -- the background tee
# reading the process-substitution pipe can still be flushing when the
# script exits, so a `cat` run immediately after (e.g. by Claude via
# the allowlisted binaries) can read a stale/incomplete file. Fixed by
# explicitly closing our own stdout/stderr at the end (so tee sees EOF)
# and waiting for its PID before this script actually exits -- verified
# with 20 stress-test iterations with no incomplete log observed,
# alongside a normal ShellCheck pass, before relying on it here.
log_dir="${repo_root}/logs"
mkdir -p "${log_dir}"
stages_joined="$(IFS=-; echo "${stages[*]}")"
log_file="${log_dir}/$(date -u +%Y%m%dT%H%M%SZ)-${stages_joined}.log"
echo "Logging this run to ${log_file}" >&2
exec > >(tee -a "${log_file}")
tee_pid=$!
exec 2>&1

for stage in "${stages[@]}"; do
    case "${stage}" in
        lint) run_stage lint stage_lint ;;
        test) run_stage test stage_test ;;
        build) run_stage build stage_build ;;
        server) run_stage server stage_server ;;
        verify) run_stage verify stage_verify ;;
        *)
            echo "error: unknown stage '${stage}'" >&2
            usage
            exit 1
            ;;
    esac
done

if [[ ${#FAILED_STAGES[@]} -gt 0 ]]; then
    echo "RESULT: FAILED -- ${FAILED_STAGES[*]}" >&2
    # Drain and reap tee before exiting, so the log file is guaranteed
    # complete the instant this script returns -- see the note above.
    exec 1>&- 2>&-
    wait "${tee_pid}" 2>/dev/null || true
    exit 1
fi
echo "RESULT: ALL PASS (or SKIPPED where a required tool is missing)" >&2

# Same drain/reap, success path. Deliberately NOT followed by an
# explicit `exit 0` -- confirmed live 2026-08-06: ShellCheck 0.10.0
# treats a bare, unconditional `exit N` as the script's literal final
# statement as breaking its own reachability tracing for every
# function only ever invoked through run_stage's indirect `"$@"`
# dispatch (spurious SC2317 "unreachable" on all of them) -- isolated
# via a minimal repro after ruling out the tee/process-substitution
# machinery itself as the cause. Falling through naturally (implicit
# exit 0, matching this script's original pre-logging structure)
# avoids it entirely; only the failure path above still needs a real
# explicit exit, and being inside a conditional there doesn't trigger
# the same issue.
exec 1>&- 2>&-
wait "${tee_pid}" 2>/dev/null || true
