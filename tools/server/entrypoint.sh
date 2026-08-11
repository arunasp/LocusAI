#!/bin/sh
# Run the server as the invoking host user so that files it writes to the
# bind-mounted project directory are owned by that user, not root.
#
# Without this the container runs as its image default (root), and anything
# it builds -- build/liblocus.so, build/gpu_probe, core/.venv -- lands
# root-owned on the host bind mount. cicd_runner's worker already solves
# this the same way; this brings the project container onto that pattern
# rather than special-casing it.
#
# Two independent failure causes are handled here, and fixing only one
# looks like progress while the other still bites:
#
# 1. MISSING PASSWD ENTRY. An image is built without knowledge of the real
#    host's uid, so the kernel cannot resolve it to a user record and any
#    execve fails with EAGAIN -- not a permissions error, which is what
#    makes it confusing. Created dynamically below when absent.
#
# 2. RLIMIT_NPROC. Handled in docker-compose.yml, not here, because it is a
#    daemon-side limit rather than an image concern. After a set*uid() call
#    the kernel checks the NEW real uid's nproc limit, which is enforced
#    per-uid SYSTEM-WIDE -- it counts every process already running as that
#    uid on the host, not just those in the container. Docker's default can
#    be as low as 128, which an ordinary desktop session exceeds on its own.
#    The symptom is identical EAGAIN, so both must be fixed together.
#
# setpriv rather than su/sudo/gosu: it is part of util-linux and already
# present in a standard Debian image, and --reset-env re-initialises HOME,
# SHELL, USER, LOGNAME and PATH from the target uid's own passwd entry
# instead of leaving root's environment behind under a different uid.
set -e

if [ -n "${TARGET_UID:-}" ] && [ -n "${TARGET_GID:-}" ]; then
    if ! getent passwd "${TARGET_UID}" >/dev/null 2>&1; then
        if ! getent group "${TARGET_GID}" >/dev/null 2>&1; then
            addgroup --gid "${TARGET_GID}" worker
        fi
        adduser --uid "${TARGET_UID}" --gid "${TARGET_GID}" \
            --home /home/worker --shell /bin/sh \
            --disabled-password --gecos "" worker >/dev/null 2>&1
        # adduser does not chown a home directory that already exists, so
        # chown unconditionally rather than relying on it.
        chown "${TARGET_UID}:${TARGET_GID}" /home/worker
    fi

    # --reset-env re-initialises the environment from the target uid's own
    # passwd entry, which is what makes the dropped user's shell properly
    # initialised -- and which also DISCARDS anything set in
    # docker-compose.yml. That silently removed the GPU setup: PATH lost
    # its /opt/rocm/bin prefix and LD_LIBRARY_PATH disappeared entirely,
    # so hipcc became unreachable even though it was mounted and present.
    # Re-apply exactly those two afterwards via env, so setpriv still
    # supplies HOME, USER, LOGNAME and SHELL from the passwd entry and
    # only the two the container genuinely needs are carried across.
    gpu_path="${PATH}"
    gpu_ld="${LD_LIBRARY_PATH:-}"

    if [ -n "${gpu_ld}" ]; then
        exec setpriv --reuid="${TARGET_UID}" --regid="${TARGET_GID}" \
            --clear-groups --reset-env \
            env PATH="${gpu_path}" LD_LIBRARY_PATH="${gpu_ld}" "$@"
    fi
    # Deliberately not exported when empty: an empty LD_LIBRARY_PATH is
    # treated by the loader as the current directory, which is worse than
    # having it unset.
    exec setpriv --reuid="${TARGET_UID}" --regid="${TARGET_GID}" \
        --clear-groups --reset-env env PATH="${gpu_path}" "$@"
fi

exec "$@"
