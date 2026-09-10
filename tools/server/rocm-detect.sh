#!/bin/sh
# Detect how this boot reaches the GPU and where its ROCm SDK should come
# from, then emit shell-assignable settings for docker-compose to consume.
#
# WHY THIS EXISTS: the two boots differ in ways no single compose file can
# express. WSL2 reaches the GPU through /dev/dxg and needs the host's
# dxg-aware HSA runtime plus the /usr/lib/wsl shims; a native boot uses
# /dev/kfd + /dev/dri and, on Mageia, has no host ROCm at all. So the device
# path and the SDK source are decided here, at invocation, rather than baked
# into a file that is then wrong on the other boot.
#
# USAGE
#   eval "$(./rocm-detect.sh)"                 # settings only, no pull
#   ./rocm-detect.sh --write-env .env          # merge into compose's .env
#   ./rocm-detect.sh --pull                    # also docker pull the image
#
# OUTPUT KEYS
#   ROCM_BOOT          wsl2 | native | nogpu
#   GPU_DEVICES        space-separated device nodes to pass through
#   ROCM_HOST_PATH     resolved host ROCm root, or empty
#   ROCM_HOST_VERSION  host ROCm release (e.g. 7.14.0), or empty
#   ROCM_HIP_TRIPLE    host HIP major.minor.patch, or empty
#   ROCM_SOURCE        host | image
#   ROCM_IMAGE         image reference when ROCM_SOURCE=image, else empty
#   COMPOSE_FILE       base compose file plus the boot's overlay
#
# Everything it prints is derived from something it checked. A value it could
# not determine is emitted EMPTY rather than guessed, because a wrong device
# node or a wrong image tag both fail in ways that look like a broken GPU.

set -eu

# SYSROOT is an INJECTION POINT, not a convenience: every path this script
# probes is a real device node or a real ROCm tree, so without it the
# branches below can only be exercised on a machine that already happens to
# be in the state under test -- which is a coincidence, not a test. Empty in
# production; a fixture directory under test.
SYSROOT="${SYSROOT:-}"

IMAGE_REPO="${IMAGE_REPO:-rocm/dev-ubuntu-24.04}"
# AMD publishes no thin dev image at 7.x -- 7.14 and 10.0 are "-full" only.
# See doc/core/HARDWARE.md before changing this suffix; the thin tags that
# exist are ROCm 6.4 and would skew against a 7.x host runtime.
IMAGE_SUFFIX="${IMAGE_SUFFIX:-full}"

DO_PULL=0
WRITE_ENV=""
while [ $# -gt 0 ]; do
    case "$1" in
        --pull) DO_PULL=1 ;;
        --write-env) WRITE_ENV="${2:?--write-env needs a path}"; shift ;;
        -h|--help) sed -n '2,30p' "$0"; exit 0 ;;
        *) echo "rocm-detect: unknown argument: $1" >&2; exit 2 ;;
    esac
    shift
done

note() { echo "rocm-detect: $*" >&2; }

# ---------------------------------------------------------------- boot ------
# Keyed on the DEVICE NODES rather than on a distro name or a uname string:
# the node is what the container actually needs passed through, and it is
# present or absent as a fact. /dev/dxg and /dev/kfd are mutually exclusive
# in practice, but check dxg first -- under WSL2 a stray kfd would be wrong.
if [ -e "$SYSROOT/dev/dxg" ]; then
    ROCM_BOOT=wsl2
    GPU_DEVICES=/dev/dxg
elif [ -e "$SYSROOT/dev/kfd" ] && [ -d "$SYSROOT/dev/dri" ]; then
    ROCM_BOOT=native
    GPU_DEVICES="/dev/kfd /dev/dri"
else
    ROCM_BOOT=nogpu
    GPU_DEVICES=""
    note "no /dev/dxg and no /dev/kfd+/dev/dri -- GPU stages will skip"
fi

# ------------------------------------------------------- host ROCm ----------
# Resolved, not assumed. /opt/rocm's bin/lib/include may be
# update-alternatives symlinks into /etc/alternatives, which a container does
# not mount, so the canonical path can exist and still be unusable from
# inside. Take the first candidate that actually holds a real bin/hipcc.
ROCM_HOST_PATH=""
for d in "$SYSROOT"/opt/rocm "$SYSROOT"/opt/rocm/core-* "$SYSROOT"/opt/rocm-*; do
    if [ -x "$d/bin/hipcc" ]; then ROCM_HOST_PATH="$d"; break; fi
done

# Two DIFFERENT version representations, and they are not interchangeable:
# .info/version is the ROCm release (7.14.0) and is what an image tag is
# named after; share/hip/version carries the HIP triple (7.14.60850) and is
# what core/Makefile's hip-header-check compares. Emit both.
ROCM_HOST_VERSION=""
ROCM_HIP_TRIPLE=""
if [ -n "$ROCM_HOST_PATH" ]; then
    if [ -f "$ROCM_HOST_PATH/.info/version" ]; then
        ROCM_HOST_VERSION=$(tr -d ' \t' < "$ROCM_HOST_PATH/.info/version" \
            | head -n1)
    fi
    if [ -f "$ROCM_HOST_PATH/share/hip/version" ]; then
        ROCM_HIP_TRIPLE=$(sed -n \
            's/^HIP_VERSION_\(MAJOR\|MINOR\|PATCH\)=\([0-9]*\)$/\2/p' \
            "$ROCM_HOST_PATH/share/hip/version" | tr '\n' '.' \
            | sed 's/\.$//')
    fi
fi

# ------------------------------------------------------- SDK source ---------
# A host ROCm is PREFERRED where one exists, and this is not laziness: under
# WSL2 the host's HSA runtime is a dxg-aware build (it references /dev/dxg,
# HSA_ENABLE_DXG_DETECTION and librocdxg), and AMD's generic images are built
# for /dev/kfd. Executing against an image runtime there would not reach the
# GPU at all. Where there is no host ROCm -- the Mageia case -- the image is
# the whole SDK and defines the version.
if [ -n "$ROCM_HOST_PATH" ]; then
    ROCM_SOURCE=host
    ROCM_IMAGE=""
else
    ROCM_SOURCE=image
    if [ -n "${ROCM_IMAGE_TAG:-}" ]; then
        ROCM_IMAGE="$IMAGE_REPO:$ROCM_IMAGE_TAG"
    else
        # No host ROCm means no version to match, so the tag must be stated.
        # Refuse to invent one: a wrong tag is a multi-gigabyte pull that
        # then fails, and "latest" silently changes ROCm major versions.
        ROCM_IMAGE=""
        note "ROCM_SOURCE=image but no ROCM_IMAGE_TAG given and no host"
        note "ROCm to match -- set ROCM_IMAGE_TAG (e.g. 7.14.0-$IMAGE_SUFFIX)"
    fi
fi

# When a host ROCm DOES exist and an image is wanted anyway, match the tag to
# it so headers and runtime cannot skew. Callers opt in with ROCM_PREFER_IMAGE.
if [ "${ROCM_PREFER_IMAGE:-0}" = "1" ] && [ -n "$ROCM_HOST_VERSION" ]; then
    ROCM_SOURCE=image
    ROCM_IMAGE="$IMAGE_REPO:$ROCM_HOST_VERSION-$IMAGE_SUFFIX"
    if [ "$ROCM_BOOT" = "wsl2" ]; then
        note "WARNING: boot is wsl2 and an image SDK was requested."
        note "The image's HSA runtime targets /dev/kfd and will NOT reach"
        note "the GPU here. Use the image for HEADERS only and keep the"
        note "host's dxg-aware runtime for execution."
    fi
fi

# ------------------------------------------------------- compose values ----
# The keys above describe the boot; these are what docker-compose.yml
# actually interpolates. They are derived here rather than branched on in
# compose, which has no conditional syntax -- see that file's header.
#
# GPU_DEV_* are two FIXED SLOTS because a compose list cannot grow or shrink.
# An unused slot gets a real harmless node, and the two are DIFFERENT nodes
# on purpose: the same device twice is a duplicate entry.
#
# ROCM_MOUNT_SRC and WSL_LIB_SRC exploit compose short-syntax: a source with
# a leading slash is a bind mount, one without is a named volume. So the same
# compose line binds the host's ROCm on one boot and mounts a Docker-managed
# volume on the other, with no branch anywhere.
case "$ROCM_BOOT" in
    wsl2)
        GPU_DEV_1=/dev/dxg
        GPU_DEV_2=/dev/zero
        WSL_LIB_SRC=/usr/lib/wsl
        ;;
    native)
        GPU_DEV_1=/dev/kfd
        GPU_DEV_2=/dev/dri
        WSL_LIB_SRC=wsl-lib-unused
        ;;
    *)
        GPU_DEV_1=/dev/null
        GPU_DEV_2=/dev/zero
        WSL_LIB_SRC=wsl-lib-unused
        ;;
esac

if [ "$ROCM_SOURCE" = host ]; then
    # Bind the whole /opt/rocm so the container sees the same layout the host
    # has, including the versioned core-* directory the loader path points
    # into. ROCM_HOST_PATH is already the resolved real directory, and the
    # mount is 1:1 at /opt/rocm, so it is valid inside the container too.
    ROCM_MOUNT_SRC=/opt/rocm
    ROCM_BIN_PATH="$ROCM_HOST_PATH/bin"
    ROCM_LIB_PATH="$ROCM_HOST_PATH/lib"
    # WSL2 additionally needs the host's WSL GPU userspace on the loader
    # path; natively that directory does not exist and must not appear.
    [ "$ROCM_BOOT" = wsl2 ] && \
        ROCM_LIB_PATH="$ROCM_LIB_PATH:/usr/lib/wsl/lib"
    COMPOSE_PROFILES=""
else
    # Image-sourced: a named volume the rocm-sdk service seeds, and AMD's
    # images use a plain /opt/rocm layout with no alternatives indirection.
    ROCM_MOUNT_SRC=rocm-sdk
    ROCM_BIN_PATH=/opt/rocm/bin
    ROCM_LIB_PATH=/opt/rocm/lib
    COMPOSE_PROFILES="rocm-image"
fi

# ------------------------------------------------------- compose overlay ----
# ONE compose file covers both boots -- every difference is a variable value
# above, and the single genuinely-conditional piece (the rocm-sdk seeder) is
# a compose profile. So there is no overlay to select and COMPOSE_FILE is
# emitted only for callers that want it stated explicitly.
COMPOSE_FILE="docker-compose.yml"

# ------------------------------------------------------- pull ---------------
# Preflight the disk before a multi-gigabyte pull rather than discovering it
# full. The compressed size roughly doubles to two-and-a-half times on
# extraction, and the daemon holds blob and snapshot at once, so require 3x.
if [ "$DO_PULL" = "1" ]; then
    if [ -z "$ROCM_IMAGE" ]; then
        note "nothing to pull (ROCM_SOURCE=$ROCM_SOURCE, no image resolved)"
    elif ! command -v docker >/dev/null 2>&1; then
        note "docker not on PATH -- cannot pull"
        exit 1
    elif docker image inspect "$ROCM_IMAGE" >/dev/null 2>&1; then
        note "$ROCM_IMAGE already present -- not pulling"
    else
        avail=$(df -Pk /var/lib/docker 2>/dev/null || df -Pk /)
        avail_gib=$(echo "$avail" | awk 'NR==2 {printf "%d", $4/1048576}')
        note "pulling $ROCM_IMAGE ($avail_gib GiB free where images live)"
        note "7.x tags are 7.4-7.7 GiB COMPRESSED, so budget ~20 GiB"
        if [ "${avail_gib:-0}" -lt 20 ]; then
            note "REFUSING: under 20 GiB free. Free space or set"
            note "ROCM_PULL_ANYWAY=1 to override."
            [ "${ROCM_PULL_ANYWAY:-0}" = "1" ] || exit 1
        fi
        docker pull "$ROCM_IMAGE"
    fi
fi

# ------------------------------------------------------- emit ---------------
emit() {
    cat <<EOF
ROCM_BOOT=$ROCM_BOOT
GPU_DEVICES=$GPU_DEVICES
ROCM_HOST_PATH=$ROCM_HOST_PATH
ROCM_HOST_VERSION=$ROCM_HOST_VERSION
ROCM_HIP_TRIPLE=$ROCM_HIP_TRIPLE
ROCM_SOURCE=$ROCM_SOURCE
ROCM_IMAGE=$ROCM_IMAGE
COMPOSE_FILE=$COMPOSE_FILE
COMPOSE_PROFILES=$COMPOSE_PROFILES
GPU_DEV_1=$GPU_DEV_1
GPU_DEV_2=$GPU_DEV_2
ROCM_MOUNT_SRC=$ROCM_MOUNT_SRC
WSL_LIB_SRC=$WSL_LIB_SRC
ROCM_LIB_PATH=$ROCM_LIB_PATH
ROCM_BIN_PATH=$ROCM_BIN_PATH
EOF
}

if [ -n "$WRITE_ENV" ]; then
    # Replace only the keys this script owns; anything else in the .env is
    # the project's and must survive. Written via a temp file and moved, so
    # an interrupted run cannot leave a half-written .env behind.
    tmp="$WRITE_ENV.rocm-detect.$$"
    if [ -f "$WRITE_ENV" ]; then
        grep -vE '^(ROCM_BOOT|GPU_DEVICES|ROCM_HOST_PATH|ROCM_HOST_VERSION|ROCM_HIP_TRIPLE|ROCM_SOURCE|ROCM_IMAGE|COMPOSE_FILE|COMPOSE_PROFILES|GPU_DEV_1|GPU_DEV_2|ROCM_MOUNT_SRC|WSL_LIB_SRC|ROCM_LIB_PATH|ROCM_BIN_PATH)=' \
            "$WRITE_ENV" > "$tmp" || :
    else
        : > "$tmp"
    fi
    emit >> "$tmp"
    mv "$tmp" "$WRITE_ENV"
    note "wrote $(emit | wc -l) settings into $WRITE_ENV"
else
    emit
fi
