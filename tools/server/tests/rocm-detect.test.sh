#!/bin/sh
# Branch verification for rocm-detect.sh. One driver, one report.
pass=0; fail=0
mkfix() { # $1=dir $2=boot(dxg|kfd|none) $3=rocm(yes|no)
  rm -rf "$1"; mkdir -p "$1/dev" "$1/opt"
  case "$2" in
    dxg) : > "$1/dev/dxg" ;;
    kfd) : > "$1/dev/kfd"; mkdir -p "$1/dev/dri" ;;
  esac
  if [ "$3" = yes ]; then
    mkdir -p "$1/opt/rocm/core-7.14/bin" "$1/opt/rocm/core-7.14/.info" "$1/opt/rocm/core-7.14/share/hip"
    printf '#!/bin/sh\n' > "$1/opt/rocm/core-7.14/bin/hipcc"; chmod +x "$1/opt/rocm/core-7.14/bin/hipcc"
    echo '7.14.0' > "$1/opt/rocm/core-7.14/.info/version"
    printf 'HIP_VERSION_MAJOR=7\nHIP_VERSION_MINOR=14\nHIP_VERSION_PATCH=60850\n' > "$1/opt/rocm/core-7.14/share/hip/version"
  fi
}
check() { # $1=name $2=out $3=needle
  if printf '%s' "$2" | grep -qF -- "$3"; then pass=$((pass+1)); echo "  [PASS] $1 :: $3"
  else fail=$((fail+1)); echo "  [FAIL] $1 :: expected '$3'"; fi
}
# Exact whole-line match. Needed for any assertion about an EMPTY value:
# a substring test for 'COMPOSE_PROFILES=' matches every non-empty value too,
# so it would pass while proving nothing.
checkx() { # $1=name $2=out $3=exact line
  if printf '%s\n' "$2" | grep -qxF -- "$3"; then pass=$((pass+1)); echo "  [PASS] $1 :: [$3]"
  else fail=$((fail+1)); echo "  [FAIL] $1 :: expected exact line '$3'"; fi
}
run() { SYSROOT="$1" sh -c "$2 ./rocm-detect.sh" 2>&1; }

mkfix /tmp/f1 dxg yes;  O=$(run /tmp/f1 "")
check "wsl2 + host rocm" "$O" "ROCM_BOOT=wsl2"
check "wsl2 + host rocm" "$O" "GPU_DEVICES=/dev/dxg"
check "wsl2 + host rocm" "$O" "ROCM_SOURCE=host"
check "wsl2 + host rocm" "$O" "ROCM_HOST_VERSION=7.14.0"
check "wsl2 + host rocm" "$O" "ROCM_HIP_TRIPLE=7.14.60850"
# ONE compose file for both boots: every difference is a variable value, so
# there is no overlay to select. These are the keys docker-compose.yml
# actually interpolates.
checkx "wsl2 single compose file" "$O" "COMPOSE_FILE=docker-compose.yml"
checkx "wsl2 no seeder profile" "$O" "COMPOSE_PROFILES="
checkx "wsl2 device slot 1" "$O" "GPU_DEV_1=/dev/dxg"
checkx "wsl2 unused device slot" "$O" "GPU_DEV_2=/dev/zero"
checkx "wsl2 binds host rocm" "$O" "ROCM_MOUNT_SRC=/opt/rocm"
checkx "wsl2 binds wsl userspace" "$O" "WSL_LIB_SRC=/usr/lib/wsl"
check  "wsl2 loader path has wsl lib" "$O" "ROCM_LIB_PATH=/tmp/f1/opt/rocm/core-7.14/lib:/usr/lib/wsl/lib"

mkfix /tmp/f2 kfd yes;  O=$(run /tmp/f2 "")
check "native + host rocm" "$O" "ROCM_BOOT=native"
check "native + host rocm" "$O" "GPU_DEVICES=/dev/kfd /dev/dri"
checkx "native device slots" "$O" "GPU_DEV_1=/dev/kfd"
checkx "native device slots" "$O" "GPU_DEV_2=/dev/dri"
# The wsl tree has no meaning natively, so the mount source becomes an empty
# NAMED VOLUME rather than a bind of a path docker would create on the host.
checkx "native wsl mount is a volume" "$O" "WSL_LIB_SRC=wsl-lib-unused"
# Host ROCm present, so no seeder and no image even on a native boot.
checkx "native+host needs no profile" "$O" "COMPOSE_PROFILES="
check  "native loader path omits wsl" "$O" "ROCM_LIB_PATH=/tmp/f2/opt/rocm/core-7.14/lib"

mkfix /tmp/f3 kfd no;   O=$(run /tmp/f3 "")
check "ABSENT host rocm (Mageia)" "$O" "ROCM_SOURCE=image"
check "ABSENT host rocm (Mageia)" "$O" "ROCM_HOST_PATH="
check "ABSENT host rocm (Mageia)" "$O" "no ROCM_IMAGE_TAG given"
check "ABSENT host rocm (Mageia)" "$O" "ROCM_IMAGE="
# No host ROCm: the SDK comes from a named volume the seeder populates, so
# the profile that defines the seeder must be active.
checkx "Mageia sdk from volume" "$O" "ROCM_MOUNT_SRC=rocm-sdk"
checkx "Mageia activates seeder" "$O" "COMPOSE_PROFILES=rocm-image"
checkx "Mageia plain rocm layout" "$O" "ROCM_LIB_PATH=/opt/rocm/lib"

O=$(run /tmp/f3 "ROCM_IMAGE_TAG=7.14.0-full")
check "Mageia + explicit tag" "$O" "ROCM_IMAGE=rocm/dev-ubuntu-24.04:7.14.0-full"

O=$(run /tmp/f1 "ROCM_PREFER_IMAGE=1")
check "wsl2 + prefer-image" "$O" "ROCM_IMAGE=rocm/dev-ubuntu-24.04:7.14.0-full"
check "wsl2 + prefer-image WARNS" "$O" "will NOT reach"

mkfix /tmp/f4 none no;  O=$(run /tmp/f4 "")
check "no GPU at all" "$O" "ROCM_BOOT=nogpu"
check "no GPU at all" "$O" "GPU_DEVICES="
check "no GPU at all" "$O" "COMPOSE_FILE=docker-compose.yml"

# .env merge must preserve foreign keys and not duplicate its own
printf 'TARGET_UID=1000\nROCM_BOOT=stale\n' > /tmp/env.test
SYSROOT=/tmp/f1 ./rocm-detect.sh --write-env /tmp/env.test >/dev/null 2>&1
E=$(cat /tmp/env.test)
check "env merge keeps foreign key" "$E" "TARGET_UID=1000"
check "env merge replaces own key" "$(grep -c '^ROCM_BOOT=' /tmp/env.test)" "1"
check "env merge wrote fresh value" "$E" "ROCM_BOOT=wsl2"
# Not `A && B || C` -- that runs C when A succeeds but B fails, which would
# report a pass on the failure path. Spelled out as if/else instead.
if printf '%s' "$E" | grep -q 'stale'; then
  fail=$((fail+1)); echo "  [FAIL] stale value survived"
else
  pass=$((pass+1)); echo "  [PASS] stale value replaced"
fi

echo; echo "$pass passed, $fail failed"
echo "RESULT: $([ "$fail" -eq 0 ] && echo 'ALL PASS' || echo FAIL)"
[ "$fail" -eq 0 ]
