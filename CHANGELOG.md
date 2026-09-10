# Changelog

Notable changes to LocusAI. Format follows [Keep a
Changelog](https://keepachangelog.com/en/1.1.0/); dates rather than semantic
versions, since nothing is released yet.

## [Unreleased]

### Added

- `doc/core/METHODOLOGY.md`, `doc/core/AUTONOMIC.md` and
  `doc/core/HARDWARE.md` — design content that previously existed only as
  working notes. AUTONOMIC states an open decision the code currently
  answers by default rather than by choice: whether the autonomic layer is
  opaque to the pathways above it, or readable.
- `core/py/locus/constitution.py` — supplies the reversibility cost
  `Cascade` already consumed but nothing produced, plus a categorical
  `permits()`. First code implementing anything from `CONSTITUTION.md`.
- `core/gpu/probe.cpp` and a `gpu` make target — compiles a HIP translation
  unit, reads device properties from the HIP runtime, launches a kernel and
  verifies every element. Deliberately outside `all`, and skips rather than
  fails where `hipcc` is absent, so a checkout without a GPU still builds.
- `cicd/transfer.py` with `pack` / `unpack` / `wheels` targets — checksummed
  base64 transfer, verified before extraction; offline wheelhouse install.
- `core/gpu/caps.cpp` and a `caps` make target — reads the memory hierarchy
  and the cache-control surface from the HIP runtime, and *tests* claims
  rather than restating them: each capability query's result is reported, and
  every assertion is an invariant of a working probe rather than a value
  specific to this card, so the same source runs on either boot.
- `core/gpu/persist.cpp` and a `persist` make target — occupancy bound,
  cooperative-launch support and LDS state across ticks. Gated behind
  `LOCUS_GPU_STRESS=1` because a long kernel on a GPU also driving a display
  can trip a driver reset; the tick count starts low and is raised by hand.
- `core/gpu/mock/hip/hip_runtime.h` and `core/gpu/tests/gpu_mock_driver.py`
  with a `caps-mock` target, now part of `test` — 19 scenarios covering the
  four tool-presence states, including the two usually missed (a field
  *absent* from the runtime's struct, and a query *present but refusing*).
  Needs no GPU, no hipcc and no venv, so it is the only stage that can guard
  the probes' reporting logic where `caps` can merely skip.
- `tools/server/rocm-detect.sh` and `tools/server/tests/rocm-detect.test.sh`
  — decides at invocation how this boot reaches the GPU and where its ROCm
  SDK comes from, emitting shell-assignable settings (`ROCM_BOOT`,
  `GPU_DEVICES`, `ROCM_HOST_VERSION`, `ROCM_HIP_TRIPLE`, `ROCM_SOURCE`,
  `ROCM_IMAGE`, `COMPOSE_FILE`) or merging them into compose's `.env`.
  `--pull` fetches a version-matched image with a disk preflight, refusing
  under 20 GiB free rather than discovering the partition full. A host ROCm
  is preferred where one exists because under WSL2 the host's HSA runtime is
  a dxg-aware build and AMD's generic images target `/dev/kfd`; where there
  is none — the Mageia case — the image is the whole SDK and defines the
  version. It refuses to invent a tag rather than guess into a multi-gigabyte
  pull. `SYSROOT` exists as an injection point so all six boot/SDK states are
  testable on any machine; 23 assertions cover them plus the `.env` merge.

### Measured

- Full memory hierarchy on `gfx1100`, recorded in `doc/core/HARDWARE.md`.
  The design-relevant result: **32.25 MiB of vector registers against 6 MiB
  of L2**, so the innermost tier is five times the one outside it and "spill
  to the next level down" is not a size ladder here.
- `cooperativeLaunch` is **0** on this device, confirmed by both the property
  field and `hipDeviceGetAttribute`. A grid-wide barrier is therefore
  illegal, which closes the whole-grid form of a resident-kernel ACTIVE
  tier. Per-block persistence remains available; 8 blocks per WGP at 256
  threads gives a 336-block, 86016-thread co-resident grid.
- The hierarchy is not runtime-configurable. `hipLimit_t` has no
  persisting-cache member, `accessPolicyMaxWindowSize` is 0, and
  `hipDeviceSetCacheConfig` returns success as a silent no-op. The only
  software lever is negative (non-temporal hints), so a hot/cold split must
  work by sizing, never by placement.
- `memoryClockRate` is unreliable under GPU-PV: it yields 50 or 100 GB/s
  against a real ~800 GB/s, so DRAM bandwidth cannot be derived from HIP
  here.

### Changed

- `tools/server/docker-compose.yml` is now ONE file covering both the WSL2
  and native-Linux boots. Compose has no conditional syntax, so every
  difference is a variable value: two fixed device slots (`GPU_DEV_1/2`), and
  a `${ROCM_MOUNT_SRC}` / `${WSL_LIB_SRC}` pair that exploits compose short
  syntax — a source with a leading slash is a bind mount, one without is a
  named volume — so the same line binds the host's ROCm on WSL2 and mounts a
  Docker-managed volume on a host that has none. The one genuinely
  conditional piece is a compose profile: a `rocm-sdk` service that seeds
  that volume from AMD's image and exits, with `pull_policy: missing` so the
  pull is part of an ordinary `up` rather than a separate step, and a
  `depends_on: {condition: service_completed_successfully, required: false}`
  so the main service cannot win the race and create the volume EMPTY while
  `required: false` keeps the same line valid on WSL2 where the seeder is not
  in the active profile. Both branches verified with `docker compose config`
  against real and fixture env files before anything was installed.
- `tools/server/start.sh` runs `rocm-detect.sh --write-env` before building,
  so those values are populated rather than falling back to defaults.
  Deliberately without `--pull`: compose's own `pull_policy` already fetches
  the image during `up`, and a multi-gigabyte download should not be an
  invisible part of starting the server.
- `tools/pipeline.sh`'s `lint` stage now asserts executable bits against the
  git index, checking BOTH the index and the working tree. The Filesystem
  connector writes and edits at 644 and does not preserve 755, and the defect
  is invisible to a read-back because the mode is not part of file content —
  so it needed a stage rather than a rule. It caught its own introduction:
  the edit that added it stripped `pipeline.sh`'s own bit, and the first run
  went red naming that file.
- Design docs moved from `core/` to `doc/core/`, one directory per
  component. `core/README.md` stays as the component entry point.
- `tools/server` scope reduced to testing project code only: the gh CLI
  install, the `/root/.ssh` deploy-key setup, the key mount and `git`/`gh`
  allowlist entries are gone. Deployment goes through the separate CI/CD
  runner instead.
- `tools/server` now runs as the invoking host user rather than root, via an
  entrypoint that creates the passwd entry and drops privilege, plus a
  raised `nproc` ulimit. Removing the `/root/.ssh` setup is what made this
  possible — mode 700 is untraversable to the dropped user.
- Desktop extension rewritten onto `@modelcontextprotocol/sdk` transports
  held in-process (3.0.0). It detects a stale session, replays the handshake
  and retries, so rebuilding the server container no longer requires a
  manual reconnect.

### Fixed

- Lease expiry, stochastic selection, surprise-gated promotion and the
  re-stabilisation gate — the four corrections in roadmap stage 1. Absolute
  leases had made a full store stop accepting traces with no reclaim path.
- Falsy-success trap across three enums whose zero-valued member is the
  *ordinary* outcome, so `if result:` read backwards. `Restabilize` defines
  `__bool__` correctly; `Tier` and `Pathway` raise, having no sensible
  boolean reading.
- `gpu` and `build` make targets collided with same-named directories and
  silently did nothing. Both are now `.PHONY`.
- **`make gpu` could no longer compile at all**, silently since the host
  moved to ROCm 7.14 on 2026-08-18 — the target had not been re-run in
  between. Two independent causes, both documented in
  `doc/core/HARDWARE.md`: `/opt/rocm`'s `bin`/`lib`/`include` are
  update-alternatives symlinks into `/etc/alternatives`, which the container
  does not mount, so they dangle inside the mounted tree; and that ROCm
  install ships no HIP headers at all. `core/Makefile` now *resolves*
  `ROCM_PATH` by finding a real `bin/hipcc` instead of assuming
  `/opt/rocm`, and picks up extracted headers via `-isystem` only when
  present, so a host with a complete SDK needs no flag.
- `lint` skipped `core/gpu/tests` entirely, so the new driver was outside
  pycodestyle. Added to the target, and the seven long lines it found are
  fixed.
- The extracted-header workaround could go stale silently: skewed HIP
  headers compile fine while reporting wrong struct offsets. `make
  hip-header-check` now compares the headers' `hip_version.h` triple against
  the host's `share/hip/version` and fails loudly, and it is a prerequisite
  of `gpu`, `caps` and `persist` so skew cannot reach a compile. Surveyed
  the alternative first (see `doc/core/HARDWARE.md`): AMD publishes no thin
  ROCm dev image at 7.x, only `-full` at 7.4-7.7 GiB, and the thin ones that
  exist are 6.4 — so the 494 KB package extraction is the only source that
  is both thin and version-exact, and the fix belongs in the assertion
  rather than in a different source.
- Extension tests still asserted the pre-rewrite design and would have been
  committed red.

## 2026-08-10

Substrate work begins. The trace store and pathway layer land as a C core
with a Python layer over a ctypes binding, with the design written up in
`doc/core/` rather than left in conversation. GPU passthrough reaches the
project container end to end.

## 2026-08-04 – 2026-08-08

Tooling and CI, before the substrate existed.

- `tools/pipeline.sh` reached its first fully green run. Two real defects
  surfaced in doing so: ShellCheck needed `-P SCRIPTDIR` to resolve sourced
  files, and a health check used `curl --fail`, which treats the server's
  correct HTTP 406 as failure.
- Auth split: `git` over ssh with a read-only deploy key, everything else
  through the gh CLI's own token, with GitHub's ed25519 host key pinned
  rather than trusted on first use. Superseded on 2026-08-10 when this
  container's scope narrowed.
- The CI/CD tooling itself was split out into a separate project after
  proving hard to maintain in place. The two coexist: that runner executes
  pipelines across projects, this container keeps device access for testing
  LocusAI's own code.
