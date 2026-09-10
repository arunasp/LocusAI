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
