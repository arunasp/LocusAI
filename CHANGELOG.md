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
