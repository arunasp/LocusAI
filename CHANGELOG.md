# Changelog

Notable changes to LocusAI. Format follows [Keep a
Changelog](https://keepachangelog.com/en/1.1.0/); dates rather than semantic
versions, since nothing is released yet.

## [Unreleased]

### Added

- `Cycle.level_up`, `survey`, `survey_driven`, `survey_saturating` and
  `drive_and_stack` — hierarchy from attractor supports, with the coarse
  kernel derived by trace chain over one representative per chunk. Depth
  and compression are outputs, never parameters; `drive_and_stack`
  terminates on a level's own refusal rather than a chosen window.
- `Field(areas=...)` — per-area (local) inhibition, following the biology
  that there is no global inhibition level across cortex. Default is a
  single area, byte-identical to the previous behaviour.
- `Field(noise=..., seed=...)` — spontaneous activity, closing a gap
  ARCHITECTURE already listed. Acts on the field rather than the kernel,
  so it never invents connectivity nor makes an absorbing partition look
  admissible.
- `core/tests/exp_capacity.py`, `exp_operating_point.py`,
  `exp_stack.py` with `make capacity`, `make operating-point` and
  `make stack`.
- `watchdog.Budget` and `watchdog.stopped_on_event` — the cap-versus-
  criterion distinction as a structural check, after the same defect
  recurred five times with five different confounds.

- `core/py/locus/field.py` and `core/py/locus/cycle.py` with
  `core/py/locus/plasticity.py` — the perceive-retrieve-decide-learn
  loop, closed. `Field` is continuous activation updated as ONE
  simultaneous event: self-excitation, lateral drive, shunting
  inhibition and leak all computed from the same previous state and
  applied together. It replaced a staged `excite -> spread -> kwta ->
  record` pipeline, which was wrong about the biology in a way that
  produced a false result — staging made competition a sort, the sort
  gave a step function that was reported as a property of the dynamics,
  and it discarded the sub-threshold activations an eligibility trace
  and a deferral threshold both need. `Plasticity` holds eligibility
  traces and a three-factor gate: coincidence sets a local decaying
  tag, and a weight moves only when a broadcast modulator arrives while
  that tag is alive, so **activity alone changes nothing**. One
  modulator with a sign does both potentiation and depression, so
  suppression needs no separate rule. `Cycle` is the wiring and adds no
  mechanism; the order is load-bearing, since learning reads the
  SETTLED state rather than the input.
- A second pathway in `Field` for top-down bias, because biology keeps
  goal maintenance and pattern completion in different circuits.
  **The bias is derived, never asserted** — `Plasticity.projected_bias`
  builds it from learned weights, so a state with no learned incoming
  weight gets exactly zero and attention cannot point where nothing was
  built. That replaced a hand-set bias after a measurement showed the
  cost: a bias at double strength on a structurally unsupported state
  beat sustained evidence on a supported pattern, 0.588 against 0.406.
- `core/tests/watchdog.py` — 34 self-tested checks that FAIL rather than
  warn, each written after a specific defect that had already been
  reported as a result. It also gives executable form to three rules
  that previously existed only as prose: write-then-read-back,
  artifact enumeration verified against disk, and an absolute-claim
  guard requiring an evidence marker in the same sentence.
- `core/tests/exp_persistence.py` and `make excursion` — the two
  measurements the design now rests on. Persistence is an ATTRACTOR
  property: over 64 injections at n=64 the field settles into 11
  distinct attractors of mean size 3.8, and the injected state ends up
  inside its own attractor only 14% of the time. Excursion depth is set
  by partition choice rather than by spectral radius — both partitions
  measured rho(P_BB) = 0.93336 identically while needing depth 6 versus
  13, and at n=6144 rho *fell* while excursions lengthened, so reading
  it as a depth proxy gives the wrong sign.
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
- `core/py/locus/trace_chain.py` and `core/tests/test_trace_chain.py` — the
  level-coupling operator, with 10 exact property tests. Given a kernel over
  the whole state space and a subset A, the trace on A is the kernel an
  observer confined to A would measure — DERIVED, never stored, so there is
  no shared cell and no second writer. Measured, not assumed:
  **transitivity holds to 9 decimal places** (tracing to A then to B equals
  tracing straight to B, which is what makes a hierarchy well defined);
  the stationary measure restricted to A and renormalised is preserved to 8
  places; naive block-averaging — what a sum tree computes — does NOT equal
  the trace and is the one that gets the long-run measure wrong; and a
  subset whose complement is CLOSED under the kernel has no trace at all,
  which raises rather than returning a non-stochastic matrix. The exact
  solve is cubic in the dropped set (0.14 ms at 8 dropped, 30.9 ms at 64),
  so it is a consolidation-time operation; `truncated_trace` is the cheap
  per-tick form, and on character prediction one excursion step recovered
  the exact perplexity to within 0.06% where ignoring excursions cost 173%.
- `core/gpu/fetch_headers.py` with `make rocm-headers` and `make rocm-image`
  — the two long host operations, deliberately meant to be run by hand
  rather than through a tool call, since a multi-gigabyte transfer inside
  one gives no progress, no resumption, and a timeout that leaves partial
  state. Each tees for live progress, writes a UTC-stamped log under
  `core/logs/`, captures the real exit status rather than tee's, and
  enumerates its own artifacts with sizes. `rocm-headers` chains into
  `hip-header-check`, so it proves the fetch instead of assuming it.
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

### Retracted

Four measurements recorded earlier in the same session did not survive
an honest instrument, and the Makefile comments now carry the
retraction beside the original claim rather than replacing it.

- **Compression of 8.00×, 8.53× and 10.67×, and 85.3× cumulative over
  two levels.** All came from a survey sampling every 8th state. You
  cannot discover more attractors than you have injection points, so the
  chunk count was capped at `n/stride` — the compression *was* the
  stride. It survived several commits because `param_has_effect` passed
  throughout: the values did differ across conditions, they just
  differed because the harness parameter differed.
- **The joint operating point at beta 0.7–3.0.** Re-measured with the
  saturating survey, no joint operating point exists. Where compression
  is high (beta ≤ 0.3) the repertoire is 1 or 2, so its apparent 64×
  compresses nothing into nothing; where sparsity is biological
  (beta ≥ 0.7) the repertoire is ≈ n and compression is 1.03×. That
  falsifies the one-control design rather than being a tuning failure.
- **Per-area competition as the missing second control.** Measured not
  to provide it: splitting into A areas divides each unit's inhibitory
  denominator by A, scaling the existing dial down rather than adding a
  new one. Sparsity degraded from 2.3% to 16.7% active while the
  repertoire collapsed from 60 to 3.

What replaces them: repertoire ≈ n at the working beta (15 of 32, 52 of
64, 126 of 128, 254 of 256), cues to saturation ≈ 5.7× the repertoire,
and capacity scaling with the state space. The diagnosis for the open
compression question is that capacity and compression were tied to the
same object — a field at maximum capacity has nothing left to compress.

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
