# Roadmap

Ordered by dependency, not by preference. Items in a later stage assume
the earlier ones exist, and the ordering is the point: pattern completion
cannot be built before separation, and tuning thresholds is meaningless
while the mechanism they gate is still the wrong shape.

Design rationale for each item lives in
[ARCHITECTURE.md](ARCHITECTURE.md); this file tracks sequence and status
only.

## 1. Correct the shipped mechanisms

These are places where working code implements the wrong semantics. They
come first because everything later builds on them.

- **Lease preemption.** `weakest()` skips any slot holding a lease, so a
  fully leased store returns `-1` from `locus_put()` and stops accepting
  traces. A leaked or long-held lease is an encoding deadlock. Needs a
  reclaim path keyed on a signal the pathway layer cannot write — lease
  age in ticks qualifies, `activation` does not.
- **Stochastic competition.** `locus_kwta()` is deterministic top-k.
  Competition should be noisy, with neuromodulatory tone setting its
  sharpness, so that exploration is a property of the mechanism rather
  than something layered on top.
- **Promotion trigger.** `locus_reinforce()` counts raw repetitions. The
  trigger should be a recurring winner sequence with consistently small
  prediction error — the outcome having stopped being surprising.
  Repetition alone promotes things that are still surprising.
- **Gate on re-stabilisation.** `locus_restabilize()` accepts anything.
  Consolidation is neuromodulator-gated in biology; without a gate, every
  retrieval can rewrite its trace unconditionally.

Depends on: nothing. All four are local to existing code.

**Status:** lease expiry, Gumbel stochastic selection and gated
re-stabilisation done 2026-08-10. Promotion is gated on a run of
low-surprise re-applications per trace; the recurring winner-sequence
form is open. See `CHANGELOG.md`.

## 2. Complete the competition model

- **Two scales of competition.** Features within a moment, and whole
  episodes against each other. Only the first exists. Episode-level
  competition is what pattern separation is.
- **Per-area competition.** A single global `active_k` lets one busy area
  starve every other. Competition is local to a population in biology, so
  the bound belongs per area, and residency for a dependent pair of areas
  becomes a joint decision rather than two independent ones.

Depends on: stage 1's stochastic competition, since both change the same
selection path.

**Status:** per-area competition exists as `Field(areas=...)` and was
measured not to act as a second control (see
[ARCHITECTURE.md](ARCHITECTURE.md)). Two-scale competition is open.

## 3. Recurrence and completion

- **A recurrent layer.** Everything currently in the substrate is
  feedforward. Attractor dynamics need genuine recurrent connectivity.
- **Pattern separation stage.** Decorrelates similar inputs before they
  are stored, so that near-identical episodes do not collide.
- **Pattern completion.** Converges on a stored attractor from a partial
  cue. Its capacity depends directly on separation working, so the two
  must be built and wired in that order.

Depends on: stage 2. Episode-level competition is the separation stage's
selection mechanism.

**Status:** the recurrent layer exists as `py/locus/field.py`, closed
into a loop by `py/locus/cycle.py`. Separation is open. Hierarchy and
compression: [HIERARCHY.md](HIERARCHY.md).

## 4. Learning dynamics

- **Eligibility traces and three-factor learning.** Local plasticity
  gated by a delayed global signal, so that credit assignment does not
  require backpropagation through the whole system.
- **Structural plasticity.** Unused connections stop existing rather than
  being zeroed. Growth is wake-only.
- **Sleep phase.** Two separable processes: replay as attractor
  re-settling (not log playback), and global downscaling scaled inversely
  by accumulated usage, with gist extraction as its byproduct. Downscaling
  is the one operation that touches the entire store rather than a sparse
  subset, so it must be scheduled as a batch sweep and can never run on
  the active path.

Depends on: stage 3. Replay-as-completion requires the completion
mechanism to exist.

**Status:** eligibility traces and the three-factor rule exist as
`py/locus/plasticity.py`, with ordered (pre before post) tags and
per-source scaling. On this repository's text the rule learns next-byte
structure and reproduces normalised counts exactly when its surprise
modulator, trace carry-over and per-file rescaling are removed; with
one or more of them it scores 0.53–1.20 bits per byte worse (`ARCHITECTURE.md`,
Learning a byte stream). A biology-faithful form (local delta rule,
noradrenergic gain, tag capture, soft bounds, per-target homeostasis)
exists and scores 1.57–2.01 bits per byte worse than normalised counts.
A per-input form (a separate error per input, a metaplastic rate of 1/n,
no soft bounds or homeostasis) equals normalised counts exactly; which of
those three differences accounts for the gap is not separated. Structural
plasticity and the sleep phase are
open.

Open question, unresolved: whether episodic and procedural replay share
one buffer or need two. They want different sampling criteria — diverse
and important versus literally repeated.

## 5. Substrate

- **A compute backend.** Nothing has run against a compute device. The
  open question is which backend actually serves these kernels on the
  target hardware, which is not answered by other software running on the
  same device through a different stack.
- **Hand-written kernels**, only where profiling shows a hot loop.
  Deliberately deferred until there is something measured to optimise.

Depends on: stages 1–4 being settled enough that the kernel shapes stop
changing.

## 6. Beyond the memory substrate

Named because they are known gaps, not because they are scheduled.

- Sensory encoding, and the efficient-coding transform feeding separation
- Motor output
- Top-down attention
- A utility and affect estimator assigning priority weight — currently
  salience exists as a consolidation input, but nothing estimates utility
- Multi-region integration
- Embodiment

## Standing constraints

These apply to every item above and are not milestones.

- A categorical exclusion list alongside the reversibility threshold:
  some action classes must never take the fast route at any evidence
  level.
- Any reclaim or gating signal must be written only by the autonomic
  process, never by the layer it constrains.
- A hard resource budget as a constraint the system must satisfy to
  produce output at all, rather than a soft penalty added afterwards.
- Biology and nature guide the inputs to every solution. The target is
  how a real brain performs and how little energy it spends doing so:
  every result reports its resource cost (time, and energy where it can
  be read) next to its accuracy.
- Experiments run in the project's own GPU container, not in CI workers.
- Work is split across the GPU and every CPU core. A single CPU-bound
  process is not an acceptable default for any component.
