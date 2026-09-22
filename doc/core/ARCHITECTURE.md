# Architecture

Why LocusAI is built as three structurally separate pathways over one
tiered trace store, and what each design constraint is answering.

The short version: an architecture in which every signal enters the same
weighted computation cannot give any signal categorical priority. Biology
does not work that way, and neither does this.

## The single-pathway problem

When instructions, learned associations and current context all flow
through one computation and are weighed against each other, no input can
structurally outrank another. It can be made *more likely* to win, never
guaranteed. There is no discrete decision at which a signal could be
gated, because there is no seam — the weighing is the computation.

Biology has several mechanisms that do grant genuine structural priority:

- **Spinal reflex arcs** bypass the brain entirely. Roughly 20–40 ms,
  against 150–300 ms or more for a cortical route.
- **The amygdala low road** (LeDoux). A fast, crude, thalamus-direct
  path can act before the slower accurate path finishes, which then
  retroactively corrects.
- **Locus coeruleus / norepinephrine.** A novel or salient stimulus
  triggers a global neuromodulatory reset that *interrupts* ongoing
  cortical processing rather than merely outvoting it.

Each of these is a separate route, not a heavier weight on a shared one.
That is the property LocusAI reproduces.

## Discreteness is the load-bearing property

It is tempting to treat "baked in early" as what makes instinct strong.
It isn't. Fixed action patterns (Lorenz, Tinbergen) separate two
properties that happen to coincide in biology:

1. **Timing** — fixed before the individual's operational lifetime,
   shaped only by the aggregate prior process, never by within-lifetime
   experience.
2. **Discreteness** — a releasing stimulus triggers an all-or-nothing
   program that runs independent of further correction once started.

The classic demonstration of (2): a greylag goose completes its full
egg-retrieval motion even when the egg is removed mid-movement.

Property 1 without property 2 gives you a strong tendency, not a
guarantee — soft, continuous, and outweighable by strong enough competing
input, no matter how heavily reinforced. This is why the fast pathway
here cannot be "a smaller, faster sub-network still doing continuous
weighting". It is closer to a state machine than to a scaled-down neural
computation.

In code: `Dispatcher.fires()` is exact membership in a releaser table. No
scoring, no probability mass, no partial match. Discreteness lives there
and only there.

## Reversibility scales commitment, and some actions are excluded

A fast pathway that can be corrected afterwards is safe. One that commits
to an irreversible action is not — there is no later state to correct
from.

Signal Detection Theory (Green & Swets) puts the optimal criterion at the
relative cost of the two error types rather than at a fixed point. When
a false positive is catastrophic and a false negative merely slow, the
threshold for *that* action rises.

Real predator behaviour is staged rather than single-shot: orienting and
preparing stay cheap and abortable, and only the terminal stage demands
convergent multi-signal confirmation. Abort is cheap before commitment,
impossible after.

In code: `Cascade` grades `HOLD / ORIENT / PREPARE / COMMIT`.
`commit_threshold()` and `required_signals()` both scale inversely with
reversibility; orienting and preparing deliberately do not.

`Constitution.permits()` (`py/locus/constitution.py`) bars action
classes categorically, alongside the continuous threshold. The veto sits
outside `Cascade`.

## Learning is two mechanisms, not one

Admitting a correction and making it cheap to reapply have different
triggers and belong in different subsystems.

- **Encoding** fires on novelty and prediction error (Schultz, on
  dopaminergic reward-prediction-error signalling). One occurrence is
  enough; no repetition is required.
- **Consolidation** fires on repetition count of a record that already
  exists (Balleine & Dickinson's goal-directed versus habitual
  distinction; basal-ganglia-mediated). It only ever operates on what
  encoding already produced.

A system that records corrections but never counts successful reuse has
the first and not the second: every retrieval stays an equally effortful
fresh judgement, whether it is the first or the fiftieth.

Encoding strength is also modulated by significance at encoding time
(McGaugh), which is why salience is an input to `NoveltyGate` rather than
a separate ranking pass.

In code: `NoveltyGate` gates encoding on prediction error;
`Consolidator.cost()` discounts geometrically with confirmed reuse; and
`locus_reinforce()` *is* the tier-promotion call — repetition-count
promotion and residency promotion are one mechanism.

`locus_reinforce()` promotes a trace only after `promote_after`
consecutive re-applications whose prediction error is at or below
`surprise_floor`; a surprising outcome resets the run.

**Not yet implemented.** The run is counted per trace. A recurring
sequence of winners across traces is not tracked.

### Learning a byte stream, measured

`make stream-learn` (`core/tests/exp_learn_stream.py`) learns next-byte
structure from this repository's text with `observe_transition`,
`consolidate` and `scale_sources`, and scores held-out files in bits per
byte (54,969 predictions; figures belong to the tree they ran on,
because the corpus is the repository).

| learner | bits per byte |
|---|---|
| count reference, add-one | 3.284 |
| normalised counts (control) | 2.276 |
| surprise modulator, trace decay 0.7, rescale per file | 3.473 |
| constant modulator, trace decay 0.7, rescale per file | 3.084 |
| surprise modulator, no carry-over, rescale per file | 3.241 |
| constant modulator, no carry-over, rescale per file | 2.804 |
| constant modulator, no carry-over, one rescale, no cap | 2.276 |

- The last learner equals the control exactly, on validation and test:
  the rule computes normalised counts when its three differences are
  removed.
- Cost of each difference, between learners differing only in it: the
  surprise modulator 0.39–0.44 bits, trace carry-over 0.23–0.28 bits,
  per-file rescaling with the 5.0 weight cap 0.53 bits.
- No learner beats the normalised-count control.

## Retrieval is spreading activation, not search

Getting a relevant trace *considered* is a separate problem from ranking
traces once you have them. Query-driven retrieval requires already
knowing what to look for.

Spreading activation (Collins & Loftus) surfaces associatively linked
items mechanically, in proportion to associative closeness, with no
explicit query. Closer to graph traversal with decay than to keyword or
embedding search.

In code: `AssociativeGraph.spread()` accumulates activation with per-hop
decay and a floor cutoff. `prefetch_set()` turns that into warming
candidates — advisory only, so a wrong guess costs latency, never
correctness.

## Store semantics

The store is tiered but is not a cache. See `src/locus.h` for the full
contract; the reasoning behind the parts that differ from a cache:

- **Tiers hold representations, not copies.** Systems consolidation
  transforms a pattern-separated trace into an overlapping, statistical
  one (complementary learning systems — McClelland, McNaughton &
  O'Reilly). Identity is deliberately not preserved. Multiple trace
  theory (Nadel & Moscovitch) further suggests both coexist rather than
  one replacing the other.
- **Retrieval destabilises.** Recall returns a memory to a labile state
  requiring re-stabilisation (Nader, Schafe & LeDoux), so the read path
  is also a write path. `locus_retrieve()` marks a trace labile;
  `locus_restabilize()` must follow and may rewrite it. A trace never
  re-stabilised weakens rather than persisting unchanged.
  `locus_lookup()` does not destabilise — the instinct and procedural
  routes cannot afford a reconsolidation cost. `locus_restabilize()`
  returns `LOCUS_GATED` and discards the update when neuromodulatory
  tone is too low.
- **Capacity is interference, not bytes.** Working memory limits are
  competition-based (Cowan), so the active set is bounded by
  k-winners-take-all. Pinned traces are required residents and are
  exempt.
- **Weak traces can be rescued.** Synaptic tagging and capture (Frey &
  Morris): a weakly potentiated synapse can capture plasticity-related
  proteins produced by a nearby unrelated strong event. Cross-item,
  non-local promotion, which no eviction policy provides.
  `locus_note_salient()` implements this across a bounded window.

`ACTIVE` is not simply the fastest tier. Short-term maintenance is
persistent activity or residual presynaptic facilitation (Fuster &
Alexander; Mongillo, Barak & Tsodyks) — a state held up by ongoing
energy expenditure, not stored bytes. It has to live where the compute
does.

## Kernel-row residency

Design, not implemented. Applies to the `Field` kernel; the trace-store
tiers above are unchanged.

- The unit of residency is a kernel row: the outgoing weights of one
  source unit. The drive reads row i only when unit i is above the floor,
  so a row whose source is quiet is not read that tick.
- Tiers: VRAM holds the field state and the rows of active and recently
  active units; host RAM holds rows of dormant units; disk holds rows
  unused for long. The field state never leaves VRAM.
- Demotion is by ticks since the row was last read, counted by the
  autonomic tick. VRAM capacity is the only limit: when a row must be
  placed, the least recently read one moves down a tier.
- Rows whose sources are rising below threshold are prefetched from host
  RAM while the current tick computes.
- Precision is identical in every tier. A row round-trips exactly, so
  residency never changes the dynamics. This differs from the trace
  store, where tiers hold transformed representations.
- A row needed but not resident is fetched before the tick completes. A
  miss costs time, never correctness.
- Prerequisite: a sparse row layout with a compacted list of active
  rows.
- Quiet is defined exactly: a term below half an ulp of the target's
  running drive leaves the fp32 sum unchanged, so skipping it is
  bit-identical (`make field-quiet`); a row is quiet in a step when all
  its terms are. Measured on the ring kernel at beta 1.0: no term and no
  row was quiet in 300 driven steps at n up to 4096.
- The sparse kernel costs about 60 bytes per unit (7 non-zeros × 8 bytes
  plus a 4-byte offset), under 1 MiB at n = 16384, so residency tiers
  are not needed until the kernel approaches VRAM capacity. A source-row
  layout suited to tiering is exact but costs 1.48–1.50× per step; the
  compute layout gathers by target until then.

## Autonomic processes and signal ownership

`locus_tick()` runs decay, tier enforcement and expiry. No pathway
commands it, which makes it autonomic in the same sense as the
mechanisms biology places outside conscious access.

Voluntary override of an autonomic process is normal — breathing is the
obvious case — but it is reclaimed when a homeostatic signal crosses a
threshold, not when attention wanders. Biology reclaims on a proxy
(CO₂ via central chemoreceptors) rather than on the quantity that
matters (O₂), because the proxy is cheaper and faster to sense. That
choice has a documented failure mode: decouple the proxy from the hazard
and the override survives past harm.

A synthetic system can compute the real quantity instead, so that
failure class is avoidable here. The remaining constraint is subtler and
is a design rule:

> A reclaim signal must be written only by the autonomic process, never
> by the layer it constrains.

Otherwise the constrained layer is asked to grade its own need. By that
rule `activation` is disqualified — it is writable from the pathway side
via `locus_excite()` — while a tick-incremented lease age is not.

Leases expire: a lease held longer than `cfg.lease_max_ticks` is
revoked, freeing its slot (see `src/locus.h`).

## What the loop now does

Three modules added 2026-09-18, all verified against the real device
rather than in a sandbox. Each is deliberately separate, so a surprising
behaviour belongs to one of them rather than to the wiring.

- `py/locus/field.py` — continuous activation over the state space,
  updated as ONE simultaneous event. Every term (self-excitation,
  lateral drive through the kernel, shunting inhibition, leak) is
  computed from the same previous state and applied together. This
  replaced a staged `excite -> spread -> kwta -> record` pipeline, which
  was wrong about the biology in a way that produced a false result:
  staging it made competition a sort, the sort produced a step function
  that was reported as a property of the dynamics, and it discarded the
  sub-threshold activations that an eligibility trace and a deferral
  threshold both need. A second pathway carries top-down bias from
  outside the pool, since biology keeps goal maintenance and pattern
  completion in different circuits.
- `py/locus/plasticity.py` — eligibility traces and a three-factor gate.
  Coincident activity sets a local decaying tag; a weight moves only
  when a broadcast modulator arrives while the tag is alive. Activity
  alone changes nothing, which is what separates this from Hebbian
  learning. One modulator with a sign does both potentiation and
  depression, so suppression needs no separate rule.
  `observe_transition` tags ordered pairs (pre before post) for
  sequence learning; `scale_sources` rescales each source's outgoing
  weights multiplicatively to a fixed sum, so a source's targets compete
  without depression.
- `py/locus/encode.py` — byte-stream input: byte units plus hashed
  n-gram units, table sizes derived from the stream.
- `py/locus/cycle.py` — the perceive-retrieve-decide-learn loop, closed.
  Settle, decide, learn from the SETTLED state, then derive the next
  bias. It adds no mechanism of its own.
- `py/locus/trace_chain.py` — the level-coupling operator. A coarse
  level is DERIVED from a fine one, never stored, so there is no shared
  cell and no second writer.
- `Cycle.level_up`, `survey_saturating` and `drive_and_stack` — the
  hierarchy. `level_up` forms a coarse level from stable attractor
  supports, with the coarse kernel derived by `trace_chain` over one
  representative per support. `survey_saturating` drives the field until
  the set of supports stops growing and reports the number of cues
  consumed. `drive_and_stack` builds levels until one refuses. Depth and
  compression are outputs, not parameters.

## Not implemented

Beyond the gaps noted above:

- Neuromodulated competition. `Field(noise=..., seed=...)` adds seeded
  spontaneous activity to each unit's input after shunting inhibition.
  Tonic/phasic neuromodulation does not set competition sharpness;
  `beta` is the only sharpness control. Measured in one configuration
  per beta (`make operating-point`, n=64): the active share reaches the
  biological 1–2% band at beta ≥ 0.7, and beta 0.5 leaves 10.9% active.
- Competition at two scales — features within a moment, and whole
  episodes against each other. Only one scale exists.
- A second competition control. `Field(areas=...)` implements local
  inhibition, and it is not a second control: A areas divide each unit's
  inhibitory denominator by A. Measured, the active share rose from 2.3%
  to 16.7% and the repertoire fell from 60 to 3.
- Compression. `level_up` chunks by attractor support. At biological
  sparsity the repertoire is about n, so measured compression is about
  1.0x (1.03x at beta 1.0, n=64). No beta gives biological sparsity, a
  large repertoire and compression together. Earlier figures of 8.00x,
  8.53x, 10.67x and 85.3x are retracted; see `CHANGELOG.md`. The
  mechanism under design is a projection between pools of different
  sizes.
- Sensory encoding beyond text. Byte streams are encoded as byte units
  plus hashed n-gram units (`py/locus/encode.py`), with table sizes
  derived from the stream; no other modality is encoded. Motor output,
  sleep-phase consolidation, and the pattern-separation stage that must
  precede completion.
- A language renderer. No English chain-of-thought, which the
  transparency requirement needs. Measured ceiling on what one could
  ever report: about 24 states carry real activation per tick while 4
  are selected, so a narration covers roughly 17% of what competed.
- Any GPU kernel for the substrate itself. The device probes DO run —
  `make gpu`, `caps`, `persist` and `excursion` all execute against
  gfx1100 — but no part of the store, field or plasticity has a device
  implementation. Kernel shapes are deliberately not frozen while the
  shapes above are still changing.

### What recurrence produced, once it existed

The recurrent layer listed here as missing now exists, and measuring it
changed a design assumption worth recording. Over 64 injections at
n=64 the field settles into 11 distinct attractors of mean size 3.8
units — and the injected state ends up inside its own attractor only
14% of the time. So persistence is a property of a SET, not of a node:
an injection selects a basin and then dissolves into something it is
not part of. Three candidate single-unit persistence mechanisms were
ruled out before this was understood, all of them sharing the
assumption that a node is what persists.

One attractor captured 48 of those 64 basins. The count of 11 is not a
capacity figure: it depends on inhibition and on the number of
injections. With `survey_saturating`, the measured repertoire is 15 of
32, 52 of 64, 126 of 128 and 254 of 256 states, and saturation takes
about 5.7 cues per repertoire member.
