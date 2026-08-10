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

**Not yet implemented.** Scaling a threshold is not sufficient on its
own. Some action classes should be barred from the fast route at any
evidence level — a categorical exclusion list alongside the continuous
threshold.

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

**Not yet implemented.** The trigger should be a recurring winner
sequence with consistently small prediction error — the outcome having
stopped being surprising — not raw repetition count. Repetition alone
promotes things that are still surprising.

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
  routes cannot afford a reconsolidation cost.
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

**Not yet implemented.** The lease is currently absolute, with no
preemption: if every slot is leased, `locus_put()` fails and encoding
stops. A long-held or leaked lease is a deadlock with no reclaim path.

## Not implemented

Beyond the gaps noted above:

- Stochastic k-winners-take-all. The current implementation is
  deterministic top-k; competition should be noisy, with tonic/phasic
  neuromodulation setting its sharpness.
- A gate on re-stabilisation. Consolidation is neuromodulator-gated in
  biology; here anything may currently re-stabilise.
- Competition at two scales — features within a moment, and whole
  episodes against each other. Only one scale exists.
- A recurrent layer. Pattern completion needs genuine recurrent
  connectivity for attractor dynamics; everything here is feedforward.
- Per-area competition. A single global `active_k` lets one busy area
  starve the rest.
- Sensory encoding, motor output, sleep-phase consolidation, and the
  pattern-separation stage that must precede completion.
- Any GPU backend. Nothing here has run against a compute device.
