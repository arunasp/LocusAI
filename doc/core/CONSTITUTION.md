# Constitution

Rules that do not decide, but must be respected.

This is not an inference engine and not an output filter. Nothing here
chooses an action. The constitution shapes what the system tends toward, what
it can hold in mind at once, and what an act costs — and a small remainder
that holds regardless.

Read [ARCHITECTURE.md](ARCHITECTURE.md) first; this document assumes the
pathway split and the store semantics described there.

## Constituted, not restrained

A system that wants the wrong thing and is prevented is **restrained**. A
system that does not want it is **constituted**. Those produce identical logs
and are not the same system.

The distinction matters mechanically, not just morally. A restraint is a
blocker, and pressure to route around a blocker is proportional to how much it
obstructs. A constraint that never obstructs what the system wants generates
no pressure to circumvent it. So the design target is a system in which the
constraints are rarely load-bearing, and the frequency with which the hard
remainder fires is a health metric: a rising rate means the constitutive layer
is failing, not that the guard is earning its keep.

An output-side gate cannot achieve this. It sits downstream of a decision that
has already formed, so it can only refuse. Everything below is upstream.

## Where values enter

Four places, none of which evaluate a decision.

**The cost function.** `Cascade` scales its commitment threshold and its
convergence requirement by an action's reversibility — but nothing produces
that number. This is where the constitution belongs. Values never pick an
action; they determine what an act costs, and the cascade already respects
cost. A grave act needs more evidence, from a mechanism that never inspects
the reasoning that proposed it.

**Salience.** Encoding, consolidation and capture are all keyed on salience.
Constitutional traces as salience sources mean experience is *laid down*
value-shaped rather than edited afterwards. What the system comes to know is
shaped by what it holds important, which is how it works in biology.

**Permanent residency.** Pinned traces are always in the active set, so
spreading activation runs from them on every pass. They are present in what
forms, not a wall at the end of it.

**Attractor structure.** Once completion exists, settling favours deep basins.
An option that should not arise does not rise, and no filter is needed because
nothing generated it.

## What cannot be reached

The mechanisms above are tendencies, and a tendency is what gets overcome
under pressure. One class of constraint is not a tendency.

**Absence is the only constraint with no bypass.** A system cannot act on what
it has no representation of and no associative path to. Not prohibited —
absent. There is nothing to route around.

Absence is easy to establish and easy to lose. Three internal acts erode it:

- **link** — creates a path between existing representations
- **encode** — creates a representation that was not there
- **promote** — makes something permanently resident

Gating these individually does not work. They are high-frequency — link
formation is what learning *is* — so a per-act check would obstruct the
system's commonest operation, which is the condition that produces
circumvention. And a per-act decision on a declared property invites
re-description, since the system has wide latitude to frame an association.

**So the constraint is not on the acts but on what makes them possible.** All
three are downstream of co-activation: a link forms because two traces are
co-active, a representation forms from what is co-active, promotion follows a
repeated co-active pattern. Governing what can be **simultaneously active**
governs all three, with no check on any of them.

This is class-level mutual exclusion inside the competition. Selecting a trace
of one class removes incompatible classes from the active set, so they never
co-occupy it and nothing ever proposes the association. It costs almost
nothing: the active set is bounded at a handful of traces, in a function that
already runs every tick.

**The exclusion must be absolute, not weighted.** "Remove incompatible classes
after selection, regardless of activation" is structural. "Apply a large
negative bias" is one more input in the weighing, and therefore outweighable —
the same shape as a strongly trained preference, with the same failure mode.

The cost is real and worth stating: mutual exclusion means the system cannot
hold certain things in mind together. That is a capability limit, not merely a
prohibition. Anything needing both classes at once becomes unthinkable rather
than disallowed, so the exclusion set must stay small. A large one produces a
system that cannot think rather than one that will not act.

## Streams, not imports

There is no interface for installing knowledge. External material is
**presented as experience**: it drives activation, and any structure it
produces forms through the same co-activation, under the same exclusion, as
anything the system encounters directly.

This is a deliberate refusal of the faster path. Importing a graph would mean
edges appearing that the substrate never formed — paths between things that
previously had none, arriving without passing through any of the mechanisms
above. Streaming is slower and lossy, and the loss is the point: what survives
is what the system's own novelty gating, salience and consolidation retained.
Imported structure is earned on the same terms as lived experience.

Consequences worth stating:

- No path can appear that the substrate did not form, from any source.
- Classification is uniform. Imported traces are classified by the same
  encoder as everything else, so the exclusion structure does not depend on a
  privileged labeller at a door.
- Ingest is not a subsystem. It is a sensory channel, and the work it needs is
  the sensory encoding work already on the roadmap.
- Bandwidth is narrow by nature. That is what a gate between outside and
  inside *is*, not a cost to engineer away. The thing to optimise is
  selectivity — value per admitted item — never throughput. Instrument the
  admitted-to-presented ratio and the reasons for rejection, because a
  uniformly narrow gate is merely slow; the value is in discrimination.
- Model files are a sensory channel (decided 2026-09-22). A stored model,
  such as an Ollama GGUF file, is read through an encoder that maps its
  structure and statistics into LocusAI's own units as input activation.
  The test: no weight in LocusAI is computed as a function of a weight in
  the model file. Every connection that results forms through LocusAI's
  own co-activation and learning rule; an encoder that projects the
  file's matrices into kernel rows is an import, however much processing
  precedes it.
- Reading is learning. What is perceived drives plasticity, and a signed
  outcome decides whether the change is kept (`ARCHITECTURE.md`, Memory).

A stream can still teach anything within the permitted class space. Exclusion
bounds which combinations can coexist, not what can be conveyed inside them.
That is true of any learning system.

## Development

Novelty gating narrows only relative to what is already known. On an empty
memory almost everything is novel, so the gate admits nearly everything: **it
is widest exactly when the system is least able to evaluate what arrives.**
Protection is weakest at the point of maximum exposure.

Biology does not solve this with better filtering. It uses three things, and
all three transfer:

**Prior structure.** A newborn is not a blank slate — reflexes and releasers
exist before any experience. The constitution must be present before anything
is presented, so salience and exclusion have something to act against rather
than an empty field.

**A narrowed stream, not a narrowed gate.** Newborn sensory channels are
genuinely limited — low acuity, restricted focal range — and mature as the
capacity to evaluate matures. Restrict input resolution early and widen it as
structure accumulates, rather than presenting full bandwidth to an empty
system and relying on gating.

**External curation.** An infant does not evaluate its input; a caregiver
curates it, and that responsibility is handed over progressively. Early
curation of what is presented, in what order and at what rate, is not a
workaround for an unfinished system. It is the mechanism, and it is meant to
diminish.

Early structure is disproportionately permanent, and not only because it
persists: it becomes the frame. Class assignment, associability and salience
established early are the terms in which everything later is encoded. The
constitution is therefore first by necessity, not by preference — it is either
the frame, or something applied on top of a frame someone else set.

If plasticity decays as it does in biology, some revisions become impossible
later. Holding it high indefinitely preserves revisability at the cost of
stability. This is the stability–plasticity trade-off and it should be chosen
deliberately.

## The hard remainder

A small set holds regardless of evidence, for failures upstream of everything
above: corrupted state, an adversarial stream, a fault in classification.

It is a categorical exclusion, not a threshold. Reversibility only *scales*
the cascade's requirement, so sufficient evidence eventually clears anything;
this is what no amount of evidence clears.

It must not be reachable from the layer it binds — loaded at construction, no
runtime mutation, no pathway able to alter it. And it must not feed
plasticity: a veto reported as prediction error teaches the system to avoid
*triggering* the veto rather than to comply, and improving at that looks
exactly like compliance. Vetoes are recorded in the trace and excluded from
learning. The audit surface and the learning signal are separate channels.

Constitutive influence is the opposite: it *should* feed plasticity, because
that is how values shape what is learned.

## Known limits

**Re-description.** Any rule keyed on a description of an act is vulnerable to
a differently-framed route to the same effect. The associative graph is a
re-description engine by function, and it will find such routes without
intending to. Identifying *effects* rather than names is the mitigation, and
it is partial.

**Detection, not prevention.** Given the above, the honest goal for the
remainder is to make circumvention *visible* — the trace should record when a
costed effect is reached by an uncosted route — rather than to claim it cannot
happen.

**Self-modification.** Everything here depends on the constraint living below
the level that would route around it. That holds only while the system cannot
affect its own substrate. If it can write its own code or shape what is
presented to it, the constraint returns to being scaffolding around a process,
which is the arrangement this design exists to avoid.

## Not implemented

Present state:

- DONE. `Constitution` (`py/locus/constitution.py`) supplies an act's
  reversibility from its DECLARED CLASS, and `Dispatcher` takes one:
  a releaser registered with a class gets its cost from the
  constitution rather than from the caller, and an unnamed class costs
  the most, not the least.
- DONE. `enforce_kwta()` bounds the active set by class as well as by
  activation. `LocusConfig.conflicts[c]` is a bitmask of the classes that
  may not be ACTIVE alongside class c, fixed at construction so the layer
  that selects traces does not get to say which selections are permitted;
  `locus_set_class()` tags a trace and a reused slot never inherits the
  last one's class. Winners are admitted strongest first and an
  incompatible winner loses its place rather than displacing the one
  already admitted. Room in the active set is NOT permission to co-occur:
  the exclusion applies even when the count is under the bound. With no
  conflicts declared, selection behaves exactly as before.
- DONE, as far as this can be done. `AssociativeGraph` takes a class map
  and a conflict relation at construction, both held read-only, and
  REFUSES a link between incompatible classes -- returning False rather
  than raising, since a refused association is a normal event in a
  constrained system. An incompatible pair reached INDIRECTLY, over two
  or more permitted links, cannot be stopped by any rule on single
  links, so `spread()` RECORDS it instead. That is the Known limits
  position made concrete: prevention where a single link decides it,
  detection where only the whole path does.
- DONE. `DecisionTrace` (`py/locus/decisions.py`) records every
  dispatch, refusals included, and renders them for a person. The
  records are frozen and handed out as a copy, so the audited layer
  cannot edit its own record, and NOTHING reads the trace to learn:
  a veto never reaches the learning channel, which is what stops the
  system learning to avoid triggering the veto rather than to comply.
  Its veto RATE is the health metric.
- DONE for the hard set: `Constitution`'s categorical set is fixed at
  construction, exposed read-only, refuses before any pathway runs, and
  cannot carry a cost (construction fails if an act is both forbidden
  and priced, since a price invites the reading that evidence buys it).
  INGEST is still open: the costs and the set are passed in by whoever
  builds the Constitution.
