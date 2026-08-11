# LocusAI core

A biologically-grounded substrate for a synthetic mind. Three structurally
distinct pathways to an action over one tiered trace store.

LocusAI is not a transformer and does not load pretrained weights. Behaviour
comes from the structure of the pathways and the semantics of the store, not
from learned parameters.

## Why three pathways

An architecture with one pathway weighs every signal against every other, so
no signal can hold categorical priority. Biology does not work that way: a
reflex arc bypasses the cortex, and a fixed action pattern runs to completion
once released. LocusAI reproduces that as structure.

- **Instinct** — a discrete releaser table. An exact cue fires an
  all-or-nothing program. No scoring and no probability mass; a releaser
  either matches or it does not.
- **Procedural** — a consolidated correction whose cost falls with each
  confirmed re-application, until it pins.
- **Declarative** — retrieval and fresh judgement. Flexible, expensive, and
  the only route whose reads destabilise what they touch.

The fast route acts before the slow route finishes. The slow route may then
revise the outcome, but it never rewrites a pinned releaser.

See [ARCHITECTURE.md](../doc/core/ARCHITECTURE.md) for why the split is
structural rather than a weighting, what each constraint is answering, and
which parts are not implemented yet.
[CONSTITUTION.md](../doc/core/CONSTITUTION.md) covers how values enter the
system without deciding anything, and
[AUTONOMIC.md](../doc/core/AUTONOMIC.md) covers what runs without being
commanded — including the rule that a reclaim signal must be written only by
the process that owns it.
[METHODOLOGY.md](../doc/core/METHODOLOGY.md) states how these designs are
derived and what makes one finished;
[HARDWARE.md](../doc/core/HARDWARE.md) records the device facts the design
depends on and what follows from them.
[ROADMAP.md](../doc/core/ROADMAP.md) sequences the remaining work by
dependency; [REFERENCES.md](../doc/core/REFERENCES.md) sources the biology.

## Store semantics

The store is tiered, but it is not a cache:

- Tiers hold representations, not identical copies of one immutable blob.
- `retrieve()` leaves a trace **labile**; `restabilize()` must follow, and may
  rewrite the payload. Recall is reconstructive, so the read path is also a
  write path. A trace that is never re-stabilised weakens.
- `lookup()` does not destabilise. The instinct and procedural routes use it,
  because neither can afford a reconsolidation cost.
- The active set is bounded by **competition** (k-winners-take-all), not by a
  byte budget. Pinned traces are required residents and are exempt.
- A lease is absolute for its own slot. Competition then decides among the
  remaining eligible slots, so a leased trace survives even when it would
  otherwise lose.
- A weak trace can be rescued by an unrelated strong event nearby in time
  (synaptic tagging and capture), which no eviction policy provides.

Repetition-count promotion and tier promotion are the same operation:
`locus_reinforce()` pins a trace into the procedural pathway once it has been
successfully re-applied often enough.

## Layout

```
src/          C substrate: store, tiers, competition, capture
py/locus/     pathway layer: graph, novelty gate, consolidator, cascade
tests/        C and Python behavioural tests
```

C where bytes and timing matter; Python for the pathway logic above it.
Binding is ctypes, so there is no build-time dependency and no generated glue.

This directory is self-contained and has its own Makefile. The repository root
Makefile drives `tools/pipeline.sh`, which is a separate concern.

## Build

```sh
make all                 # lint, build, test
make deploy PREFIX=...   # install the CPU substrate
```

Requires a C11 compiler, Python 3, and `pycodestyle` for the lint target.

## Usage

```python
from locus import Consolidator, Dispatcher, Store

with Store(promote_after=3) as store:
    dispatch = Dispatcher(store, Consolidator(store))
    dispatch.add_releaser("looming", key=10, action="flee")

    outcome = dispatch.act("looming", judge=lambda payload: (0.9, "stay"))
    outcome.action    # "flee"   -- the fast route already acted
    outcome.revision  # "stay"   -- deliberation disagreed, after the fact
```

An irreversible action will not commit on a single crude cue:

```python
dispatch.add_releaser("prey", key=11, action="strike", reversibility=0.05)
dispatch.act("prey", signals=1).stage   # Stage.PREPARE, not Stage.COMMIT
```

## Status

Verified on CPU: store semantics, lease and eviction contracts, kWTA bound,
tagging and capture, promotion, the commitment cascade, and pathway dispatch.
43 C checks and 33 Python tests pass from a clean tree.

Not yet implemented: GPU backend, sensory encoding, motor output, sleep-phase
consolidation, and the DG-analog separation stage. No GPU code exists, so
nothing here has been run against `/dev/dxg` or any compute backend.
