# Attention

Attention windows, defined from biology rather than from the
transformer sense of the term. A transformer's attention window is a
fixed span of input every prediction attends over. Nothing here has
one, and adding one would import the mechanism this design exists to
avoid: what is in play is settled by competition, not by a span.

Biology gives five distinct windows. They are different quantities and
none of them is a byte count.

## 1. The sampling window — one theta period

Attention samples rhythmically rather than continuously: behavioural
detection oscillates at ~7 Hz, with gamma cycles nested inside
(VanRullen & Koch 2003; Landau & Fries 2012; Lisman & Idiart 1995).

The window is one slow period, and the unit inside it is one fast
update.

Built: `Field.cycle(fast=7)` — one slow period running seven fast
updates, the theta/gamma nesting ratio. `Cycle.task_step` runs
settle → decide → learn → orient once per period.

## 2. The capacity window — items, not bytes

Four or so items, and the limit is set by mutual interference, not by
storage (Cowan 2001).

Built: `active_k` in `LocusConfig` bounds the active set, and
`enforce_kwta` resolves the competition. `LOCUS_KWTA_MAX` bounds the
noisy path's working set.

Open: `active_k` is configured. Biology sets it by interference among
what is active, so it should fall out of the field's own dynamics
rather than be chosen.

## 3. The selection window — biased competition

Attention is not a gate in front of processing. Competing
representations run in parallel and a top-down bias tilts the
competition; the winner is the outcome of the competition, not of the
bias (Desimone & Duncan 1995).

Built: `Cycle.orient` derives the next bias FROM the learned weights,
so a bias is a consequence of outcomes. `bias_strength` scales it.
`Field.step` already runs the competition the bias enters.

Open: the bias comes from learned association only. Nothing estimates
utility or priority, which ROADMAP names under "Beyond the memory
substrate".

## 4. The integration window — a consequence of level

Cortical areas integrate over progressively longer windows, from tens
of milliseconds in early sensory areas to minutes in higher ones
(Hasson et al. 2008). The length is a property of where a region sits,
not a parameter set per region.

Not built. `doc/core/HIERARCHY.md` specifies the settle loop that would
produce it: a level's settled active set is the input to the level
above, so effective span grows by composition. Depth and compression
are outputs there, and integration window is the same kind of quantity.

## 5. The refractory window — what attention cannot do twice

Two limits, both consequences of occupancy rather than rules:

- Inhibition of return: a recently attended location is disfavoured for
  ~300 ms (Posner & Cohen 1984).
- Attentional blink: a second target is missed ~200–500 ms after the
  first, while the first is being consolidated (Raymond et al. 1992).

Partly built: `heat` decays slower than activation, so usage history
outlives the moment, and the labile window in `store.c` holds a
retrieved trace open before it restabilises. Neither implements return
inhibition directly.

## What this means for the input path

An attention window bounds RESIDENCY: what must be held is what is
currently in play, not the whole stream. The tiering work in
`doc/core/HARDWARE.md` streams a window of the training input for
hardware reasons; these five windows are the reason a model does not
need the whole stream in the first place. The two meet, and the
hardware one should not be designed as if it were the cognitive one.

## Measuring before building

The effective window LocusAI has today is unmeasured. It is bounded
above by `NgramEncoder(orders=(2, 3, 4))` — three bytes of context per
unit — and below by whatever the store's own dynamics carry between
periods. That measurement comes before any change to the mechanism.
