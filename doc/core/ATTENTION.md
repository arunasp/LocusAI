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

Built, and it now falls out of interference rather than being set.
`competing_total` sums the activation a trace competes with and
`effective` divides by it, the same divisive term `Field.step` runs,
with `beta` the same quantity. Promotion is settled strongest-first
against the winners so far, so admitting one raises the bar for the
next. `active_k` remains as a bound on the working set.

Measured, cueing n traces at equal strength with active_k and slots
set high enough that neither binds: 1, 2, 4 promote in full, then the
set levels off however many are cued (8, 16, 32 all give the same).
The limit is arithmetic, not chosen: a/(1 + beta*won) >= 1. It was
seven before the refractory depression and is FIVE after, because the
depression lowers every drive -- the number is a consequence of two
mechanisms, which is why it is not written anywhere as a setting.

What this replaced, measured before the change: 32 cued traces were
each as strong as one, and the active count simply tracked the cue
count -- `enforce_kwta` truncated a list rather than resolving a
competition.

## 2b. One window per map, in priority order

A single pool is wrong for a reason observation makes plain: focused
work, background sensing and internal reasoning run AT THE SAME TIME,
in priority order, not in turns. With one pool a saturated focal load
silences the rest -- measured: seven declarative winners and nothing
left over.

Biology separates the maps rather than sharing one budget:
interference is strongest between similar representations (Desimone &
Duncan 1995), a goal-directed dorsal set runs alongside a
stimulus-driven ventral monitor (Corbetta & Shulman 2002), and the
short-term stores for different content do not compete with each other
(Baddeley 2000).

Built: competition is WITHIN a pathway. Declarative, procedural and
instinct each carry their own capacity window, so a busy foreground
cannot starve a monitor. Measured with a declarative load of 1 to 32
against two procedural and one instinct trace: declarative saturates
at 7, procedural stays 2, instinct stays 1, unchanged at every load.

## 2c. Priority between pools is an interrupt, not a ranking

Phasic noradrenaline interrupts ongoing network activity and
reorganises which network determines the output (Bouret & Sara 2005;
Dayan & Yu 2006). Tonic level sets the regime: moderate while engaged
in a focused task and filtering, high while uncommitted and responsive
to unanticipated change. So the pools are not ranked -- one scalar
decides how porous they are, and it is transient.

Built: the phasic component of tone IS the coupling between pools. At
baseline each pathway settles alone; while a salient event holds tone
above baseline, the winners of one pool raise the bar in the others,
so the foreground gives up slots. `note_salient` drives it and
`gate_decay` ends it without anything resetting it.

Measured, sixteen declarative traces against one procedural:

  salience 0, background at 2.0   tone 1.00   focal 7
  salience 0, background at 6.0   tone 1.00   focal 7
  salience 2, background at 2.0   tone 2.00   focal 6
  salience 6, background at 2.0   tone 4.00   focal 4

A merely STRONG background trace costs the foreground nothing; a
SALIENT one costs it slots, and more of them the louder the event.

Not built, deliberately: whether breakthrough should require the event
to match current top-down settings. Contingent capture (Folk et al.
1992) says yes and the additional-singleton results say no, and load
theory's account of what a busy foreground filters out has recent
replication failures. The interrupt is undisputed; the condition on it
is not, and a disputed result does not belong in the substrate.

INSTINCT SITS OUTSIDE ALL OF THIS: `locus_put` pins an instinct trace
ACTIVE, and pinned traces are excluded from the competition. A reflex
does not queue for attention. The consequence for the three windows
above: the background MONITOR is procedural, not instinct -- measuring
it as instinct measures a pinned trace and proves nothing.

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

Built, and biphasic. Biology is not a flat suppression: detection
improves for ~100-300 ms after attending and worsens from ~500-3000 ms
(Posner & Cohen 1984; Samuel & Kat 2003 on the time course). The
depression sits on the INPUT -- superficial collicular visual neurons
are depressed in cue-target-compatible conditions while intermediate
neurons are not less sensitive to direct stimulation (Satel et al.
2011; Dukewich 2009). The trace keeps what it has; what would
re-orient to it is weakened.

`locus_excite` therefore divides the incoming drive by heat, which is
the record of prior response and is carried in the same units as
drive. NO COEFFICIENT: a number in front of heat would be fitted, and
the competition's beta -- tried first -- gave a 9% cut with no
crossover inside twenty ticks, because it is the wrong quantity.

Measured, a recently attended trace against a fresh one at equal
drive:

  away 1   revisited 2.1107   fresh 1.7000   both ACTIVE
  away 2   revisited 1.9427   fresh 1.7000   both ACTIVE
  away 4   revisited 1.6853   fresh 1.7000   revisited ARCHIVE
  away 8   revisited 1.3885   fresh 1.7000   revisited ARCHIVE

Facilitation for two ticks, then the revisited trace is worse off than
a fresh one and loses its slot. The crossover needs no window and no
timer: activation decays fast and gives the early advantage, heat
decays slowly and gives the later cost, and both rates already
existed.

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
