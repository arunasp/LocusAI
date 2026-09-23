# Reopening plasticity

Status: designed, not implemented. Written 2026-09-23.

## Why

`make know` learns each unit at a metaplastic rate of 1/n over that
unit's own history. Measured consequences:

- Re-reading correct text changes nothing (`tests/exp_reread.py`:
  ±0.0001 bits per byte over three passes).
- A mistaken reading is corrected in one pass
  (`tests/exp_recover.py`: +0.004083 damage, −0.004183 after one pass of
  the correct text).
- Order dominates a curriculum (`tests/exp_curriculum.py`): whichever
  corpus is read first is the one the store is better at. Simple English
  before complex English scored 2.639901 on complex text against
  2.455813 for complex text alone with half the data.

The third result blocks the developmental curriculum: any arm measures
the rate schedule rather than the curriculum.

## Mechanisms, in the order to build them

### 1. Context inference

On sustained mismatch, open a new context-tagged set of rows instead of
overwriting consolidated ones. The readout uses the context whose recent
predictions fit the current stream.

- State: a context id per row set; per-context recent error.
- Trigger: a unit's error stays above its own baseline for a run of
  events, not a single event.
- Keeps: the old rows, unchanged.
- References: Bouton 2004 (renewal); Gershman, Blei & Niv 2010 (latent
  causes).

### 2. Reconsolidation gated by prediction error

A row that is used AND wrong becomes labile: its effective n drops, so
the next events move it. A row used and right re-stabilises.

- State: none beyond the existing per-unit count and the current error.
- Trigger: retrieval with error above the unit's baseline.
- References: Nader, Schafe & LeDoux 2000; Sevenster, Beckers & Kindt
  2013.

### 3. Expected versus unexpected uncertainty

Judge a unit's error against its OWN usual error, not an absolute
threshold. A unit that is always uncertain (the byte after a space) must
not relearn forever.

- State: per-unit running mean and spread of its error.
- Measured basis: `tests/exp_recognition.py` shows corrupted text reads
  near 3.5 bits per byte at every level of experience while clean text
  falls from 1.921 to 1.821, so an absolute threshold does not track what
  the store has learned.
- Reference: Yu & Dayan 2005.

### 4. Novelty into the fast store

New contexts form in the hippocampal store; the neocortical store
changes through replay at sleep.

- Reference: McClelland, McNaughton & O'Reilly 1995.

## Acceptance test

`tests/exp_shift.py` (to write): read corpus A, then corpus B, measuring
on held-out material from both.

| criterion | passes when |
|---|---|
| adaptation | bits per byte on B after the shift beats the current build |
| retention | bits per byte on A after reading B stays within noise of A alone |
| stability | a unit whose error is merely noisy does not relearn: its weights move less than a unit whose error rose and stayed high |
| order | with the mechanisms on, the curriculum arms in `tests/exp_curriculum.py` stop being decided by which corpus came first |

Corpora already on disk: `build/corpus-480e191` (code),
`build/corpus-ts-128` (TinyStories), `build/corpus-simplewiki`
(encyclopedic), plus the curriculum arms under `build/curric/`.

## Constraints

- Every quantity above is event-driven. No fixed rates, thresholds or
  lifetimes; a value that cannot yet be derived is recorded as initial
  state with its open question.
- The Python path stays the equality oracle for the device path.
- Figures must be reported with their resource cost.
