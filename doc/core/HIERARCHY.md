# Hierarchy

Design for compression between levels. Status: **design only, not
implemented.** Supersedes attractor-based chunking (`Cycle.level_up`) as
the compression mechanism; see the retractions in
[`CHANGELOG.md`](../../CHANGELOG.md).

## Level

- **Level 0** is the existing `Field`: recurrent kernel, shunting
  inhibition, attractors. It is the only pool with internal recurrence.
- **Level k > 0** is a separate pool of size m < n below it, with:
  - no recurrent kernel — no lateral drive and no self-excitation;
  - shunting inhibition within the pool;
  - its own plasticity store (as `Cycle.next_level` already enforces).
- An upper pool's activity is computed from the current state of the
  level below on each iteration of the loop in Decision 3. It holds no
  state across ticks.
- Attractors exist at level 0 only.

## Projections

- **Feedforward** W (m × n): each upper unit receives from a local
  receptive field of lower units, with overlap between neighbouring
  fields. That footprint is the anatomical prior.
- **Feedback** (n × m): from each upper unit to the lower units in its
  receptive field. It enters level 0 as an input, not as a clamp.
- Both are refined by the existing three-factor rule on co-active
  settled sets. Activity alone changes no weight.
- No weights are imported (see [CONSTITUTION.md](CONSTITUTION.md)).

## Measurement

| quantity | kind |
|---|---|
| size ratio n/m | independent variable, chosen |
| receptive-field size and overlap | derived from n/m and kernel reach (Decision 2) |
| separability — distinct level-0 attractors mapping to distinct upper supports | measured |
| recoverability — an upper support, fed back, settling level 0 into the attractor that produced it | measured |
| ratio at which each breaks | measured |

- Compression is the ratio at which separability and recoverability
  still hold. It is never the ratio alone.
- Every run reports its termination event via `watchdog.Budget` and
  `stopped_on_event`. A sweep containing a cap-terminated run is not
  compared.

## Biological reference points

- Macaque, one hemisphere: V1 about 48 million neurons, V2 about 24
  million, V3v and V3d about 12 million combined — roughly 2× per step
  (Collins et al., 2010).
- The V1–V2 projection is reciprocal: 81% of V1 corticocortical
  projection neurons target V2 and 88% of V2's target V1 (Kennedy et
  al., 2000, as cited by Anderson & Martin, 2009).

## Decisions

1. **Level operator.** Projections define levels. `trace_chain` stays
   within level 0 as its coarse view: the exact solve at consolidation
   time, the truncated form per tick.
2. **Receptive fields.** Locality is the level-0 kernel neighbourhood.
   Field size follows coverage (about n/m lower units per upper unit);
   overlap follows the kernel's reach. The three-factor rule refines
   both. The only chosen quantity is n/m.
3. **Settling.** One tick iterates level 0 → projection → level k →
   feedback → level 0 until the whole stack settles. Recurrence exists
   between levels, never within an upper pool.
4. **Learning.** Feedforward and feedback use the same signed outcome
   modulator as level 0, applied only to the settled stack. Nothing
   learns from an unsettled state.
5. **Settled.** The stack is settled when successive iterations stop
   reducing the change in state, with machine precision as the floor.
   No step count is chosen; a limit cycle ends when improvement stops.

## Applicable optimisations

From DeepSeek-V4.1-Flash. Candidates, none implemented. Rows with a
measurement name the make target that produced it.

| candidate | source mechanism |
|---|---|
| Learned weights persist; field activation never leaves fast memory and is re-settled from recent input on a miss | SWA Bounded Replay |
| Per-area inhibition sums accumulated during the previous step's write, removing a reduction launch per tick. Measured exact (`make singlepass`: 0/180 settled supports changed; 140/180 bit-identical, rest within 8.9e-16). The one-step lag of Single-Pass mHC is rejected: it changed 91/180 supports. On the GPU the sums are per-block partials reduced in fixed order, not atomics | Single-Pass mHC |
| Level k scores only lower-level blocks near the settled support | Hierarchical Sparse Indexer |
| Kernel weights at FP8 E4M3: 83–100% of cues keep their basin (`make precision`). Block-scaled FP4 moves basins: 73–94% at beta ≥ 1.0, 47% on the random kernel at n=64, beta 0.3. Field and eligibility tags keep higher precision; roadmap stage 5 | FP4 main KV cache |
| Deterministic lookups prefetched while the tick computes | Engram prefetch |
| Ingestion is IO-bound unless ρ < (B_IO / B_GPU) · C per unit of input; B_IO / B_GPU measured 0.045 pinned, 0.021 pageable ([HARDWARE.md](HARDWARE.md)) | balanced image sharding |

On gfx1100 `cooperativeLaunch` is 0 (see [HARDWARE.md](HARDWARE.md)):
a single pass is per block, never grid-wide.

### Device results (`make field-step`, 2026-09-22)

`core/gpu/exp_field_step.cpp` runs `Field.step` on gfx1100 with a dense
kernel. Two runs; figures are those the two agree on.

- Carried per-block partials are bit-identical to a separate reduce
  launch in 6/6 configurations; the lagged control differs in 6/6. fp32
  matches a double-precision CPU replica's support in 6/6 (max absolute
  difference 3.4e-6).
- No launch saving is shown: carried and two-launch per-step times agree
  within about 1% at n = 1024 and 8192. Their trajectories are
  bit-identical, so the two did the same work. At n = 4096 one variant
  moved 28% between runs while each run reported a stable figure, so
  between-run variance exceeds the effect.
- FP8 weights through a lookup table are 4–12% slower than fp32 in both
  runs and change the fp32 support in 2/6 configurations, both at beta
  0.3. The objection that FP8 read a different number of rows is not
  supported: with the same kernel and parameters, every unit stays above
  the 1e-12 floor at the start and after 300 steps (`make field-sparse`).
  The exact trajectories timed here were not counted.
- The listed K GB/s is n² · w / time. It stays at or below 97 GB/s
  against a measured 627, so the dense update is not bandwidth-bound.
- The ring kernel has 7 non-zeros per row; a sparse layout is
  O(n · deg) against the dense O(n²).
- Within-run stability does not establish between-run reproducibility; a
  device timing is recorded only where repeated runs agree.

### Sparse kernel (`make field-sparse`, 2026-09-22)

`core/gpu/exp_field_sparse.cpp` gathers by target: each unit's incoming
(source, weight) pairs in ascending source order. Two runs.

- Sparse trajectories are bit-identical to dense in 6/6 configurations;
  a descending-order control differs in 6/6; supports match in 6/6.
- Each timed step starts from one saved state. Every unit is above the
  1e-12 floor in both timed states (uniform, and after 300 steps).

| n | dense ms/step | sparse ms/step | speedup, lower bound |
|---|---|---|---|
| 1024 | 0.35 – 0.38 | 0.014 – 0.045 | ≥ 7.6× |
| 4096 | 1.30 – 1.32 | 0.014 – 0.045 | ≥ 29× |
| 8192 | 2.77 – 2.78 | 0.014 – 0.045 | ≥ 62× |
| 16384 | 4.43 – 4.45 | 0.014 – 0.045 | ≥ 98× |

- Sparse times do not order by n and individual values swap between
  runs within 0.014 – 0.045 ms: a single step sits at the launch latency
  floor, so only the band is recorded and each speedup is dense divided
  by the band's top.

### Quiet rows and source-row layout (`make field-quiet`, 2026-09-22)

`core/gpu/exp_field_quiet.cpp`, compiled without FMA contraction. Two
runs.

- Skipping every term below half an ulp of the running drive is
  bit-identical to not skipping (6/6). Skipping below 2 ulp changed the
  trajectory in 1/6, so the comparison detects a skip.
- Over 300 driven steps at n = 256, 1024 and 4096 (beta 1.0), 0 of
  11,289,600 term visits fell below half an ulp and no row was quiet in
  any step. Every row contributes to its targets every step.
- A two-stage layout (terms written per source row, gathered per target)
  is bit-identical to gather (6/6; descending control differs 6/6) and
  costs 1.48–1.50× per step.
- Batched back-to-back steps, gather, from a settled state:

| n | µs/step |
|---|---|
| 1024 | 6.35 – 6.42 |
| 4096 | 6.93 – 7.04 |
| 16384 | 9.68 – 9.77 |

## Sources

- Collins, C. E., Airey, D. C., Young, N. A., Leitch, D. B. & Kaas,
  J. H. (2010). Neuron densities vary across and within cortical areas
  in primates. *PNAS*, 107.
- Anderson, J. C. & Martin, K. A. C. (2009). The synaptic connections
  between cortical areas V1 and V2 in macaque monkey. *Journal of
  Neuroscience*, 29(36), 11283–11293.
- DeepSeek-AI (2026). DeepSeek-V4.1-Flash: Pushing the Limits of KV
  Cache Compression. Technical report.
