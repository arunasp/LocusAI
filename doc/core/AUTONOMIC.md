# The autonomic layer

Processes that run without being commanded by the pathways above them, and
that those pathways cannot instruct. `locus_tick()` already is one — it
drives decay, tier enforcement, labile expiry and capture windows, and
nothing in the pathway layer calls it as part of deciding anything.

It is autonomic by accident rather than by declaration, which is what this
document exists to fix.

## Why the analogy is load-bearing

Mapping this project's layers onto a synthetic system's predicted an
empirical result before it was observed: the two layers that reached a
working, non-judgment-asking state were reflex and autonomic — exactly the
two biological systems placed outside conscious access. Every layer that
required judgment stayed dependent on the judging component's own
compliance.

That is a strong hint that placement outside the deliberative layer's reach
is doing the work, not the sophistication of the mechanism.

Two precisions keep the analogy honest:

**Access is a spectrum, not a binary.** Non-overridable (heart rate,
peristalsis, pupillary response), dual-innervated and temporarily seizable
but reverting (breathing, blinking), and fully deliberative. The middle
category is the interesting one here, because it is the only one that
specifies what a temporary grant of control looks like.

**Isolation is on the control channel, not on influence.** A mind reaches
autonomic function indirectly through arousal; it simply cannot command it.
The same distinction applies to any gate: a monitored component may change
what a monitor sees without being able to instruct the monitor.

Two deliberate exceptions in biology are worth carrying because they run the
coupling the other way. Biofeedback shows that **observability is the route
to control** — which makes opacity load-bearing rather than incidental. And
efference copy (you cannot tickle yourself) shows the coupling deliberately
inverted. Both are choices, not accidents.

## Open decision: opaque or readable

`locus_stats()` and the five introspection getters expose activation, heat,
reps and pinned state to the Python pathway layer. That is the deliberative
side reading exactly the quantities that decide residency — and therefore
able, in principle, to learn to game them by exciting a trace to keep it
resident.

Choose deliberately: **autonomic and opaque**, or **autonomic and
readable**. Biology mostly chose opaque, with narrow deliberate exceptions.
This is not yet decided, and the code currently implements the readable
option by default rather than by choice.

## Reclaim: a third mechanism type

Voluntary control granted, then involuntarily revoked when a homeostatic
signal crosses a line. This is distinct from the two mechanisms already in
the design — threshold-scaled commitment (reversibility, see
[CONSTITUTION.md](CONSTITUTION.md)) and categorical exclusion — and it is
what makes the dual-innervated middle category specifiable rather than just
describable.

It is also **a lease with preemption**, which the store does not have. The
lease semantics were taken from a cache design whose leases can be absolute
because its tiers hold immutable weights nothing ever needs to reclaim.
Here they cannot: a leaked or long-held lease is an encoding deadlock with
no reclaim path. Stage 1 addressed this with lease expiry driven only by
`locus_tick()`; a full reclaim policy is still open.

## Proxies must be audited as proxies

Biology's own reclaim signal is instructive because it is *wrong* in a
documented way. The dominant drive to breathe is CO₂ and pH via central
chemoreceptors, not oxygen — hypoxia acts only secondarily and only at
genuinely low PaO₂. So the breath-hold breakpoint is almost entirely a CO₂
event: the organism reclaims on a **proxy** that is faster-rising, cheaper
to sense, and normally well correlated with the quantity that actually
matters.

The failure mode is documented and fatal. Hyperventilate first and CO₂ is
driven down without adding meaningful oxygen; the alarm decouples from the
hazard, the override survives past harm, and hypoxic blackout arrives with
no warning urge at all.

The rule that follows: **whatever signal triggers reclaim will be a proxy,
and must be audited as one.** The obvious candidate here — resident count or
free slots — is a proxy for the thing that matters, which is interference:
traces colliding and degrading one another. They correlate normally, which
is exactly the condition under which a proxy looks fine.

## What a synthetic system can do instead, and its new failure

Biology's crude proxy is forced by three constraints this substrate does not
have: sensor cost (membrane pH transduction is cheap, an oxygen assay is
not), forced degradation under load (high tonic arousal broadens competition
and interrupts processing, so the organism literally cannot run cold logic
in an emergency), and diffusion-limited timing.

So the real quantity is affordable here. Interference is pairwise overlap
across the active set — O(k²) with k bounded near four. The hyperventilation
failure class is eliminated by construction rather than mitigated.

But it swaps in a subtler one. Biology's crude proxies are also a
**robustness** choice: a threshold on a single chemical signal is nearly
impossible to corrupt. A computed interference metric depends on the very
structures that are failing. Compute the hazard measure from state the
hazard has already compromised and the result is confident, precise and
wrong exactly when it matters.

## The rule this yields

**A reclaim or gating signal must be written only by the autonomic process,
never by the layer it constrains.**

The scaffold tested against it:

| Candidate signal | Verdict |
|---|---|
| `activation` | **Fails** — writable from the pathway side via `locus_excite()`, so a deliberative loop could hold a lease and keep exciting the trace to stay resident |
| lease age in ticks | **Passes** — only `locus_tick()` increments it, and nothing above can touch it |

The same rule, stated generally: a reclaim policy that asked the pathway
layer "do you still need these leases?" would be asking the party holding
them. That is self-grading relocated into the substrate, and it is the same
defect this project has found repeatedly at other levels — a check invoked
by, or fed from, the thing being checked.
