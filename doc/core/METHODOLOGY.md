# Methodology

How design decisions in this project are derived, and what makes one
finished. Recorded because the method is load-bearing: the same biological
mechanism yields very different code depending on which of these steps is
skipped.

## Strip to function, then re-express

Take a biological mechanism, strip it to its computational function, and
discard the biophysics. Then re-express that function in whatever dense,
batched, structured form the substrate actually rewards.

This is explicitly **not** literal biological fidelity. Neurons are
asynchronous, unstructured and per-unit; a GPU is none of those. Copying the
biology's implementation would inherit constraints that exist because of
membranes and diffusion, not because of what the mechanism is for.

## Check effects, not mechanisms in isolation

Every mechanism must be checked for its effect on the mechanisms already
mapped, not evaluated alone. Most of this project's real findings came from
that check rather than from the mechanism itself — a mechanism that looks
correct in isolation frequently contradicts the contract of one already
shipped.

Two examples that came from exactly this, both in
[ARCHITECTURE.md](ARCHITECTURE.md):

- Reconsolidation makes retrieval a write. That is fine for the declarative
  pathway and destroys the procedural one, whose whole value is being cheap
  and stable. The dispatcher had to be corrected so only the declarative
  route destabilises.
- Lease protection copied from a cache design works there because cached
  weights are immutable and nothing reclaims them. In a store whose entries
  compete for residency, absolute leases produce an encoding deadlock.

## Correctness at three levels

The target is correctness against real biology at all three of Marr's
levels — computational (what problem is being solved), algorithmic (what
representation and process), implementational (how it runs on the
substrate) — not merely being bio-inspired at the implementation level.

A mechanism reproduced at the implementational level without its
computational justification is decoration; a computational story with no
implementable form is not a design.

## Resource discipline

At implementation time, every byte, timing and instruction matters. The
substrate is written in C for the trace store and Python only for the
pathway layer, with a ctypes binding rather than cffi or pybind so there is
no build-time dependency. Defaults are chosen from the biology where the
biology gives a number (an active set bounded near four, after Cowan) rather
than from convenience.

## First-person observation as a design source

Lived observation is treated as a legitimate source of design questions, and
this project has repeatedly found that a reported observation lands on real,
independently characterised literature when it is checked — the procedural
pathway, salience-weighted encoding, reversibility-scaled commitment and
spreading activation all entered this way.

The limit is named rather than assumed. Introspection is reliable for
reporting **that** something happened and what it felt like, and much weaker
for **why** — Nisbett and Wilson (1977) documents confident, coherent,
confabulated explanations for one's own behaviour that are not the actual
cause. See [REFERENCES.md](REFERENCES.md).

So the standing practice, applied to every such observation regardless of
whose it is: take it seriously the moment it is reported, then check whether
the mechanism behind it is already independently characterised, rather than
accepting the first intuitive explanation for it.

## Build order

Scaffold first, then add accumulated design points incrementally against the
existing backlog — not a full-design-then-implement pass.
[ROADMAP.md](ROADMAP.md) holds the dependency order; the constraint that
matters is that a stage is not started before the one it depends on is
boringly reliable.
