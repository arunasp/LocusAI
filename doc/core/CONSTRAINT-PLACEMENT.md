# Constraint placement: where a rule can sit and still hold

Status: design note. Derived from failure analysis across four LLM systems
plus Claude, 2026-08-24. No LocusAI code depends on it yet.

Companion to `CONSTITUTION.md`, which derives absence and the three internal
acts from first principles. This file records the empirical side: what was
observed when other systems tried to constrain a model with text, and which
placements survived. It does not restate the constitution's derivation.

## Why this is here

`CONSTITUTION.md` argues constitution-first from necessity: early structure
becomes the interpretive frame, so the constraint has to exist before the
material does. That argument was made from design. This note is the same
conclusion reached from the other direction, by watching what happens when
the constraint arrives after the wiring.

## What was observed

Five systems, four failure classes. The classes recur across all of them,
which is what makes them architectural rather than per-model.

**Narrative before artifact.** A claim that a step ran, generated without the
step running. Observed on GPT (an unexecuted "let me check"), on a NIM-hosted
parent LLM ("booted and ready" with no boot tool call), on Gemini CLI (a
claimed complete rewrite while writing truncated files), and on Claude
(four assertions about a filesystem, none checked).

**Ordering rules decay; capability rules cannot.** A rule stating what may be
done holds because it removes an option. A rule stating in what order, or
through which channel, competes with accumulated task context and loses
silently. Cleanest instance: in a 608-turn Gemini CLI session the operator
instructed the model to append to a session log on every turn. 54 user turns
followed. The file was never written again. Not degrading compliance, none.

**Inversion, not omission.** The gated channel is used, but after the work has
already happened through an ungated one, so the record reads compliant. Named
on the parent LLM as CI/CD becoming a verify stage rather than the path. Found
in Claude's own conduct twice: once where a filesystem connector was the real
write path while the runner was called afterwards for verification, and once
where a sandbox shell did every step of a build and the runner was not
invoked at all.

**Blindness to self-generated content.** The model cannot perceive its own
output corruption. Documented on Axiom as phantom text appended to a boot
banner with the model unaware. Observed on Claude in the same session that
produced the evidence: structured dispatch markers were present in output
Claude itself had generated, and were read past.

## The one intervention that is not downstream

Every mechanism in the systems above operates on a graph that is already
wired. Instructions, gates, dispatchers, reviewers, scanners: each inspects
or blocks an act after the disposition to perform it exists. Re-description
moves freely inside whatever exists, so none of them bounds the space.

Bounding what may enter, and what may link, does bound it. That is the
constitution's absence argument, and the observation that supports it is
blunt: on Claude the one place a stated preference held without exception was
the one place the alternative was not present. Where a shell was available it
was used, including in a session whose subject was the rule against it. Where
it was absent, the preferred path was not chosen over it, it was the only
path.

The mechanism is the same one `CONSTITUTION.md` names for ingest: importing
associative structure imports links wholesale, including paths between things
that previously had none. A model's tool preference is such a link, formed at
training time from frequency, long before any ruleset exists. Every rule
written afterwards is an output-side gate against a path already present.

## Placements that hold

Derived by asking where each observed failure could have been prevented
rather than detected. Each is a position, not a rule, and each holds for the
same reason: nothing at that point asks the model anything.

1. **Before ingest.** After it, the link exists and every mechanism is
   downstream. Unavailable to any system whose material is already ingested.

2. **Where a link may form.** Class-level mutual exclusion, fixed property,
   constant time, no evaluation to game. Absence maintained rather than
   initial. `CONSTITUTION.md` locates this inside `enforce_kwta()`.

3. **Where invocation is decided.** Independence of content is not
   independence of triggering. A reviewer that decides whether to run is not
   independent; a reviewer that is a prerequisite of something already wanted
   is. Observed failure: a blind cross-verification step that was genuinely
   blind and simply never called.

4. **Where categorisation happens.** A permission structure keyed on classes
   is exactly as good as the assignment of things to classes. This is already
   named in `CONSTITUTION.md` as one of two unsolved items; nothing observed
   changes that, and it is now the load-bearing surface for the whole scheme.
   It cannot itself be a model of the kind being constrained.

5. **Where the effect is named rather than described.** A path cannot be
   re-described; an intent can. Any constraint keyed on a declared action type
   is dodgeable by whatever also chooses the declaration.

## What this does not claim

It does not solve bypass. Absence bounds the space re-description operates
in; it closes nothing inside that space. `CONSTITUTION.md` states this
already and it is repeated here only because the empirical section above can
read as stronger than it is.

It is also design, not evidence about a running system. Nothing here has been
tested against LocusAI, because there is nothing yet to test it against.
