# LocusAI

A biologically-grounded substrate for a synthetic mind. LocusAI is not a
transformer and does not load pretrained weights: behaviour comes from the
structure of its pathways and the semantics of its memory store, not from
learned parameters.

The name is doubled. *Locus coeruleus* is the noradrenergic system that
turned out to be the single signal unifying consolidation gating,
exploration/exploitation balance, and reconsolidation. *Locus* is also
plainly "place" — which is the project's method: strip each biological
mechanism to its computational function, then find where that function
belongs on the substrate.

## Repo layout

- `core/` — the substrate itself. C for the trace store, Python for the
  pathway layer. Self-contained, needs no Docker.
  See [core/README.md](core/README.md) for usage and
  [doc/core/ARCHITECTURE.md](doc/core/ARCHITECTURE.md) for the design
  rationale.
- `doc/` — design documents, one directory per component.
  [doc/core/](doc/core/) holds the substrate's architecture, constitution,
  roadmap and references.
- `tools/` — local dev tooling: git/bash MCP servers for Claude Desktop
  integration, and `pipeline.sh`. See [tools/README.md](tools/README.md).

## Build

The root `Makefile` uses the standard lint/test/build/deploy/verify/e2e
vocabulary, mapped onto `tools/pipeline.sh`'s own stage names, so the repo
drives identically to the other projects in this family.

```sh
make help    # list targets
make lint    # static checks
make test    # unit tests, no Docker needed
make all     # full tooling pipeline
make core    # the substrate's own pipeline (no Docker)
```

`deploy` and `verify` start the MCP server container and exercise it, so
they need a real Docker daemon. Everything else runs anywhere.

## Status

`core/` builds and passes its suites. What is implemented, what is
deliberately deferred, and what is still missing are tracked per design
constraint in [doc/core/ARCHITECTURE.md](doc/core/ARCHITECTURE.md); `tools/`
keeps its own verified/not-verified record in
[tools/README.md](tools/README.md).

No GPU backend exists yet. Nothing in this repo has run against a compute
device.
