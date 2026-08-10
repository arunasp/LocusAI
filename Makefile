# Standard cicd_runner target vocabulary mapped onto LocusAI's own
# tools/pipeline.sh stage names (lint/test/build/server/verify/all),
# which stay unchanged there. Lets calls through cicd_runner use the
# same targets as every other project:
#   run_in_directory(relative_path="LocusAI", binary="make",
#                    args=["lint"])
# See cicd_runner's examples/project-skeleton/ for the convention.
#
# The substrate under core/ has its own self-contained pipeline with
# the same vocabulary -- it needs no Docker and is not part of the
# targets below. Use the `core` target, or run make inside core/.
# /etc/cicd-common.mk is baked into cicd-runner's own images; the
# vendored copy beside this file covers a plain checkout. Unlike
# core/ -- which a worker mounts on its own, so a sibling copy can
# never resolve there -- the repo root has BOTH visible at once
# inside a worker, and including both makes make warn about a
# duplicate `help` recipe (confirmed live 2026-08-10). Prefer the
# image's copy when it exists.
ifeq ($(wildcard /etc/cicd-common.mk),)
-include cicd-common.mk
else
include /etc/cicd-common.mk
endif

PYTHON3    ?= python3
CICD       ?= cicd
TRANSFER   ?= $(CICD)/transfer.py
PAYLOAD    ?= $(CICD)/payload.b64
WHEELHOUSE ?= $(CICD)/wheels
REQS       ?= core/requirements.txt

# Explicit rather than a $(shell find ...): the set is deterministic, needs
# no git or shell at parse time, and never sweeps up build/ or .venv/.
PACK_FILES ?= $(wildcard core/src/*.h core/src/*.c core/py/locus/*.py \
                         core/tests/*.c core/tests/*.py core/*.md) \
              core/Makefile core/requirements.txt

.PHONY: lint test build deploy verify e2e all core pack unpack wheels

lint: ## Static checks (tools/pipeline.sh lint)
	tools/pipeline.sh lint

test: ## Unit tests, no Docker needed (tools/pipeline.sh test)
	tools/pipeline.sh test

build: ## Pack the .mcpb extension (tools/pipeline.sh build)
	tools/pipeline.sh build

deploy: ## Start the MCP server container (tools/pipeline.sh server)
	tools/pipeline.sh server

verify: ## Endpoint health + one real run_command (pipeline.sh verify)
	tools/pipeline.sh verify

# Alias for verify, not a distinct stage -- tools/pipeline.sh's own
# verify already exercises the deployed pipeline end to end (endpoint
# health check AND one real run_command call through call.sh), so
# there is no separate, more thorough suite to point at yet. Kept as
# its own target for vocabulary consistency across projects; point it
# at a real e2e suite if one is ever built.
e2e: verify ## Alias for verify

all: ## Run the full tooling pipeline (tools/pipeline.sh all)
	tools/pipeline.sh all

core: ## Run the core substrate's own pipeline (no Docker required)
	$(MAKE) -C core all

# --- data transfer -------------------------------------------------------
# The Filesystem connector writes UTF-8 only, so a source tree crosses as
# base64 text rather than as an archive. cicd/transfer.py checksums the
# payload and verifies it before extracting anything.
#
# The payload is a real file target with real prerequisites, so make skips
# repacking when nothing changed -- the same caching discipline core/'s
# .deps sentinel uses, applied to the transfer.

$(PAYLOAD): $(PACK_FILES)
	@mkdir -p $(CICD)
	$(PYTHON3) $(TRANSFER) pack $@ $(PACK_FILES)

pack: $(PAYLOAD) ## Pack core/ sources into one checksummed text payload

unpack: ## Verify and extract a received payload, then remove it
	$(PYTHON3) $(TRANSFER) unpack $(PAYLOAD)

# Wheelhouse: resolve dependencies once, then carry them with the payload.
# A worker that receives the wheels never touches the network, which is what
# makes repeated rebuilds of an ephemeral container cheap.
$(WHEELHOUSE)/.stamp: $(REQS)
	@mkdir -p $(WHEELHOUSE)
	$(PYTHON3) -m pip download -q -r $(REQS) -d $(WHEELHOUSE)
	@touch $@

wheels: $(WHEELHOUSE)/.stamp ## Populate an offline wheelhouse from requirements
