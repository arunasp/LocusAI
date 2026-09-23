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
                         core/tests/*.c core/tests/*.py core/*.md \
                         doc/core/*.md) \
              core/Makefile core/requirements.txt

.PHONY: lint test build deploy verify e2e all core pack unpack wheels \
        digests gitops-verify commit-verified squash-for-push groups

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

# --- delivery -------------------------------------------------------------
# cicd/gitops.py: each step is one call with its checks built in. Verified
# against clean, dirty, no-op and failure scenarios by `make gitops-verify`
# (throwaway repos, nothing here is touched).
#
# COMMIT MESSAGES LIVE IN AN IGNORED WORKING DIRECTORY, not under .git/.
# Hiding a file inside git's own directory is .gitignore's job done in the
# wrong place: it is undeclared, invisible to anyone reading the repo, and
# it BREAKS OUTRIGHT in a linked worktree or submodule, where `.git` is a
# FILE containing a gitdir: line -- `mkdir .git` fails there, so the path
# cannot be created at all. `.locus/` is declared in .gitignore, so writing
# a message still never dirties the tree, and the mechanism is visible
# where people look for it.
GITOPS ?= $(PYTHON3) cicd/gitops.py
MSG    ?= .locus/commit-msg.txt

# NAMED FILE SETS. The same lists were being typed by hand on every
# commit -- four times in one session for the babble set alone -- and a
# hand-typed list is where a file gets forgotten: a Makefile target
# committed without the script it calls, a test without the tool it
# tests. GROUP=<name> resolves one of these; FILES= still takes an
# explicit list for anything that is genuinely one-off.
#
# A group is the unit that has to land TOGETHER to leave the tree
# consistent, which is why the .cpp, its packer and its experiment are
# one group and not three.
GROUP_babble  = core/tests/exp_babble.py core/tests/exp_selftrain.py \
                core/tests/exp_selftrain_gpu.py core/tools/babble_gpu.py \
                core/gpu/babble_device.cpp
GROUP_audit   = core/tools/audit.py core/audit-baseline.json
GROUP_jobs    = core/tools/jobs.py core/tests/test_jobs.py
GROUP_perfmon = core/tools/perfmon.py
GROUP_store   = core/src/store.c core/src/locus.h core/tests/test_store.c
GROUP_docs    = $(wildcard doc/core/*.md) CHANGELOG.md
GROUP_cicd    = Makefile cicd/groups.py cicd/gitops.py

# FILES wins when both are given, so an explicit list can always
# override a group without editing it.
RESOLVED = $(if $(FILES),$(FILES),$(if $(GROUP),$(GROUP_$(GROUP)),))

groups: ## List the named file sets and check every path exists
	@$(PYTHON3) cicd/groups.py $(MAKEFILE_LIST)

digests: ## Size and sha256 of FILES (read-back after a transfer)
	@$(GITOPS) digests $(FILES)

gitops-verify: ## Verify gitops.py against throwaway repos (23 scenarios)
	@cd cicd && $(PYTHON3) verify_gitops.py

commit-verified: ## Pipeline, then stage exactly FILES or GROUP=<name> and commit
	$(if $(RESOLVED),,$(error commit-verified: give FILES= or GROUP=; \
	  `make groups` lists the names))
	@mkdir -p $(dir $(MSG))
	@$(GITOPS) commit-verified --msg $(MSG) -- $(RESOLVED)

# DESTROYS per-commit history on the current branch. The surviving copy is
# the backup/pre-squash-<UTC stamp> branch it creates before moving anything.
squash-for-push: ## Squash all commits ahead of upstream into one, backup branch first
	@$(GITOPS) squash --msg $(MSG)
