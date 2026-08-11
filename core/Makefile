# LocusAI core -- CPU substrate (C) + pathway layer (Python).
#
# Uses the standard cicd_runner target vocabulary
# (lint/test/build/deploy/verify/e2e/all) so this directory runs
# unchanged through:
#   run_in_directory(relative_path="LocusAI/core", binary="make",
#                    args=["all"])
# See cicd_runner's examples/project-skeleton/ for the convention.
-include ../cicd-common.mk
-include /etc/cicd-common.mk

CC      ?= gcc
CFLAGS  ?= -std=c11 -Wall -Wextra -Werror -O2
PYTHON3 ?= python3
VENV    ?= .venv
PY       = $(VENV)/bin/python3
# store.c uses log() for the Gumbel transform in locus_kwta_noisy(), so the
# C side now needs libm explicitly. Kept as a variable rather than inlined so
# a platform where libm is part of libc can override it to empty.
LDLIBS  ?= -lm
HIPCC   ?= hipcc
GPUSRC   = gpu/probe.cpp
GPUBIN   = $(BUILD)/gpu_probe
PREFIX  ?= /usr/local
SRC      = src/store.c
HDR      = src/locus.h
BUILD    = build
LIB      = $(BUILD)/liblocus.so
TESTBIN  = $(BUILD)/test_store

# `gpu` is listed here for a reason that already bit this project once: the
# target shares its name with the gpu/ source directory, so without .PHONY
# make sees the directory, considers the target satisfied, and silently does
# nothing. Same collision the `build` target hit during the scaffold build.
.PHONY: lint test build deploy verify e2e all clean gpu

# Sentinel file, deliberately NOT in .PHONY: a .PHONY listing would
# force the install to re-run on every invocation regardless of
# whether requirements.txt changed. Same pattern as cicd_runner's
# examples/hello-python.
#
# Installs into a project-local .venv rather than the interpreter's
# own site-packages, matching cicd_runner's own ensure_venv() in
# deploy_cicd_runner.sh. Three reasons this is the right shape here:
# a worker running as the host uid:gid cannot write to system
# site-packages; Debian-family system Pythons refuse a plain install
# outright under PEP 668; and .venv sits on the bind-mounted project
# directory, so it survives between ephemeral worker calls. Package
# downloads still land in PIP_CACHE_DIR via cicd_runner's own cache
# mount, so a rebuild of the venv is cheap.
# WHEELHOUSE, when populated by the repo root's `make wheels`, makes this
# install fully offline: no index lookup, no download. Falls back to the
# network when it is absent, so a plain checkout still works.
WHEELHOUSE ?= ../cicd/wheels
PIP_OFFLINE = $(if $(wildcard $(WHEELHOUSE)/*.whl),--no-index --find-links $(WHEELHOUSE),)

.deps: requirements.txt ## Create .venv and install dev dependencies
	@test -x "$(PY)" || $(PYTHON3) -m venv $(VENV)
	$(PY) -m pip install -q $(PIP_OFFLINE) -r requirements.txt
	@touch .deps

# gcc -Wall -Wextra -Werror -fsyntax-only is a genuine static pass for
# the C side; pycodestyle covers the Python layer.
lint: .deps ## Static checks (gcc -fsyntax-only, pycodestyle)
	$(CC) $(CFLAGS) -Isrc -fsyntax-only $(SRC) tests/test_store.c
	$(PY) -m pycodestyle --max-line-length=79 py/locus tests

# No rule for $(BUILD) itself: it would collide with the phony `build`
# target and silently drop the order-only prerequisite.
build: $(LIB) $(TESTBIN) ## Compile the shared library and C test binary

$(LIB): $(SRC) $(HDR)
	@mkdir -p $(BUILD)
	$(CC) $(CFLAGS) -fPIC -Isrc -shared $(SRC) -o $@ $(LDLIBS)

$(TESTBIN): $(SRC) $(HDR) tests/test_store.c
	@mkdir -p $(BUILD)
	$(CC) $(CFLAGS) -Isrc tests/test_store.c $(SRC) -o $@ $(LDLIBS)

test: .deps build ## C behavioural suite + Python unittest suite
	./$(TESTBIN)
	$(PY) -m unittest discover -s tests -p 'test_*.py'

deploy: build ## Install the substrate to PREFIX
	install -d $(DESTDIR)$(PREFIX)/lib $(DESTDIR)$(PREFIX)/include
	install -m 0755 $(LIB) $(DESTDIR)$(PREFIX)/lib/
	install -m 0644 $(HDR) $(DESTDIR)$(PREFIX)/include/

verify: .deps build ## Smoke-test the built artifact through its real ABI
	$(PY) tests/smoke.py

e2e: verify ## Alias for verify

# Deliberate deviation from the skeleton's `all: ... deploy ...`:
# deploy writes outside the project tree (PREFIX, default
# /usr/local) and needs privileges a cicd_runner worker running as
# the host uid:gid does not have. Run it explicitly with a writable
# prefix instead: make deploy PREFIX=$$HOME/.local
all: .deps lint build test verify ## Run the full pipeline

# Deliberately NOT part of `all`. The substrate is CPU-only and must stay
# runnable anywhere -- cicd_runner's workers have no device by design, so
# folding this into `all` would make the ordinary pipeline fail on the
# machine that runs it most. This target is for the one container that has
# /dev/dxg, and it skips rather than fails where the toolchain is absent,
# so a plain checkout is never broken by a missing GPU.
gpu: ## Compile and run the GPU toolchain probe (skipped without hipcc)
	@# One shell for the whole recipe: make runs each recipe LINE in its
	@# own shell, so an `exit 0` in a guard line ends only that line and
	@# the build below still runs -- which is exactly what happened the
	@# first time this was written.
	@#
	@# && rather than ; between the steps: with `;` a failed compile still
	@# reaches the run step, and if a STALE build/gpu_probe were present it
	@# would execute the old binary and print PASS. A failed build must not
	@# be able to report success.
	@if command -v $(HIPCC) >/dev/null 2>&1; then \
		mkdir -p $(BUILD) && \
		echo "$(HIPCC) -O2 -o $(GPUBIN) $(GPUSRC)" && \
		$(HIPCC) -O2 -o $(GPUBIN) $(GPUSRC) && \
		$(GPUBIN); \
	else \
		echo "gpu: $(HIPCC) not found -- skipped"; \
	fi

# .venv is deliberately NOT removed here: rebuilding it is the
# expensive part, and .deps alone being gone is enough to force a
# reinstall into the existing venv on the next run. Remove $(VENV) by
# hand for a genuinely clean slate.
clean: ## Remove build artifacts
	rm -rf $(BUILD) .deps py/locus/__pycache__ tests/__pycache__
