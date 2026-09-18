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
# ROCM_PATH is RESOLVED, not assumed, and this is load-bearing on two counts.
# (1) /opt/rocm's bin/lib/include are update-alternatives symlinks into
# /etc/alternatives, which the GPU container does not mount, so they dangle
# and a bare `hipcc` is not found even though the tree is mounted -- the
# resolved /opt/rocm/core-<ver> path works. (2) Hardcoding a version would
# break on the next ROCm move and on the native boot, where /opt/rocm may be
# a normal tree. So: take the first candidate that actually has hipcc in it.
ROCM_PATH ?= $(shell for d in /opt/rocm /opt/rocm/core-*; do [ -x "$$d/bin/hipcc" ] && echo "$$d" && break; done)
HIPCC   ?= $(if $(ROCM_PATH),$(ROCM_PATH)/bin/hipcc,hipcc)

# -std=c++17 is EXPLICIT: caps.cpp uses the struct-field detection idiom, and
# hipcc's default standard has moved between ROCm releases.
HIPCXXFLAGS ?= -std=c++17 -O2

# The HIP SDK headers. Added ONLY when present, so a host whose ROCm ships
# its own hip/ needs no flag and a checkout without the extracted tree still
# builds there. See doc/core/HARDWARE.md for why this tree exists at all.
ROCM_INCLUDE ?= gpu/.rocm-include
HIPINC = $(if $(wildcard $(ROCM_INCLUDE)/hip/hip_runtime.h),-isystem $(ROCM_INCLUDE),)

# WSL2 reaches the GPU through /dev/dxg and needs the host's WSL shim libs;
# a native boot uses /dev/kfd + /dev/dri and has no such directory. Appended
# only if it exists, so one line covers both boots.
WSLLIB = $(if $(wildcard /usr/lib/wsl/lib),:/usr/lib/wsl/lib,)
GPU_ENV = ROCM_PATH="$(ROCM_PATH)" HIP_PATH="$(ROCM_PATH)" \
          LD_LIBRARY_PATH="$(ROCM_PATH)/lib$(WSLLIB)"

GPUSRC   = gpu/probe.cpp
GPUBIN   = $(BUILD)/gpu_probe
CAPSSRC  = gpu/caps.cpp
CAPSBIN  = $(BUILD)/gpu_caps
CAPSLOG  = $(BUILD)/gpu-caps.log
PERSRC   = gpu/persist.cpp
PERBIN   = $(BUILD)/gpu_persist
PERLOG   = $(BUILD)/gpu-persist.log
MOCKDRV  = gpu/tests/gpu_mock_driver.py
EXPSRC   = gpu/exp_excursion.cpp
EXPBIN   = $(BUILD)/exp_excursion
EXPLOG   = $(BUILD)/gpu-excursion.log
# Overridable so the result can be SWEPT rather than taken on trust:
#   make excursion EXPN=8192 EXPNA=2048
EXPN    ?= 4096
EXPNA   ?= 1024
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
.PHONY: lint test build deploy verify e2e all clean gpu caps persist caps-mock hip-header-check rocm-headers rocm-image excursion capacity operating-point stack

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
	$(PY) -m pycodestyle --max-line-length=79 py/locus tests gpu/tests

# No rule for $(BUILD) itself: it would collide with the phony `build`
# target and silently drop the order-only prerequisite.
build: $(LIB) $(TESTBIN) ## Compile the shared library and C test binary

$(LIB): $(SRC) $(HDR)
	@mkdir -p $(BUILD)
	$(CC) $(CFLAGS) -fPIC -Isrc -shared $(SRC) -o $@ $(LDLIBS)

$(TESTBIN): $(SRC) $(HDR) tests/test_store.c
	@mkdir -p $(BUILD)
	$(CC) $(CFLAGS) -Isrc tests/test_store.c $(SRC) -o $@ $(LDLIBS)

test: .deps build caps-mock ## C behavioural suite + Python unittest suite + GPU-probe mock suite
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

# Measures how many distinguishable states the field holds, which is
# the blocker for anything task-shaped: a coding task must tell apart
# the files it touches, the errors it sees and the actions it can take,
# and that is hundreds rather than tens.
#
# MEASURED 2026-09-18, and the first reading was wrong for an
# instructive reason. Capacity looked capped at about 11 states, which
# would have meant the architecture was finished at toy size. It was a
# bad operating point: at beta=0.1 the field has 2 attractors with one
# basin holding 98%, while at beta>=1.0 it reaches 64 of 64 at n=64.
# Capacity SCALES with the state space (12, 31, 60, 96 at n=32, 48, 64,
# 96) and the dominant basin falls from 47% to 1%.
#
# TWO CORRECTIONS WORTH KEEPING. Modularity does NOT add capacity --
# predicted it would, measured modular x4 at 4 attractors against
# random's 60, and only blocks of four match random rather than beating
# it; block size sets attractor size and capacity follows from how many
# blocks fit. And SUPPORT stability is the criterion, not value
# convergence: the support settles at ~45 steps while the value fixed
# point takes ~198, so a field in a limit cycle has a perfectly stable
# active set and never reaches a fixed point. An earlier run reported
# every trial unsettled because it checked the stricter criterion.
#
# Needs no GPU and no device, so it runs anywhere python3 does.
#
# RETRACTED AND RE-MEASURED 2026-09-18 with the SATURATING survey,
# which drives until the repertoire stops growing so the sample count
# is an OUTPUT. The attractor counts above came from designs that each
# fixed an arbitrary number, and each confounded the result: a stride
# tied the sample count to n, a fixed stream length drew more unique
# cues at larger n through fewer collisions, and fixing unique cues
# would have pinned the quantity under study. Same error every time --
# a constant standing in for a stopping condition.
# HONEST FIGURES: repertoire is about n at the working beta (15 of 32,
# 52 of 64, 126 of 128, 254 of 256) and cues to saturate run about 5.7x
# the repertoire (52, 299, 702, 1453). Capacity does scale with n.
capacity: .deps ## Measure how many distinguishable states the field holds
	@$(VENV)/bin/python3 tests/exp_capacity.py

# Inhibition serves three requirements that were each measured in their
# OWN configuration, and so appeared to conflict: sparsity wants the
# active set near the biological 1-2%, capacity wants enough
# distinguishable states, and compression wants few enough attractors
# that a level above is possible. Three numbers from three separate runs
# cannot be traded off, because nothing guarantees the configurations
# were comparable. This measures all three from ONE configuration per
# beta, which is what makes them comparable at all.
#
# MEASURED 2026-09-18 at n=64: a joint operating point EXISTS, beta 0.7
# to 3.0, giving PR/n 0.024 with 16 attractors and 4.00x compression at
# beta 1.0. Two earlier readings were artefacts of cross-configuration
# comparison -- sparsity's band was put at 0.1-0.5 when beta=0.1 in fact
# leaves half the field active, and capacity was thought to want high
# inhibition when that result came from a RANDOM kernel giving one
# attractor per state. Anatomy decides whether the capacity-versus-
# compression trade-off exists at all.
#
# THE JOINT OPERATING POINT ABOVE IS RETRACTED, 2026-09-18. Those 16
# attractors and 4.00x came from an ENUMERATING survey whose count was
# capped by its sampling stride; the saturating survey gives 62
# attractors and 1.03x at beta 1.0.
# NO JOINT OPERATING POINT EXISTS. Where compression is high (beta
# <= 0.3) the repertoire is 1 or 2, so its apparent 64x compresses
# nothing into nothing. Where sparsity is biological (beta >= 0.7) the
# repertoire is about n and compression is 1.03x, so no level above is
# possible. Beta 0.5 is the only middle ground and fails sparsity at
# 10.9% active. That FALSIFIES the one-control design rather than being
# a tuning failure.
# Per-area competition was the candidate second control and was
# measured NOT to provide it: splitting into A areas divides each
# unit's inhibitory denominator by A, scaling the existing dial down
# rather than adding a new one -- sparsity degraded from 2.3% to 16.7%
# active while the repertoire collapsed from 60 to 3.
operating-point: .deps ## Reconcile sparsity, capacity and compression against inhibition
	@$(VENV)/bin/python3 tests/exp_operating_point.py

# Drives the hierarchy upward until the top level fits a window, so
# DEPTH AND COMPRESSION ARE OUTPUTS rather than parameters. Each level
# is driven by the settled active set of the one below it, because a
# level cannot chunk before it has learned and what it learns from is
# what arrives from beneath.
#
# MEASURED 2026-09-18, and it corrected a conclusion. At n=64 the stack
# reached one level at 4.6x and refused, from which I concluded levels
# do not stack -- wrong, and drawn from a field too small to have state
# at level 1. At n=256 it reaches two levels and 85.3x, terminating
# because the top FITS rather than because a level failed. Compression
# also varies per level (8.00x then 10.67x), which is the first actual
# demonstration that it is emergent rather than constant.
#
# THE COMPRESSION FIGURES ABOVE ARE RETRACTED, 2026-09-18. The 85.3x,
# 8.00x and 10.67x all came from a survey sampling every 8th state, and
# you cannot discover more attractors than you have injection points --
# the chunk count was capped at n/stride, so the compression WAS the
# stride. At stride=1 per-level compression is 1.16x to 2.88x; with the
# saturating survey the repertoire is about n at the working beta, which
# puts real compression near 1.0x.
# WHAT STANDS: the stack terminates on its own refusal rather than a
# chosen cutoff, depth and compression are outputs, and each level is
# driven by the settled active set of the one below. WHAT DOES NOT: any
# compression number. Attractor-based chunking cannot compress a field
# whose repertoire is already about n -- capacity and compression were
# tied to the same object, which is why they contended.
stack: .deps ## Drive the hierarchy upward and report depth and compression
	@$(VENV)/bin/python3 tests/exp_stack.py

# --- long-running HOST operations, logged ---------------------------------
# These move megabytes to gigabytes and take minutes. They are deliberately
# meant to be run BY HAND on the host, not through an agent tool call: a
# large transfer inside a tool call gives no progress, no resumption, and a
# timeout leaves partial state behind.
#
# Each tees so progress is visible live, captures the REAL exit status
# rather than tee's, and enumerates its own artifacts with sizes at the
# end -- a summary never says what it left out, so the log states every
# path that landed.
LOGDIR ?= logs
STAMP   = $(shell date -u +%Y%m%dT%H%M%SZ)

rocm-headers: ## HOST: fetch+extract HIP SDK headers (~0.5 MiB, logged)
	@mkdir -p $(LOGDIR)
	@log="$(LOGDIR)/$(STAMP)-rocm-headers.log"; \
	echo "logging to $$log"; \
	{ \
	  echo "== rocm-headers $$(date -u +%FT%TZ) =="; \
	  echo "-- host ROCm release, which the headers must match --"; \
	  cat "$(ROCM_PATH)/.info/version" 2>&1 || echo "no host ROCm"; \
	  $(PYTHON3) gpu/fetch_headers.py; \
	  echo "fetch_exit=$$?"; \
	} 2>&1 | tee "$$log"; \
	echo "[artifact] $$log  $$(stat -c%s "$$log") bytes"; \
	$(MAKE) --no-print-directory hip-header-check

# Only needed on a boot with NO host ROCm -- the Mageia case. On WSL2 the
# host tree is bind-mounted and this is unnecessary. 7.14 and 10.0 tags are
# "-full" only at 7.40-7.70 GiB COMPRESSED, so budget ~20 GiB extracted;
# rocm-detect.sh refuses under 20 GiB free rather than filling the disk.
rocm-image: ## HOST: pull the version-matched ROCm image (GIGABYTES, logged)
	@mkdir -p $(LOGDIR)
	@log="$(LOGDIR)/$(STAMP)-rocm-image.log"; \
	echo "logging to $$log"; \
	{ \
	  echo "== rocm-image $$(date -u +%FT%TZ) =="; \
	  df -h . | tail -1; \
	  ../tools/server/rocm-detect.sh --pull; \
	  echo "pull_exit=$$?"; \
	  docker images --format \
	    '[artifact] image {{.Repository}}:{{.Tag}}  {{.Size}}' \
	    2>/dev/null | grep -i rocm || echo "no rocm image present"; \
	} 2>&1 | tee "$$log"; \
	echo "[artifact] $$log  $$(stat -c%s "$$log") bytes"

# Guards the ONE failure mode the extracted-header workaround introduces.
# The tree in $(ROCM_INCLUDE) is version-pinned to whatever ROCm the host ran
# when it was unpacked; the host can move underneath it, and skewed headers
# COMPILE FINE while reporting wrong struct offsets -- silent, and exactly
# the class caps.cpp's field detection exists to survive rather than detect.
# Both sides publish the same triple, so compare them and fail loudly.
# Checked ONLY when the extracted tree is actually in use: a host with a
# complete SDK has nothing to skew.
hip-header-check: ## Assert extracted HIP headers match the host ROCm version
	@if [ -z "$(HIPINC)" ]; then \
		echo "hip-header-check: not using extracted headers -- skipped"; \
	elif [ ! -f "$(ROCM_PATH)/share/hip/version" ]; then \
		echo "hip-header-check: no host version file -- UNCHECKED"; \
	else \
		hdr=$$(sed -n 's/^#define HIP_VERSION_\(MAJOR\|MINOR\|PATCH\) *\([0-9]*\)$$/\2/p' $(ROCM_INCLUDE)/hip/hip_version.h | tr '\n' '.'); \
		host=$$(sed -n 's/^HIP_VERSION_\(MAJOR\|MINOR\|PATCH\)=\([0-9]*\)$$/\2/p' "$(ROCM_PATH)/share/hip/version" | tr '\n' '.'); \
		if [ -z "$$hdr" ] || [ -z "$$host" ]; then \
			echo "hip-header-check: could not read a version from both sides -- UNCHECKED (hdr='$$hdr' host='$$host')"; \
		elif [ "$$hdr" = "$$host" ]; then \
			echo "hip-header-check: headers $$hdr match host runtime -- ok"; \
		else \
			echo "hip-header-check: FAIL -- extracted headers $$hdr vs host runtime $$host"; \
			echo "  The host's ROCm moved. Regenerate $(ROCM_INCLUDE) from the"; \
			echo "  host's CURRENT ROCm rather than compiling against skew;"; \
			echo "  see doc/core/HARDWARE.md, 'Reaching the device'."; \
			exit 1; \
		fi; \
	fi

# Deliberately NOT part of `all`. The substrate is CPU-only and must stay
# runnable anywhere -- cicd_runner's workers have no device by design, so
# folding this into `all` would make the ordinary pipeline fail on the
# machine that runs it most. This target is for the one container that has
# /dev/dxg, and it skips rather than fails where the toolchain is absent,
# so a plain checkout is never broken by a missing GPU.
gpu: hip-header-check ## Compile and run the GPU toolchain probe (skipped without hipcc)
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
		echo "$(HIPCC) $(HIPCXXFLAGS) $(HIPINC) -o $(GPUBIN) $(GPUSRC)" && \
		$(GPU_ENV) $(HIPCC) $(HIPCXXFLAGS) $(HIPINC) -o $(GPUBIN) $(GPUSRC) && \
		$(GPU_ENV) $(GPUBIN); \
	else \
		echo "gpu: $(HIPCC) not found -- skipped"; \
	fi

# Same one-shell / && / skip-not-fail contract as `gpu` above, for the same
# reasons -- see that comment before changing either.
caps: hip-header-check ## Probe device capabilities and the memory hierarchy (skipped without hipcc)
	@if command -v $(HIPCC) >/dev/null 2>&1; then \
		mkdir -p $(BUILD) && \
		echo "$(HIPCC) $(HIPCXXFLAGS) $(HIPINC) -o $(CAPSBIN) $(CAPSSRC)" && \
		$(GPU_ENV) $(HIPCC) $(HIPCXXFLAGS) $(HIPINC) -o $(CAPSBIN) $(CAPSSRC) && \
		$(GPU_ENV) $(CAPSBIN) $(CAPSLOG); \
	else \
		echo "caps: $(HIPCC) not found -- skipped"; \
	fi

# Gated: persist.cpp runs a deliberately long kernel and SELF-SKIPS unless
# LOCUS_GPU_STRESS=1, so invoking this target is safe by default. Read the
# warning at the top of gpu/persist.cpp before setting that variable.
persist: hip-header-check ## Probe persistent-kernel viability (gated by LOCUS_GPU_STRESS=1)
	@if command -v $(HIPCC) >/dev/null 2>&1; then \
		mkdir -p $(BUILD) && \
		echo "$(HIPCC) $(HIPCXXFLAGS) $(HIPINC) -o $(PERBIN) $(PERSRC)" && \
		$(GPU_ENV) $(HIPCC) $(HIPCXXFLAGS) $(HIPINC) -o $(PERBIN) $(PERSRC) && \
		$(GPU_ENV) $(PERBIN) $(PERLOG); \
	else \
		echo "persist: $(HIPCC) not found -- skipped"; \
	fi

# A STRUCTURE measurement, not a capability probe. It computes rho(P_BB)
# and the EXACT excursion-return distribution for two partitions of the
# same heavy-tailed kernel. That distribution's tail is what sets the
# truncation depth in locus.trace_chain.truncated_trace, so this is how
# the depth gets chosen FROM DATA instead of guessed.
#
# MEASURED 2026-09-18 at EXPN=4096 EXPNA=1024, and the result retired an
# estimator: BOTH partitions gave rho(P_BB) = 0.93336, identical to five
# decimals, while needing depth 6 versus 13 for 99% of returned mass
# (mean excursion 1.96 versus 3.94 steps). So RHO CARRIES NO INFORMATION
# ABOUT REQUIRED DEPTH -- what predicts it is whether the hubs sit in the
# kept set or the complement. Keeping hubs in A shortens excursions,
# because a wandering trajectory reaches a hub quickly and hubs return.
# Both distributions were monotone geometric with no second mode, which
# retired a predicted bimodality and means one depth per level suffices.
# Note only 32% (hubs in A) and 60% (hubs in B) of LEAKED mass returned
# within 16 steps, so "99% of returned mass" is not 99% of the leak: the
# residue sets a truncation error floor no depth removes.
excursion: hip-header-check ## Measure excursion depth on the device (skipped without hipcc)
	@if command -v $(HIPCC) >/dev/null 2>&1; then \
		mkdir -p $(BUILD) && \
		echo "$(HIPCC) $(HIPCXXFLAGS) $(HIPINC) -o $(EXPBIN) $(EXPSRC)" && \
		$(GPU_ENV) $(HIPCC) $(HIPCXXFLAGS) $(HIPINC) -o $(EXPBIN) $(EXPSRC) && \
		{ $(GPU_ENV) $(EXPBIN) $(EXPN) $(EXPNA) > $(EXPLOG) 2>&1; \
		  rc=$$?; \
		  cat $(EXPLOG); \
		  echo "[artifact] $(EXPLOG)  $$(stat -c%s $(EXPLOG)) bytes"; \
		  exit $$rc; }; \
	else \
		echo "excursion: $(HIPCC) not found -- skipped"; \
	fi

# Needs NO GPU, no hipcc and no .venv -- that is the point. It is the only
# stage that can guard the probes' reporting logic where `caps` can merely
# skip. Deliberately uses $(PYTHON3) rather than the venv interpreter: the
# driver imports only the stdlib, so depending on .deps would make it
# unrunnable exactly where it is most needed. It SKIPS rather than fails
# without a C++ compiler, because cicd_runner's worker has gcc and no g++ --
# failing there would break the pipeline on the machine that runs it most.
caps-mock: ## Verify GPU-probe branching against a mocked HIP runtime (no GPU needed)
	@if command -v $(CXX) >/dev/null 2>&1; then \
		CXX=$(CXX) $(PYTHON3) $(MOCKDRV); \
	elif command -v c++ >/dev/null 2>&1; then \
		CXX=c++ $(PYTHON3) $(MOCKDRV); \
	else \
		echo "caps-mock: no C++ compiler ($(CXX), c++) -- skipped"; \
	fi

# .venv is deliberately NOT removed here: rebuilding it is the
# expensive part, and .deps alone being gone is enough to force a
# reinstall into the existing venv on the next run. Remove $(VENV) by
# hand for a genuinely clean slate.
clean: ## Remove build artifacts
	rm -rf $(BUILD) .deps py/locus/__pycache__ tests/__pycache__
