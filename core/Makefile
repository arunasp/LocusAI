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
.PHONY: lint test build deploy verify e2e all clean gpu caps persist caps-mock hip-header-check

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
