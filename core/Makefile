# LocusAI core -- CPU substrate (C) + pathway layer (Python).
#
# Uses the standard cicd_runner target vocabulary
# (lint/test/build/deploy/verify/e2e/all) so this directory runs
# unchanged through:
#   run_in_directory(relative_path="LocusAI/core", binary="make",
#                    args=["all"])
# See cicd_runner's examples/project-skeleton/ for the convention.
#
# Prefer the image's copy. A worker mounting only core/ sees just
# /etc/cicd-common.mk, but one mounting the repo root (needed by `stream`)
# sees ../cicd-common.mk too, and including both redefines `help`.
ifeq ($(wildcard /etc/cicd-common.mk),)
-include ../cicd-common.mk
else
include /etc/cicd-common.mk
endif

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
BWSRC    = gpu/exp_bandwidth.cpp
BWBIN    = $(BUILD)/exp_bandwidth
BWLOG    = $(BUILD)/gpu-bandwidth.log
FSSRC    = gpu/exp_field_step.cpp
FSBIN    = $(BUILD)/exp_field_step
FSLOG    = $(BUILD)/gpu-field-step.log
GCSRC    = gpu/exp_gpu_cache.cpp
GCBIN    = $(BUILD)/exp_gpu_cache
GCLOG    = $(BUILD)/gpu-cache.log
SPSRC    = gpu/exp_field_sparse.cpp
SPBIN    = $(BUILD)/exp_field_sparse
SPLOG    = $(BUILD)/gpu-field-sparse.log
QTSRC    = gpu/exp_field_quiet.cpp
QTBIN    = $(BUILD)/exp_field_quiet
QTLOG    = $(BUILD)/gpu-field-quiet.log
LSSRC    = gpu/learn_stream.cpp
LSBIN    = $(BUILD)/learn_stream
LDSRC    = gpu/learn_device.cpp
LDBIN    = $(BUILD)/learn_device
KNOW    ?= $(BUILD)/know/locus
ENGINE  ?= device
CORPUS  ?= ..
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
.PHONY: lint test build deploy verify e2e all clean gpu caps persist caps-mock hip-header-check rocm-headers rocm-image excursion bandwidth field-step field-sparse field-quiet gpu-cache hierarchy capacity operating-point stack singlepass precision stream stream-learn stream-learn-gpu bg

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
	$(PY) -m pycodestyle --max-line-length=79 py/locus tests gpu/tests tools

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

# Whether the per-area inhibition sums can be carried from the previous
# step's write instead of re-reduced, removing a launch per tick on a GPU
# with no grid-wide barrier. Includes a lagged negative control; exit 2
# means the control went undetected. No GPU needed.
singlepass: .deps ## Check carried inhibition sums against the reference update
	@$(VENV)/bin/python3 tests/exp_singlepass.py

# Whether FP8 or block-scaled FP4 kernel weights change which attractor
# a cue reaches. Exit 2 means a settle ended on its budget. No GPU needed.
precision: .deps ## Attractor survival under FP8 and FP4 kernel weights
	@$(VENV)/bin/python3 tests/exp_precision.py

# Real input: the repository's own text files through the byte + hashed
# n-gram encoder, with a reference bits-per-byte score to beat. Reads the
# repo root (..), so a cicd_runner worker must mount the repo, not core/:
#   run_in_directory(relative_path="LocusAI", binary="make",
#                    args=["-C", "core", "stream"])
stream: .deps ## Encode the repo's text and score a reference next-byte predictor
	@$(VENV)/bin/python3 tests/exp_stream.py ..

# Three-factor learning on the same stream: directional tags, surprise as
# modulator, per-source scaling at each file end; scored against the
# count reference on held-out files, with an ablation of the learner and
# a check variant that must reproduce the normalised-count control.
# Repo root must be mounted, as for `stream`. Exit 2: a readout
# parameter fitted at its grid edge.
# Takes minutes on the LocusAI repo, beyond a tool call's time limit: run
# it through `bg` (below).
stream-learn: .deps ## Learn next-byte structure from the repo's text and score it (minutes)
	@$(VENV)/bin/python3 tests/exp_learn_stream.py ..

# The per-input learners (branch, CLS) on the GPU: each unit's events are
# replayed in one block, in the Python learners' fp64 operation order; on
# the LocusAI corpus the scores equal the Python run's printed figures.
# CORPUS defaults to the repo root; GPUARGS passes --salts / --only (see
# tests/exp_learn_gpu.py). ENGINE=device (default) runs encoding, event
# sorting, learning, the control and the lam grid on the GPU
# (gpu/learn_device.cpp); ENGINE=host builds events in Python and uses
# gpu/learn_stream.cpp, kept as the reference for that path. Runs in the
# project container (hipcc);
# start it through `bg`. Sequential learners stay in `stream-learn`.
stream-learn-gpu: hip-header-check ## Branch and CLS learners on the GPU, scored as stream-learn
	@if command -v $(HIPCC) >/dev/null 2>&1; then \
		mkdir -p $(BUILD) && \
		$(GPU_ENV) $(HIPCC) $(HIPCXXFLAGS) $(HIPINC) -o $(LSBIN) $(LSSRC) && \
		$(GPU_ENV) $(HIPCC) $(HIPCXXFLAGS) $(HIPINC) -o $(LDBIN) $(LDSRC) && \
		$(GPU_ENV) $(PYTHON3) -u tests/exp_learn_gpu.py $(CORPUS) \
			$(if $(filter host,$(ENGINE)),$(LSBIN),$(LDBIN)) \
			--engine $(ENGINE) $(GPUARGS); \
	else \
		echo "stream-learn-gpu: $(HIPCC) not found -- skipped"; \
	fi

# A knowledge store: the learner (default CLS, metaplastic neocortex)
# trained on CORPUS's train split on the GPU and kept at KNOW (+ .json).
# KNOWARGS passes --salt / --learner (see tools/know.py).
know: hip-header-check ## Train and keep a knowledge store at KNOW
	@if command -v $(HIPCC) >/dev/null 2>&1; then \
		mkdir -p $(BUILD) && \
		$(GPU_ENV) $(HIPCC) $(HIPCXXFLAGS) $(HIPINC) -o $(LDBIN) $(LDSRC) && \
		$(GPU_ENV) $(PYTHON3) -u tools/know.py $(CORPUS) $(LDBIN) $(KNOW) \
			$(KNOWARGS); \
	else \
		echo "know: $(HIPCC) not found -- skipped"; \
	fi

# Read a prompt against KNOW: PROMPTARGS='--file F' (or stdin),
# '--no-learn', or '--score' to rescore the store's test split.
prompt: ## Read a prompt against the knowledge store at KNOW
	$(PYTHON3) tools/prompt.py $(KNOW) $(PROMPTARGS)

# Log an outcome (modulator) at the store's current tick: VALUE=+1 / -1.
outcome: ## Log an outcome VALUE for the transient tier of KNOW
	$(if $(VALUE),,$(error outcome: VALUE is required))
	$(PYTHON3) tools/outcome.py $(KNOW) $(VALUE)

# Offline consolidation: capture rewarded tags into KNOW, lapse the rest.
# SLEEPARGS='--lifetime TICKS' bounds how far back an outcome credits.
sleep: ## Consolidate the transient tier of KNOW into its permanent tier
	$(PYTHON3) tools/sleep.py $(KNOW) $(SLEEPARGS)

# Which outcome signal should decide what reading keeps: seven methods,
# one process each, from the generation-0 store MODKNOW over CORPUS's
# validation files; stores under build/mod/. Start through `bg`.
MODKNOW ?= $(BUILD)/know/locus.gen0
modulators: ## Compare outcome signals for reading (none/all/novelty/.../mixture)
	$(PYTHON3) -u tests/exp_modulators.py $(MODKNOW) $(CORPUS) $(BUILD)/mod \
		$(MODARGS)

# A model repository's code, config, tokenizer and tensor index, without
# its tensors: REPO=org/name, HFARGS='--dry-run' / '--max-mb N'.
hf-code: ## Fetch a Hugging Face repo without tensor files into build/hf
	$(if $(REPO),,$(error hf-code: REPO is required))
	$(PYTHON3) tools/hf_code.py $(REPO) $(BUILD)/hf $(HFARGS)

# The transformers source for every model_type named by a fetched
# config.json (sparse, blob-less); HFARCHARGS='--ref <tag>' pins a release.
hf-arch: ## Fetch transformers modeling code for the architectures in build/hf
	$(PYTHON3) tools/hf_arch.py $(BUILD)/hf $(HFARCHARGS)

# The code that serves each local Ollama model at the version running:
# Ollama's source at that tag (llama.cpp pin, compat patches, parsers,
# renderers), llama.cpp at the pinned tag, and each model's anatomy from
# /api/show. OLLAMAARGS='--api URL' if the server is elsewhere.
ollama-code: ## Fetch the serving code and anatomy of the local Ollama models
	$(PYTHON3) tools/ollama_code.py $(BUILD)/ollama $(OLLAMAARGS)

# Reading material: a bounded slice of a Hugging Face dataset into
# build/data. DATASET=org/name, DATAARGS='--file PATH --max-mb N'; with no
# --file it lists the repository and stops.
hf-data: ## Fetch a bounded slice of a Hugging Face dataset into build/data
	$(if $(DATASET),,$(error hf-data: DATASET is required))
	$(PYTHON3) tools/hf_data.py $(DATASET) $(BUILD)/data $(DATAARGS)

# The field experiments are single-core by nature (Field.step is O(n^2)
# in Python) but independent of each other, so they run side by side,
# one process each, logs under build/exps. EXPS=name1 name2 picks a
# subset. Serial cost measured 2026-09-22: 111 s on one core of 24.
experiments: ## Run the field experiments concurrently into build/exps
	$(PYTHON3) tools/run_exps.py $(BUILD)/exps $(EXPS)

# Long jobs exceed a tool call, and polling them through the cicd_runner
# coordinator blocks its single event loop. Start them detached in the
# project container with ONE call, then read build/$(JOB).log through the
# Filesystem connector until its last line is exit=N:
#   docker exec -d --user 1000:1000 -w /workspace/core server-local-bash-1 \
#     make bg JOB=stream-branch \
#     CMD='python3 -u tests/exp_learn_stream.py .. branch'
# The container has no .venv of its own, so CMD uses python3 directly;
# the experiments need only the standard library. CMD reaches the shell
# only through sh -c "$$BG_CMD": the argument check is make-level and the
# header echoes the variable, so quotes in CMD are never re-parsed.
# Every job is monitored: tools/perfmon.py samples CPU and memory into
# build/JOB.perf.csv and its summary lands in the log before exit=N.
bg: export BG_CMD = $(value CMD)
bg: ## Run CMD with all output in build/JOB.log, ending in an exit=N line
	$(if $(and $(JOB),$(CMD)),,$(error bg: JOB and CMD are required))
	@mkdir -p $(BUILD)
	@{ echo "[start] $$(date -u +%FT%TZ) $$BG_CMD"; \
	   $(PYTHON3) tools/perfmon.py record $(BUILD)/$(JOB).perf.csv & pm=$$!; \
	   sh -c "$$BG_CMD"; rc=$$?; \
	   kill $$pm; wait $$pm; \
	   echo "[end] $$(date -u +%FT%TZ)"; \
	   $(PYTHON3) tools/perfmon.py summary $(BUILD)/$(JOB).perf.csv; \
	   echo "exit=$$rc"; } \
		> $(BUILD)/$(JOB).log 2>&1

# Host side of the memory hierarchy: CPU cache sweep, DRAM write and
# O_DIRECT disk I/O on this repo's filesystem (a 256 MiB temp file in
# $(BUILD), removed after). Single-threaded. Under WSL2 the Windows host
# cache may still serve disk reads. Exit 2: a figure ended on the cap.
hierarchy: ## Host cache, DRAM and disk bandwidth (no GPU needed)
	@mkdir -p $(BUILD) && \
	$(CC) -std=c11 -O2 -Wall -Wextra -Werror -o $(BUILD)/exp_hierarchy tests/exp_hierarchy.c -lm && \
	{ $(BUILD)/exp_hierarchy $(BUILD) > $(BUILD)/hierarchy.log 2>&1; \
	  rc=$$?; \
	  cat $(BUILD)/hierarchy.log; \
	  echo "[artifact] $(BUILD)/hierarchy.log  $$(stat -c%s $(BUILD)/hierarchy.log) bytes"; \
	  exit $$rc; }

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

# Times device-memory and host<->device bandwidth, because memoryClockRate
# is unreliable under GPU-PV. Feeds the ingestion IO check in
# doc/core/HIERARCHY.md. Same one-shell / && / skip-not-fail contract as
# `gpu`. Exit 2 means a figure ended on the cap rather than stabilising.
bandwidth: hip-header-check ## Measure device and host<->device bandwidth (skipped without hipcc)
	@if command -v $(HIPCC) >/dev/null 2>&1; then \
		mkdir -p $(BUILD) && \
		echo "$(HIPCC) $(HIPCXXFLAGS) $(HIPINC) -o $(BWBIN) $(BWSRC)" && \
		$(GPU_ENV) $(HIPCC) $(HIPCXXFLAGS) $(HIPINC) -o $(BWBIN) $(BWSRC) && \
		{ $(GPU_ENV) $(BWBIN) > $(BWLOG) 2>&1; \
		  rc=$$?; \
		  cat $(BWLOG); \
		  echo "[artifact] $(BWLOG)  $$(stat -c%s $(BWLOG)) bytes"; \
		  exit $$rc; }; \
	else \
		echo "bandwidth: $(HIPCC) not found -- skipped"; \
	fi

# Field.step on the device: carried inhibition partials must be bit-identical
# to a separate reduce launch, a lagged control must differ, fp32 must match
# a double-precision CPU replica's support. Then per-step time for fp32
# two-launch, fp32 carried and fp8 carried. Exit 1: correctness failed;
# exit 2: a timing ended on the cap.
field-step: hip-header-check ## Field.step on the GPU: correctness and per-step time (skipped without hipcc)
	@if command -v $(HIPCC) >/dev/null 2>&1; then \
		mkdir -p $(BUILD) && \
		echo "$(HIPCC) $(HIPCXXFLAGS) $(HIPINC) -o $(FSBIN) $(FSSRC)" && \
		$(GPU_ENV) $(HIPCC) $(HIPCXXFLAGS) $(HIPINC) -o $(FSBIN) $(FSSRC) && \
		{ $(GPU_ENV) $(FSBIN) > $(FSLOG) 2>&1; \
		  rc=$$?; \
		  cat $(FSLOG); \
		  echo "[artifact] $(FSLOG)  $$(stat -c%s $(FSLOG)) bytes"; \
		  exit $$rc; }; \
	else \
		echo "field-step: $(HIPCC) not found -- skipped"; \
	fi

# GPU read sweep from 256 KiB to 1 GiB (L2, Infinity Cache and VRAM
# plateaus) plus LDS read bandwidth. Same contract as `bandwidth`.
gpu-cache: hip-header-check ## GPU cache/VRAM read sweep and LDS bandwidth (skipped without hipcc)
	@if command -v $(HIPCC) >/dev/null 2>&1; then \
		mkdir -p $(BUILD) && \
		echo "$(HIPCC) $(HIPCXXFLAGS) $(HIPINC) -o $(GCBIN) $(GCSRC)" && \
		$(GPU_ENV) $(HIPCC) $(HIPCXXFLAGS) $(HIPINC) -o $(GCBIN) $(GCSRC) && \
		{ $(GPU_ENV) $(GCBIN) > $(GCLOG) 2>&1; \
		  rc=$$?; \
		  cat $(GCLOG); \
		  echo "[artifact] $(GCLOG)  $$(stat -c%s $(GCLOG)) bytes"; \
		  exit $$rc; }; \
	else \
		echo "gpu-cache: $(HIPCC) not found -- skipped"; \
	fi

# Field.step with a sparse (gather-by-target) kernel against the dense one:
# bit-identical trajectories, a descending-order control that must differ,
# and per-step time from fixed states with the active-row count printed.
# Exit 1: correctness failed; exit 2: a timing ended on the cap.
field-sparse: hip-header-check ## Sparse vs dense Field.step on the GPU (skipped without hipcc)
	@if command -v $(HIPCC) >/dev/null 2>&1; then \
		mkdir -p $(BUILD) && \
		echo "$(HIPCC) $(HIPCXXFLAGS) $(HIPINC) -o $(SPBIN) $(SPSRC)" && \
		$(GPU_ENV) $(HIPCC) $(HIPCXXFLAGS) $(HIPINC) -o $(SPBIN) $(SPSRC) && \
		{ $(GPU_ENV) $(SPBIN) > $(SPLOG) 2>&1; \
		  rc=$$?; \
		  cat $(SPLOG); \
		  echo "[artifact] $(SPLOG)  $$(stat -c%s $(SPLOG)) bytes"; \
		  exit $$rc; }; \
	else \
		echo "field-sparse: $(HIPCC) not found -- skipped"; \
	fi

# Exact quiet-term skipping (below half an ulp of the running drive), a
# source-row two-stage layout, and batched per-step time. Both exactness
# claims have controls that must break them. Exit 1: a check failed or a
# control went undetected; exit 2: a timing ended on the cap.
field-quiet: hip-header-check ## Quiet rows, source-row layout, batched step time (skipped without hipcc)
	@if command -v $(HIPCC) >/dev/null 2>&1; then \
		mkdir -p $(BUILD) && \
		echo "$(HIPCC) $(HIPCXXFLAGS) $(HIPINC) -o $(QTBIN) $(QTSRC)" && \
		$(GPU_ENV) $(HIPCC) $(HIPCXXFLAGS) $(HIPINC) -o $(QTBIN) $(QTSRC) && \
		{ $(GPU_ENV) $(QTBIN) > $(QTLOG) 2>&1; \
		  rc=$$?; \
		  cat $(QTLOG); \
		  echo "[artifact] $(QTLOG)  $$(stat -c%s $(QTLOG)) bytes"; \
		  exit $$rc; }; \
	else \
		echo "field-quiet: $(HIPCC) not found -- skipped"; \
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
