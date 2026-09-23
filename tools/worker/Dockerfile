# LocusAI's own cicd_runner worker image.
#
# WHY THIS EXISTS RATHER THAN A LINE IN THE SHARED WORKER. The shared
# `cicd-worker` image is bare by design: it carries what EVERY project
# needs, and a toolchain one project wants belongs in that project's own
# image (see the `docker-cicd-runner` skill, "Where a dependency
# belongs" -- pin a per-project image with a `.cicd-image` file). C++ is
# LocusAI's requirement alone, so it is pinned here and the shared image
# stays unchanged for every other project that runs through the runner.
#
# WHAT IT ADDS AND WHY. `make caps-mock` compiles core/gpu/caps.cpp and
# persist.cpp against a MOCKED HIP runtime -- 19 scenarios, no GPU and
# no hipcc involved -- and it is the only stage that can check the
# probes' reporting logic where the real `caps` target can merely skip.
# Without a C++ compiler it printed "no C++ compiler -- skipped" inside
# every green pipeline run, which is a stage nobody noticed was absent
# being reported as part of a passing build.
#
# It does NOT add hipcc or ROCm. Kernel sources need both plus the
# device, which no worker has by design; `make gpu-syntax` covers those
# and runs in LocusAI's own GPU container instead.
#
# Built FROM the shared worker so the entrypoint, the uid:gid drop and
# the allowlisted binaries stay identical -- this image differs from the
# default by exactly one package. Rebuild after the shared worker is
# rebuilt, or it keeps inheriting the older base:
#
#   docker build -t locusai-worker:latest -f tools/worker/Dockerfile \
#       tools/worker
FROM cicd-worker:latest

USER root
RUN apt-get update && apt-get install -y --no-install-recommends \
        g++ \
    && rm -rf /var/lib/apt/lists/*
