"""Run independent jobs on every core, results in the order given.

The experiments here loop over configurations that do not depend on one
another -- sizes, seeds, inhibition values, learners. Python threads
share one core under the GIL, so processes are the only way to use the
machine (the standing constraint in doc/core/ROADMAP.md: GPU first, then
all CPU cores). Results come back in the order of `specs`, so printing
and comparisons stay deterministic.
"""

import multiprocessing
import os


def run_jobs(fn, specs, procs=None):
    """[fn(s) for s in specs], computed on up to `procs` processes."""
    specs = list(specs)
    if not specs:
        return []
    procs = max(1, min(len(specs), procs or os.cpu_count() or 1))
    if procs == 1:
        return [fn(s) for s in specs]
    ctx = multiprocessing.get_context("fork")
    with ctx.Pool(procs) as pool:
        return pool.map(fn, specs, chunksize=1)


def cores(specs, procs=None):
    """How many processes run_jobs would use, for a resource line."""
    return max(1, min(len(list(specs)), procs or os.cpu_count() or 1))
