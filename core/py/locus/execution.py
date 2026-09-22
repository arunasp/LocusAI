"""Where work runs: the device first, then every core, then elsewhere.

One place decides placement, so callers describe work and not machinery.
The standing constraint (doc/core/ROADMAP.md) is the GPU first and all
CPU cores second; until now each caller re-implemented that, and the
three paths drifted (pools in the experiments, a lock in the learner
driver, a plain loop in the reader).

Two rules hold across every backend:

- RESULTS COME BACK IN THE ORDER OF THE SPECS, so a figure never depends
  on how many workers ran. Reductions elsewhere sum in that order.
- A JOB IS A SPEC PLUS A FUNCTION NAME, not a closure over live state.
  That is what lets the same job run in this process, in another one, or
  later in another container: `learn_device`'s input-file/output-file
  contract is the shape that already crosses those boundaries.

`resources()` reports what a run cost (wall, CPU core-seconds, and the
device seconds a caller reports), which every result is required to
state beside its accuracy.
"""

import contextlib
import multiprocessing
import os
import subprocess
import time

_DEVICE = None


def cores(n_jobs, procs=None):
    """Processes that would be used for n_jobs."""
    return max(1, min(n_jobs, procs or os.cpu_count() or 1))


def map_jobs(fn, specs, procs=None):
    """[fn(s) for s in specs] on up to `procs` processes, in order."""
    specs = list(specs)
    if not specs:
        return []
    n = cores(len(specs), procs)
    if n == 1:
        return [fn(s) for s in specs]
    ctx = multiprocessing.get_context("fork")
    with ctx.Pool(n) as pool:
        return pool.map(fn, specs, chunksize=1)


def device_lock():
    """The one lock guarding the GPU; shared by forked workers."""
    global _DEVICE
    if _DEVICE is None:
        _DEVICE = multiprocessing.get_context("fork").Lock()
    return _DEVICE


@contextlib.contextmanager
def device():
    """Hold the GPU for one job. The card is one resource: jobs queue
    here rather than each driver inventing its own lock."""
    lock = device_lock()
    lock.acquire()
    t0 = time.time()
    try:
        yield
    finally:
        lock.release()
        device.seconds = getattr(device, "seconds", 0.0) + time.time() - t0


def run_binary(argv, check=True):
    """Run a compiled job (the device contract: argv in, files out)."""
    r = subprocess.run(argv, capture_output=True, text=True)
    if check and r.returncode != 0:
        raise RuntimeError("%s exited %d: %s"
                           % (argv[0], r.returncode, r.stderr.strip()))
    return r


class resources:
    """Wall, CPU core-seconds (this process and its children) and any
    device seconds a caller adds, for the line every result carries."""

    def __init__(self):
        self.device = 0.0

    def __enter__(self):
        self._t, self._c = time.time(), os.times()
        return self

    def __exit__(self, *exc):
        c = os.times()
        self.wall = time.time() - self._t
        self.cpu = ((c.user + c.system + c.children_user + c.children_system)
                    - (self._c.user + self._c.system + self._c.children_user
                       + self._c.children_system))
        return False

    def line(self, procs=None):
        return ("resources: wall %.1f s, CPU %.1f core-s%s%s"
                % (self.wall, self.cpu,
                   " on %d processes" % procs if procs else "",
                   ", GPU device %.1f s" % self.device if self.device
                   else ""))
