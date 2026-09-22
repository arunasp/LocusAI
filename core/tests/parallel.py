"""Experiment-side alias for the execution layer.

The experiments import `parallel.run_jobs`; placement now lives in
core/py/locus/execution.py so every caller (reader, learner driver,
experiments) shares one implementation and one set of rules. Kept as a
name rather than folded away, because the experiment scripts run as
standalone files with tests/ on the path and no locus import.
"""

import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "py"))

from locus.execution import cores as _cores  # noqa: E402
from locus.execution import map_jobs  # noqa: E402

run_jobs = map_jobs


def cores(specs, procs=None):
    """How many processes run_jobs would use for these specs."""
    return _cores(len(list(specs)), procs)
