"""Kernels and measures shared by the tests and experiments.

WHY THESE AND NOT EVERY REPEATED HELPER. A fixture belongs here when
several files need the SAME construction and a divergence between copies
would silently move a published figure -- a kernel feeds the dynamics
every measurement is taken on, so two copies drifting apart makes two
experiments incomparable while both keep passing. Each function below
was proved identical across its call sites by comparing OUTPUT before it
was moved, not by reading the source and judging it the same.

WHAT IS DELIBERATELY LEFT ALONE. `main` is per-file by nature. `bpb`,
`corrupt`, `job` and `run` are variants that differ by experiment, and
merging them would invent a shared meaning they do not have. The
`score` / `score_one` family in the English experiments is a separate
job: it carries a module-level STORE global that `map_jobs` relies on,
so moving it changes how the workers see that state and needs its own
verification rather than riding along here.
"""

import random


def random_kernel(n, seed, deg=6):
    """Uniformly random targets. No block structure, so nothing
    distinguishes one region from another.

    Shared by exp_capacity.py and exp_repertoire.py, which is what makes
    their measurements about the same field.
    """
    rnd = random.Random(seed)
    rows = []
    for i in range(n):
        row = [0.0] * n
        row[i] = 0.3 + 0.2 * rnd.random()
        for _ in range(deg):
            row[rnd.randrange(n)] += rnd.random()
        s = sum(row)
        rows.append([v / s for v in row])
    return rows


def ring_kernel(n, seed, reach=3):
    """Row-stochastic kernel with local connectivity and a positive
    diagonal.

    Local rather than random because a random sparse kernel is
    frequently reducible, and a failure on a reducible kernel says
    nothing about the dynamics or the parameter under test.
    """
    rnd = random.Random(seed)
    rows = []
    for i in range(n):
        row = [0.0] * n
        row[i] = 0.4 + 0.3 * rnd.random()
        for d in range(1, reach + 1):
            w = (0.6 ** d) * (0.4 + rnd.random())
            row[(i + d) % n] += w
            row[(i - d) % n] += w
        s = sum(row)
        rows.append([v / s for v in row])
    return rows


def participation_ratio(a):
    """Effective number of active units: (sum a)^2 / sum(a^2).

    THRESHOLD-FREE, which is the point: an earlier reading of 41% active
    was an artefact of a chosen cutoff, and the residency figures that
    depended on one could not say whether the field was sparse. This
    measure needs no cutoff, so it can.
    """
    s1 = sum(a)
    s2 = sum(v * v for v in a)
    return (s1 * s1 / s2) if s2 > 0 else 0.0
