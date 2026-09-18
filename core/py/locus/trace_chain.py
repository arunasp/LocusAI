"""Trace chains: the effective dynamics seen by a restricted observer.

WHY THIS EXISTS
The tier model needs coarse levels whose values follow from fine-level
change without being stored twice. A trace chain is exactly that: given a
kernel P over the whole state space and a subset A, the trace on A is the
kernel an observer confined to A would measure. It is DERIVED, never
written -- so there is no shared cell, no second writer, and no race.

This is the classical censored (or "watched") chain, obtained by stochastic
complementation: partition P into blocks over A and its complement B, then

    S_A(P) = P_AA + P_AB (I - P_BB)^-1 P_BA

The inverse term sums over every excursion that leaves A and comes back, so
the result is exact rather than an approximation of the coarse dynamics.

WHAT IS MEASURED, not assumed (see core/tests/test_trace_chain.py):
  - TRANSITIVITY holds to 9 decimal places: tracing to A then to B (B a
    subset of A) equals tracing straight to B. This is what makes a
    HIERARCHY well defined; without it a three-level model would depend
    on the order the levels were derived in.
  - The trace preserves the stationary distribution restricted to A and
    renormalised, to 8 places.
  - Naive block-averaging (what a sum tree or plain reduction computes)
    does NOT equal the trace, and it is the one that gets the long-run
    measure wrong. That is why the linear solve is worth paying for.
  - A subset whose complement is CLOSED under P has no trace chain at
    all, and this raises rather than returning a non-stochastic matrix.
    At sparse connectivity that case is common, so a coarse partition
    cannot be chosen arbitrarily -- excursions must return.

COST, measured: the solve is cubic in the dropped set (0.14 ms at 8
dropped states, 30.9 ms at 64), so exact tracing is a consolidation-time
operation, not a per-tick one. A truncated Neumann series
(P_AA + P_AB (sum_j P_BB^j) P_BA) is the cheap per-tick form; on a
character-prediction task one excursion step recovered the exact
perplexity to within 0.06%, while ignoring excursions entirely cost 173%.

WHAT IS NOT CLAIMED HERE
Nothing in this module learns. The kernel is an input. A learning rule
that produces P from experience is a separate concern, and deliberately
so: this file is the level-coupling operator only, and can be tested on
its own.

Stdlib only, no numpy. The sizes that matter for the tier model are small
(the active set is bounded by design), and a dependency-free module runs
in the CI worker as well as on the GPU host.
"""


def _solve(a, b):
    """Solve a X = b for X, by Gauss-Jordan with partial pivoting.

    Solving beats forming an explicit inverse: fewer operations and it
    keeps the conditioning of the original system rather than squaring it.
    Partial pivoting is not decoration -- I - P_BB can be arbitrarily
    ill-conditioned when B is nearly closed under P, which is precisely
    the interesting case (a coarse level that rarely returns).
    """
    n = len(a)
    if n == 0:
        return []
    m = len(b[0])
    # Work on copies: callers keep their matrices.
    aug = [list(a[i]) + list(b[i]) for i in range(n)]
    for col in range(n):
        piv = max(range(col, n), key=lambda r: abs(aug[r][col]))
        if abs(aug[piv][col]) < 1e-15:
            raise ValueError(
                "singular system at column %d: the complement is closed "
                "under P, so an observer in A never sees a return" % col
            )
        aug[col], aug[piv] = aug[piv], aug[col]
        p = aug[col][col]
        aug[col] = [v / p for v in aug[col]]
        for r in range(n):
            if r == col:
                continue
            f = aug[r][col]
            if f != 0.0:
                aug[r] = [
                    vr - f * vc for vr, vc in zip(aug[r], aug[col])
                ]
    return [row[n:n + m] for row in aug]


def _submatrix(p, rows, cols):
    return [[p[i][j] for j in cols] for i in rows]


def _matmul(a, b):
    n, k, m = len(a), len(b), len(b[0])
    return [[sum(a[i][t] * b[t][j] for t in range(k)) for j in range(m)]
            for i in range(n)]


def _norm_rows(p):
    out = []
    for row in p:
        s = sum(row)
        out.append([v / s for v in row] if s > 0 else list(row))
    return out


def trace_chain(p, keep):
    """Return the exact trace of kernel ``p`` on the states in ``keep``.

    ``p`` is row-stochastic, ``keep`` a list of state indices.
    Raises ValueError if the complement is closed under ``p``, because
    then no effective kernel on ``keep`` exists -- a silent
    non-stochastic result would be worse than a refusal.
    """
    n = len(p)
    keep = sorted(keep)
    drop = [i for i in range(n) if i not in set(keep)]
    if not drop:
        # Tracing to the whole space is the identity operation. Returned as
        # a copy so a caller cannot mutate the input through the result.
        return [list(row) for row in p]
    p_aa = _submatrix(p, keep, keep)
    p_ab = _submatrix(p, keep, drop)
    p_ba = _submatrix(p, drop, keep)
    p_bb = _submatrix(p, drop, drop)
    k = len(drop)
    i_minus = [
        [(1.0 if r == c else 0.0) - p_bb[r][c] for c in range(k)]
        for r in range(k)
    ]
    excursions = _solve(i_minus, p_ba)
    out = []
    for r in range(len(keep)):
        row = list(p_aa[r])
        for c in range(len(keep)):
            row[c] += sum(
                p_ab[r][t] * excursions[t][c] for t in range(k)
            )
        out.append(row)
    return out


def truncated_trace(p, keep, depth):
    """Cheap per-tick approximation: allow excursions of at most ``depth``
    steps, then renormalise.

    ``depth`` of -1 is the no-recursion control (excursions ignored
    entirely), which is the candidate to beat rather than a sensible
    setting -- it cost 173% perplexity against the exact trace where
    depth=0 cost 0.06%.

    Renormalisation is needed because truncation drops excursion mass, so
    rows would otherwise sum to less than one. It is also why the result
    can keep the RANKING right while the probabilities are wrong: use
    this for competition, not as a calibrated distribution.
    """
    keep = sorted(keep)
    drop = [i for i in range(len(p)) if i not in set(keep)]
    p_aa = _submatrix(p, keep, keep)
    if depth < 0 or not drop:
        return _norm_rows(p_aa)
    p_ab = _submatrix(p, keep, drop)
    p_ba = _submatrix(p, drop, keep)
    p_bb = _submatrix(p, drop, drop)
    k = len(drop)
    acc = [[1.0 if r == c else 0.0 for c in range(k)] for r in range(k)]
    term = [row[:] for row in acc]
    for _ in range(depth):
        term = _matmul(term, p_bb)
        acc = [[acc[r][c] + term[r][c] for c in range(k)]
               for r in range(k)]
    corr = _matmul(_matmul(p_ab, acc), p_ba)
    m = len(keep)
    return _norm_rows([[p_aa[r][c] + corr[r][c] for c in range(m)]
                       for r in range(m)])


def lumped_chain(p, blocks):
    """NEGATIVE CONTROL: naive aggregation by averaging rows in a block.

    This is what a sum tree or a plain reduction computes, and it is what
    a trace chain is often mistaken for. It is only equal to the trace
    when the partition happens to be strongly lumpable. Kept here so the
    tests can demonstrate the difference rather than assert it, because
    if the two agreed there would be no reason to pay for the solve.
    """
    out = []
    for br in blocks:
        row = []
        for bc in blocks:
            row.append(
                sum(p[i][j] for i in br for j in bc) / len(br)
            )
        out.append(row)
    return out


def stationary(p, iters=20000, tol=1e-13):
    """Stationary distribution by power iteration.

    Deliberately not an eigensolver: power iteration is a handful of
    lines, has no dependency, and converges for the small irreducible
    kernels this module is used with. Returns early on convergence, so
    the iteration cap is a safety bound rather than the usual cost.
    """
    n = len(p)
    v = [1.0 / n] * n
    for _ in range(iters):
        nxt = [
            sum(v[i] * p[i][j] for i in range(n)) for j in range(n)
        ]
        s = sum(nxt)
        nxt = [x / s for x in nxt]
        if max(abs(a - b) for a, b in zip(v, nxt)) < tol:
            return nxt
        v = nxt
    return v


def row_sums(p):
    return [sum(row) for row in p]
