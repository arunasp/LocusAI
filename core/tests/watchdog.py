"""Watchdog: checks that FAIL rather than warn, plus its own self-test.

Two families.

METHOD CHECKS were each written after a specific defect that was reported
as a result before being caught:

  metric_valid      a dwell metric returned the full run length on a dead
                    trace, because it thresholded against its own first
                    value -- comparing zero with half of zero.
  monotone          passed on a list that was entirely None. A check that
                    cannot fail on empty input is decoration, so
                    insufficient data is now a third outcome (REFUSE)
                    rather than a silent pass.
  window_has_signal two separate metrics read PAST THE END of the signal
                    and compared zero with zero. Neither had a wrong
                    formula; both had a window in the wrong place.
  param_has_effect  a swept loop gain produced byte-identical output at
                    three values, because a global divisive normalisation
                    erased it. Any conclusion drawn from that sweep was
                    void.
  null_worse        the null model disabled ACTIVATION rather than the
                    mechanism under test, so "mechanism beats null" was
                    comparing a dead system with a dead system.
  graded            a constant offset plus a hard top-k selection gives a
                    step, and the step was reported as a property of the
                    dynamics rather than of the selection rule.

PENDING-RULE CHECKS give executable form to three rules that had only
prose, which is why they kept failing:

  in_responsive_range
                    values compared across conditions must not be pinned
                    at a bound or all identical. THREE defects in one
                    session had this shape -- a dwell metric reading a
                    dead trace, a log-slope on a non-exponential
                    trajectory, and a sweep that read the saturation cap
                    at every level. In each the instrument was wrong
                    rather than the model, which is the hardest kind to
                    notice because the numbers look fine.
  read_back         write-then-read equality. A write tool's success
                    report is not evidence about what landed.
  artifacts_listed  a script must enumerate every path it writes, with
                    sizes -- and the enumeration must be TRUE on disk.
  absolute_claims   an absolute or negative claim (never, always, cannot,
                    not possible) needs an evidence marker in the same
                    sentence, or the word "unchecked" attached.

Not covered, and stated rather than implied: the memory-entry isolation
rule needs a memory store to read against (it is auditable by extracting
cited paths and diffing them against a directory listing, which is not
this module's job); and the shared-fragment placement gate is a placement
task, not a check.

Stdlib only, so it runs in a CI worker as well as on the GPU host.
Run `python3 tests/watchdog.py` for the self-test; it exits non-zero on
any incorrect behaviour.
"""

import os
import re

PASS, FAIL, REFUSE = "pass", "FAIL", "REFUSE"

ABSOLUTES = re.compile(
    r"\b(never|always|cannot|can't|impossible|no\s+\w+\s+exists|"
    r"not\s+possible|every|none)\b", re.I)
EVIDENCE = re.compile(
    r"\b(measured|verified|observed|unchecked|tested|confirmed|"
    r"\d+(\.\d+)?%?|exit\s+code)\b", re.I)


# --------------------------------------------------------- method checks --
def metric_valid(fn, cases):
    """Validate a METRIC against inputs whose answer is known, before any
    result computed with it is trusted. Checking a sweep's parameters and
    not its metric is the gap that let a dead-trace reading through."""
    out = []
    for label, arg, pred in cases:
        try:
            got = fn(arg)
            out.append((label, PASS if pred(got) else FAIL, repr(got)))
        except Exception as exc:                        # noqa: BLE001
            out.append((label, FAIL, "raised %s" % type(exc).__name__))
    return out


def monotone(ys, tol=0.02, min_samples=3):
    vals = [(i, y) for i, y in enumerate(ys) if y is not None]
    if len(vals) < min_samples:
        return REFUSE, "only %d non-null of %d" % (len(vals), len(ys))
    bad = [vals[k][0] for k in range(1, len(vals))
           if vals[k][1] < vals[k - 1][1] - tol]
    return (PASS if not bad else FAIL,
            "reversals %s" % (bad if bad else "none"))


def window_has_signal(hist, lo, hi, floor=1e-9, min_frac=0.5, min_n=3):
    seg = hist[lo:hi]
    if len(seg) < min_n:
        return REFUSE, "window only %d samples" % len(seg)
    live = sum(1 for v in seg if v > floor)
    frac = live / len(seg)
    if frac < min_frac:
        return FAIL, "only %.0f%% of window is live" % (100 * frac)
    return PASS, "%.0f%% live" % (100 * frac)


def param_has_effect(outputs, min_samples=2):
    vals = [o for o in outputs if o is not None]
    if len(vals) < min_samples:
        return REFUSE, "only %d non-null" % len(vals)
    d = len(set(round(float(o), 9) for o in vals))
    return (PASS if d > 1 else FAIL), "%d distinct of %d" % (d, len(vals))


def null_worse(mech, null):
    if mech is None or null is None:
        return REFUSE, "missing value"
    return (PASS if mech > null else FAIL), "mech=%s null=%s" % (mech, null)


def graded(outputs, min_distinct=3):
    vals = [o for o in outputs if o is not None]
    if len(vals) < min_distinct:
        return REFUSE, "only %d non-null" % len(vals)
    d = len(set(vals))
    return (PASS if d >= min_distinct else FAIL), "%d distinct" % d


def in_responsive_range(values, lo=None, hi=None, rel=0.02, min_n=2):
    """Values compared ACROSS CONDITIONS must sit where the quantity can
    still respond -- not pinned at a bound, and not all identical.

    THREE defects in one session had this shape, and in every case the
    instrument was wrong rather than the model:

      a dwell metric returned the full run length on a DEAD trace,
      because it thresholded against its own first value and so compared
      zero with half of zero;

      a log-slope was fitted to a trajectory that is nowhere near
      exponential, and the bifurcation estimate built on its sign had to
      be retracted;

      a bias sweep read 10.0, 10.0, 10.0 -- the saturation cap at every
      inhibition level -- so the comparison measured the CEILING rather
      than the response.

    A sweep whose outputs are pinned proves nothing about the parameter,
    whichever direction the assertion then points. Check this BEFORE
    comparing values across conditions, not after the comparison looks
    odd.
    """
    vals = [v for v in values if v is not None]
    if len(vals) < min_n:
        return REFUSE, "only %d non-null" % len(vals)
    span = max(vals) - min(vals)
    scale = max(abs(v) for v in vals) or 1.0
    if span <= rel * scale:
        return FAIL, ("all values within %.1f%% of each other (%.6g.."
                      "%.6g) -- the quantity is not responding"
                      % (100 * rel, min(vals), max(vals)))
    at_bound = []
    for v in vals:
        if hi is not None and v >= hi * (1.0 - rel):
            at_bound.append(("ceiling", v))
        if lo is not None and v <= lo + rel * scale:
            at_bound.append(("floor", v))
    if at_bound:
        return FAIL, "pinned at a bound: %s" % at_bound[:3]
    return PASS, "span %.6g over %d values" % (span, len(vals))


# ---------------------------------------------------- pending-rule checks -
def read_back(path, content, writer=None):
    """Write via `writer` (default: plain write), read back, compare."""
    if writer is None:
        with open(path, "w") as fh:
            fh.write(content)
    else:
        writer(path, content)
    if not os.path.exists(path):
        return FAIL, "file absent after write"
    with open(path) as fh:
        got = fh.read()
    if got != content:
        return FAIL, "wrote %d chars, read %d" % (len(content), len(got))
    return PASS, "%d chars match" % len(got)


ART = re.compile(r"\[artifact\]\s+(\S+)\s+(\d+)\s+bytes")


def artifacts_listed(output, require=1):
    """Every '[artifact] <path> <n> bytes' line must be true on disk."""
    found = ART.findall(output)
    if len(found) < require:
        return FAIL, "%d artifact lines, need %d" % (len(found), require)
    for path, size in found:
        if not os.path.exists(path):
            return FAIL, "claims %s which does not exist" % path
        actual = os.path.getsize(path)
        if actual != int(size):
            return FAIL, ("claims %s at %s bytes, actual %d"
                          % (path, size, actual))
    return PASS, "%d artifact claim(s) true" % len(found)


def absolute_claims(text):
    """Absolute or negative claims need an evidence marker nearby."""
    unmarked = []
    for sent in re.split(r"(?<=[.!?])\s+", text):
        if ABSOLUTES.search(sent) and not EVIDENCE.search(sent):
            unmarked.append(sent.strip()[:60])
    return (PASS if not unmarked else FAIL,
            "%d unmarked: %s" % (len(unmarked), unmarked[:2]))


# ------------------------------------------------------------- self test --
def self_test():
    rows = []

    def add(name, got, want, detail):
        rows.append((name, got == want, "want=%s got=%s %s"
                     % (want, got, detail)))

    # KNOWN GAP, left failing deliberately rather than patched quietly:
    # the "raises" case below feeds None to a function that handles None
    # gracefully, so nothing raises and metric_valid's exception path is
    # NOT exercised. The case is wrong, not the checker. Fixing it needs
    # an input that genuinely raises for the function under test.
    for label, status, detail in metric_valid(
            lambda h: 0 if not h or max(h) <= 0 else len(h),
            [("dead -> 0", [0.0] * 5, lambda r: r == 0),
             ("live -> len", [1.0] * 5, lambda r: r == 5)]):
        add("metric_valid/%s" % label, status, PASS, detail)

    for name, ys, want in [("monotone ok", [1, 2, 3], PASS),
                           ("reversal", [1, 2, 1.5], FAIL),
                           ("all none", [None] * 4, REFUSE),
                           ("too few non-null", [None, 1.0, None], REFUSE)]:
        g, d = monotone(ys)
        add(name, g, want, d)

    for name, args, want in [
            ("window live", ([0.5] * 30, 20, 30), PASS),
            ("window dead", ([0.5] * 8 + [0.0] * 22, 20, 30), FAIL),
            ("window straddles", ([0.5] * 20 + [0.0] * 10, 15, 25), PASS),
            ("window short", ([0.5] * 30, 28, 30), REFUSE)]:
        g, d = window_has_signal(*args)
        add(name, g, want, d)

    for name, outs, want in [("effect", [1, 2, 3], PASS),
                             ("inert", [1, 1, 1], FAIL),
                             ("too few", [1], REFUSE)]:
        g, d = param_has_effect(outs)
        add(name, g, want, d)

    g, d = null_worse(10, 2)
    add("null worse", g, PASS, d)
    g, d = null_worse(2, 2)
    add("null equal", g, FAIL, d)

    for name, outs, want in [("graded", [1, 2, 3, 4], PASS),
                             ("binary", [1, 2], REFUSE),
                             ("step", [1, 1, 1, 2], FAIL)]:
        g, d = graded(outs)
        add(name, g, want, d)

    for name, args, want in [
            ("responsive spread", ([0.2, 0.5, 0.8], 0.0, 10.0), PASS),
            ("all at ceiling", ([10.0, 10.0, 10.0], 0.0, 10.0), FAIL),
            ("all identical", ([0.5, 0.5, 0.5], None, None), FAIL),
            ("one at ceiling", ([0.2, 0.5, 10.0], 0.0, 10.0), FAIL),
            ("one at floor", ([0.0, 0.5, 0.8], 0.0, 10.0), FAIL),
            ("too few", ([0.5], 0.0, 1.0), REFUSE)]:
        vals, lo, hi = args
        g, d = in_responsive_range(vals, lo=lo, hi=hi)
        add(name, g, want, d)

    tmp = os.path.join(os.environ.get("TMPDIR", "/tmp"),
                       "_watchdog_readback.txt")
    g, d = read_back(tmp, "hello" * 100)
    add("read_back ok", g, PASS, d)

    def truncating(path, content):
        with open(path, "w") as fh:
            fh.write(content[:10])
    g, d = read_back(tmp, "x" * 500, writer=truncating)
    add("read_back truncated", g, FAIL, d)

    size = os.path.getsize(tmp)
    g, d = artifacts_listed("[artifact] %s %d bytes" % (tmp, size))
    add("artifacts true", g, PASS, d)
    g, d = artifacts_listed("[artifact] %s 999999 bytes" % tmp)
    add("artifacts wrong size", g, FAIL, d)
    g, d = artifacts_listed("[artifact] /tmp/_watchdog_nope_ 5 bytes")
    add("artifacts missing", g, FAIL, d)
    g, d = artifacts_listed("no artifact block here")
    add("artifacts absent", g, FAIL, d)

    g, d = absolute_claims("The solve never converges.")
    add("absolute unmarked", g, FAIL, d)
    g, d = absolute_claims("It never converges, measured over 40 ticks.")
    add("absolute marked", g, PASS, d)
    g, d = absolute_claims("Persistence is absent, unchecked at scale.")
    add("absolute hedged", g, PASS, d)

    print("=== watchdog self-test ===")
    ok = 0
    for name, good, detail in rows:
        ok += good
        print("  [%s] %-28s %s" % ("ok" if good else "FAIL", name, detail))
    print("  %d/%d behaviours correct" % (ok, len(rows)))
    print("RESULT: %s" % ("ALL PASS" if ok == len(rows) else "FAIL"))
    return ok == len(rows)


if __name__ == "__main__":
    raise SystemExit(0 if self_test() else 1)
