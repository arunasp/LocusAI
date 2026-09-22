"""Read a prompt against a knowledge store.

  prompt.py KNOW [--file F] [--no-learn] [--outcome X]
  prompt.py KNOW --score

A prompt is one more stream: each byte is predicted from what came before
it, then learned (unless --no-learn). The report gives bits per byte,
the least expected spans, and the most expected next bytes after the
prompt. Learned rows are tagged into the transient tier (KNOW.tags), which
later readers see; the outcome X (default 1; 0 records none) is logged at
the current tick. LocusAI sleeps by itself, before learning more, once
its sleep pressure reaches the threshold; sleep.py forces a sleep.

--score rescores the held-out test split with learning off. On the
generation the device measured, it must equal the device's figure.
"""

import os
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "py"))
sys.path.insert(0, os.path.join(HERE, "..", "tests"))

from locus.encode import repo_files  # noqa: E402
from locus.knowledge import Knowledge  # noqa: E402


def show(b):
    c = chr(b)
    return repr(c)[1:-1] if not c.isprintable() or c in "\\'" else c


def score(k):
    import exp_learn_stream as X
    m = k.meta
    files = [d for p, d in repo_files(m["corpus"])
             if X.split(p, m["salt"]) == "test"]
    bits = n = 0
    for data in files:
        b = k.read(data, learn=False)
        bits += sum(b)
        n += len(b)
    got = bits / max(n, 1)
    gen = m.get("generation", 0)
    print("test split  %d files, %d predictions" % (len(files), n))
    if gen:
        print("reader      %.12f bits per byte (generation %d)" % (got, gen))
        print("device      %.12f bits per byte (generation 0)"
              % m["test_bpb"])
        print("difference  %.2e -- learning since generation 0, not a "
              "reproduction check" % (got - m["test_bpb"]))
        return 0
    print("reader      %.12f bits per byte" % got)
    print("device      %.12f bits per byte" % m["test_bpb"])
    print("difference  %.2e" % (got - m["test_bpb"]))
    return 0 if abs(got - m["test_bpb"]) < 1e-9 and n == m["n_test"] else 1


def main(argv):
    if not argv:
        print(__doc__)
        return 2
    t0 = time.time()
    k = Knowledge(argv[0])
    rest, src, learn, outcome = argv[1:], None, True, 1.0
    load = time.time() - t0
    if rest[:1] == ["--score"]:
        print("store       %s (%s), loaded in %.1f s"
              % (argv[0], k.meta["learner"], load))
        return score(k)
    while rest:
        if rest[0] == "--file" and len(rest) > 1:
            src, rest = rest[1], rest[2:]
        elif rest[0] == "--no-learn":
            learn, rest = False, rest[1:]
        elif rest[0] == "--outcome" and len(rest) > 1:
            outcome, rest = float(rest[1]), rest[2:]
        else:
            print("unknown argument: %s" % rest[0])
            return 2
    data = (open(src, "rb").read() if src else sys.stdin.buffer.read())
    if len(data) < 2:
        print("prompt too short: %d bytes" % len(data))
        return 2
    t1 = time.time()
    bits = k.read(data, learn=learn)
    t2 = time.time()
    if k.last_sleep:
        print("slept     by itself before reading: %d rows captured, %d "
              "lapsed; generation %d" % (k.last_sleep + (
                  k.meta.get("generation", 0),)))
    m = k.meta
    print("store     %s: %s, %d units, trained on %d bytes, loaded %.1f s"
          % (os.path.basename(argv[0]), m["learner"], k.U,
             m["train_bytes"], load))
    print("prompt    %d bytes, read in %.2f s, learning %s"
          % (len(data), t2 - t1, "on" if learn else "off"))
    print("surprise  %.3f bits per byte (store's held-out test: %.3f)"
          % (sum(bits) / len(bits), m["test_bpb"]))
    w = 8
    spans = []
    for i in range(0, len(bits), w):
        seg = bits[i:i + w]
        spans.append((sum(seg) / len(seg), i))
    spans.sort(reverse=True)
    print("least expected spans (bits per byte):")
    for v, i in spans[:5]:
        text = "".join(show(b) for b in data[i + 1:i + 1 + w])
        print("  %6.2f  at %5d  %s" % (v, i + 1, text))
    exp = k.expect(k.drive(data, len(data) - 1))
    print("expected next: " + ", ".join(
        "'%s' %.2f" % (show(b), p) for b, p in exp))
    if learn:
        if outcome:
            k.outcome(outcome)
        k.save_transient()
        print("memory    %d rows in the transient tier at tick %d; "
              "outcome %s; sleep pressure %.4f of %g"
              % (k.tagged(), k.clock,
                 "%+g" % outcome if outcome else "none", k.pressure(),
                 k.threshold))
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
