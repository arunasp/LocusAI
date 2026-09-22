"""Per-input stream learners on the GPU, scored as exp_learn_stream.py.

BranchLearner and the CLSLearner stores update each unit's row from that
unit's own error only, so this driver lays out every unit's events in the
learners' own chronological order (CLS replay included, from the same
seeded shuffle) and hands them to core/gpu/learn_stream.cpp, which
replays each unit in one GPU block and reads out every scored position.
The arithmetic follows the Python learners' fp64 operations in order;
checked through scores, which equal theirs to 12 decimals on a frozen
corpus when the program is built without hipcc. The positive-drive sum
is reduced in a different order, so scores may differ in the last bits.

Not here: learners whose update depends on the whole network's online
prediction (tag, faithful, NE gain) and the online training score; both
are sequential by construction and stay in exp_learn_stream.py.

Run: python3 tests/exp_learn_gpu.py ROOT BINARY [--salts A,B,...]
                         [--only LABEL,...] [--engine host|device]
--engine device (BINARY = learn_device) does encoding, event sorting,
learning, the control and the lam grid on the device; host (BINARY =
learn_stream) builds events in Python.
--salts repeats everything on other splits (the file hash is salted; the
empty salt is the default split) and summarises each learner against the
control per split and across splits. --only keeps the learners whose
labels contain one of the given substrings.
"""

import math
import multiprocessing
import os
import random
import struct
import subprocess
import sys
import tempfile
import time
from array import array

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "py"))
sys.path.insert(0, os.path.dirname(__file__))

from locus.encode import BYTE_UNITS, NgramEncoder, repo_files  # noqa: E402
from locus.learn import unit_order                              # noqa: E402
from locus.plasticity import Plasticity                         # noqa: E402
import exp_learn_stream as X                                    # noqa: E402

RATE = Plasticity().rate

# (label, kind, options). "branch": one store; "cls": cortex + hippocampus.
LEARNERS = (
    ("branch, constant rate", "branch", {"metaplastic": False}),
    ("check: branch, metaplastic (1/n)", "branch", {"metaplastic": True}),
    ("cls: hippocampus, cortex, replay", "cls",
     {"hippo": True, "replay": True}),
    ("cls without replay", "cls", {"hippo": True, "replay": False}),
    ("check: cls cortex alone, no replay", "cls",
     {"hippo": False, "replay": False}),
    ("cls, neocortex also metaplastic (1/n)", "cls",
     {"hippo": True, "replay": True, "cortex_meta": True}),
)


class Events:
    """Chronological (unit, next byte, local factor) events of one store."""

    def __init__(self):
        self.unit = array("i")
        self.nxt = array("B")
        self.loc = array("d")

    def add(self, u, b, loc):
        self.unit.append(u)
        self.nxt.append(b)
        self.loc.append(loc)

    def csr(self, n_units):
        """Events grouped by unit, each unit's own order kept."""
        count = [0] * (n_units + 1)
        for u in self.unit:
            count[u + 1] += 1
        for i in range(n_units):
            count[i + 1] += count[i]
        fill = count[:-1]
        nxt = bytearray(len(self.unit))
        loc = array("d", bytes(8 * len(self.unit)))
        for u, b, lo in zip(self.unit, self.nxt, self.loc):
            k = fill[u]
            nxt[k] = b
            loc[k] = lo
            fill[u] = k + 1
        return array("q", count), bytes(nxt), loc


def build(kind, opts, enc, train):
    """Event stores for one learner, mirroring locus.learn exactly."""
    top = max(enc.orders)
    if kind == "branch":
        ev = Events()
        seen = {}
        for data in train:
            for pos in range(len(data) - 1):
                nxt = data[pos + 1]
                for u in enc.units_at(data, pos):
                    loc = 1.0
                    if opts["metaplastic"]:
                        k = seen.get(u, 0) + 1
                        seen[u] = k
                        loc = 1.0 / (RATE * k)
                    ev.add(u, nxt, loc)
        return [ev]
    cortex, hippo = Events(), Events()
    rnd = random.Random(1)
    seen = {}
    cseen = {}

    def cloc(u):
        # Neocortex rate: constant, or 1/n per unit (GPU-only variant).
        if not opts.get("cortex_meta"):
            return 1.0
        k = cseen.get(u, 0) + 1
        cseen[u] = k
        return 1.0 / (RATE * k)

    earlier = []
    for fi, data in enumerate(train):
        episodes = []
        for pos in range(len(data) - 1):
            units = enc.units_at(data, pos)
            nxt = data[pos + 1]
            if opts["hippo"]:
                for u in units:
                    if unit_order(enc, u) == top:
                        k = seen.get(u, 0) + 1
                        seen[u] = k
                        hippo.add(u, nxt, 1.0 / (RATE * k))
            for u in units:
                cortex.add(u, nxt, cloc(u))
            episodes.append((fi, pos))
        if opts["replay"] and episodes:
            old = (rnd.sample(earlier, min(len(earlier), len(episodes)))
                   if earlier else [])
            batch = episodes + old
            rnd.shuffle(batch)
            for ei, pos in batch:
                d = train[ei]
                for u in enc.units_at(d, pos):
                    cortex.add(u, d[pos + 1], cloc(u))
        earlier.extend(episodes)
    return [cortex, hippo] if opts["hippo"] else [cortex]


def positions(enc, files, top):
    pptr, punit, phippo, pnext = array("q", [0]), array("i"), bytearray(), \
        bytearray()
    for data in files:
        for pos in range(len(data) - 1):
            for u in enc.units_at(data, pos):
                punit.append(u)
                phippo.append(1 if unit_order(enc, u) == top else 0)
            pptr.append(len(punit))
            pnext.append(data[pos + 1])
    return pptr, punit, bytes(phippo), bytes(pnext)


def write_input(enc, stores, scored):
    """CPU side: group events by unit and write the binary's input file."""
    pptr, punit, phippo, pnext = scored
    fd, inp = tempfile.mkstemp(suffix=".in")
    with os.fdopen(fd, "wb") as f:
        f.write(struct.pack("<idi", enc.n, RATE, len(stores)))
        for ev in stores:
            rowptr, nxt, loc = ev.csr(enc.n)
            f.write(rowptr.tobytes())
            f.write(struct.pack("<q", len(nxt)))
            f.write(nxt)
            f.write(loc.tobytes())
        f.write(struct.pack("<q", len(pnext)))
        f.write(pptr.tobytes())
        f.write(punit.tobytes())
        f.write(phippo)
        f.write(pnext)
    return inp, len(pnext)


def execute(binary, inp, p):
    """Device side: run the binary on a written input; returns
    (dnext, dpos, note)."""
    out = inp[:-3] + ".out"
    r = subprocess.run([binary, inp, out], capture_output=True, text=True)
    os.unlink(inp)
    if r.returncode != 0:
        raise SystemExit("FAIL: %s exited %d: %s" % (binary, r.returncode,
                                                     r.stderr.strip()))
    res = array("d")
    with open(out, "rb") as f:
        res.frombytes(f.read())
    os.unlink(out)
    return res[:p], res[p:], r.stdout.strip()


def run(binary, enc, stores, scored):
    return execute(binary, *write_input(enc, stores, scored))


def bpb(dnext, dpos, lo, hi, lam):
    bits = 0.0
    for i in range(lo, hi):
        bits -= math.log2((dnext[i] + lam / BYTE_UNITS) / (dpos[i] + lam))
    return bits / max(hi - lo, 1)


def job(spec):
    """One (split, learner) job, or a split's control when label is None.
    CPU work runs in parallel across jobs; the device phase holds GPU_LOCK
    so one binary uses the GPU at a time."""
    root, binary, salt, label = spec
    t0, c0 = time.time(), time.process_time()
    parts = {"train": [], "val": [], "test": []}
    for path, data in repo_files(root):
        parts[X.split(path, salt)].append(data)
    sizes = {k: (len(v), sum(map(len, v))) for k, v in parts.items()}
    out = {"salt": salt, "label": label, "sizes": sizes, "dev": 0.0}
    if not all(parts.values()):
        out["skip"] = True
        return out
    enc = NgramEncoder(parts["train"])
    out["units"] = enc.n
    if label is None:
        counts = X.count_rows(enc, parts["train"])
        vparts = X.control_parts(counts, enc, parts["val"])
        val, lam, edge = X.fit_lam(lambda x: X.control_bpb(vparts, x))
        test = X.control_bpb(X.control_parts(counts, enc, parts["test"]),
                             lam)
        out["n_test"] = sum(len(d) - 1 for d in parts["test"])
    else:
        kind, opts = next((k, o) for lb, k, o in LEARNERS if lb == label)
        top = max(enc.orders)
        scored = positions(enc, parts["val"] + parts["test"], top)
        n_val = sum(len(d) - 1 for d in parts["val"])
        n_all = len(scored[3])
        stores = build(kind, opts, enc, parts["train"])
        inp, p = write_input(enc, stores, scored)
        del stores
        with GPU_LOCK:
            d0 = time.time()
            dnext, dpos, _note = execute(binary, inp, p)
            out["dev"] = time.time() - d0
        val, lam, edge = X.fit_lam(
            lambda x: bpb(dnext, dpos, 0, n_val, x))
        test = bpb(dnext, dpos, n_val, n_all, lam)
        out["n_test"] = n_all - n_val
    out.update(val=val, lam=lam, edge=edge, test=test,
               cpu=time.process_time() - c0, wall=time.time() - t0)
    return out


LAMS = [2.0 ** k for k in range(-12, 5)]
KINDS = {"branch": 0, "cls": 1, "control": 2}


def schedule(train, replay):
    """Training positions in learning order, as global offsets into the
    concatenated training bytes: (main store, online). With ``replay``
    each file's online positions are followed by the CLS replay batch,
    drawn with the same seeded calls as CLSLearner, which depend only on
    list lengths."""
    online, main = array("q"), array("q")
    rnd = random.Random(1)
    earlier = []
    start = 0
    for data in train:
        episodes = list(range(start, start + len(data) - 1))
        online.extend(episodes)
        main.extend(episodes)
        if replay and episodes:
            old = (rnd.sample(earlier, min(len(earlier), len(episodes)))
                   if earlier else [])
            batch = episodes + old
            rnd.shuffle(batch)
            main.extend(batch)
        earlier.extend(episodes)
        start += len(data)
    return main, online


def device_run(binary, enc, train, val, test, kind, opts):
    """Everything on the device (core/gpu/learn_device.cpp); returns
    (val bits per lam, test bits per lam, n_val, n_test)."""
    flags = ((1 if opts.get("metaplastic") else 0)
             | (2 if opts.get("hippo") else 0)
             | (4 if opts.get("cortex_meta") else 0))
    main, online = schedule(train, kind == "cls" and opts.get("replay"))

    def blob(files):
        starts = array("q", [0])
        for d in files:
            starts.append(starts[-1] + len(d))
        return b"".join(files), starts

    tr, tfs = blob(train)
    sc, sfs = blob(val + test)
    fd, inp = tempfile.mkstemp(suffix=".in")
    out = inp[:-3] + ".out"
    with os.fdopen(fd, "wb") as f:
        f.write(struct.pack("<iidiii", enc.n, len(enc.tables), RATE,
                            KINDS[kind], flags, max(enc.orders)))
        for k, _h, seed, off, size in enc.tables:
            f.write(struct.pack("<iIii", k, seed, off, size))
        f.write(struct.pack("<q", len(tr)) + tr)
        f.write(struct.pack("<i", len(train)) + tfs.tobytes())
        f.write(struct.pack("<q", len(main)) + main.tobytes())
        f.write(struct.pack("<q", len(online)) + online.tobytes())
        f.write(struct.pack("<q", len(sc)) + sc)
        f.write(struct.pack("<i", len(val) + len(test)) + sfs.tobytes())
        f.write(struct.pack("<i", len(val)))
        f.write(struct.pack("<i", len(LAMS)) + array("d", LAMS).tobytes())
    r = subprocess.run([binary, inp, out], capture_output=True, text=True)
    os.unlink(inp)
    if r.returncode != 0:
        raise SystemExit("FAIL: %s exited %d: %s" % (binary, r.returncode,
                                                     r.stderr.strip()))
    res = array("d")
    with open(out, "rb") as fh:
        raw = fh.read()
    os.unlink(out)
    nl = len(LAMS)
    res.frombytes(raw[:16 * nl])
    n_val, n_test = struct.unpack("<qq", raw[16 * nl:])
    return list(res[:nl]), list(res[nl:]), n_val, n_test


def device_job(spec):
    """One (split, learner) job with every per-position loop on the
    device; the CPU reads files, sizes the encoder and draws the replay
    order. label None is the control."""
    root, binary, salt, label = spec
    t0, c0 = time.time(), time.process_time()
    parts = {"train": [], "val": [], "test": []}
    for path, data in repo_files(root):
        parts[X.split(path, salt)].append(data)
    sizes = {k: (len(v), sum(map(len, v))) for k, v in parts.items()}
    out = {"salt": salt, "label": label, "sizes": sizes, "dev": 0.0}
    if not all(parts.values()):
        out["skip"] = True
        return out
    enc = NgramEncoder(parts["train"])
    out["units"] = enc.n
    if label is None:
        kind, opts = "control", {}
    else:
        kind, opts = next((k, o) for lb, k, o in LEARNERS if lb == label)
    with GPU_LOCK:
        d0 = time.time()
        vb, tb, _nv, nt = device_run(binary, enc, parts["train"],
                                     parts["val"], parts["test"], kind, opts)
        out["dev"] = time.time() - d0
    fits = list(zip(vb, LAMS))
    val, lam = min(fits)
    edge = lam in (fits[0][1], fits[-1][1])
    out.update(val=val, lam=lam, edge=edge, test=tb[LAMS.index(lam)],
               n_test=nt, cpu=time.process_time() - c0,
               wall=time.time() - t0)
    return out


GPU_LOCK = None


def _init(lock):
    global GPU_LOCK
    GPU_LOCK = lock


def main():
    args = sys.argv[1:]
    if len(args) < 2:
        print(__doc__)
        return 2
    root, binary, rest = args[0], args[1], args[2:]
    salts, only, engine = [""], None, "host"
    while rest:
        if rest[0] == "--engine" and len(rest) > 1:
            engine = rest[1]
        elif rest[0] == "--salts" and len(rest) > 1:
            salts = rest[1].split(",")
        elif rest[0] == "--only" and len(rest) > 1:
            only = rest[1].split(",")
        else:
            print("unknown argument: %s" % rest[0])
            return 2
        rest = rest[2:]
    learners = [x[0] for x in LEARNERS
                if only is None or any(o in x[0] for o in only)]
    if not learners:
        print("INVALID: --only matched no learner")
        return 2
    specs = [(root, binary, s, lb) for s in salts
             for lb in learners + [None]]
    procs = min(os.cpu_count() or 1, len(specs))
    if engine not in ("host", "device"):
        print("unknown engine: %s" % engine)
        return 2
    print("%d jobs on %d processes, engine %s, GPU shared behind a lock"
          % (len(specs), procs, engine))
    t0 = time.time()
    ctx = multiprocessing.get_context("fork")
    lock = ctx.Lock()
    with ctx.Pool(procs, initializer=_init, initargs=(lock,)) as pool:
        results = pool.map(device_job if engine == "device" else job,
                           specs, chunksize=1)
    wall = time.time() - t0
    runs, edge = [], False
    for salt in salts:
        rs = [r for r in results if r["salt"] == salt]
        print("\n=== split salt=%r ===" % salt)
        for k in ("train", "val", "test"):
            n, b = rs[0]["sizes"][k]
            print("%-5s %3d files %8d bytes" % (k, n, b))
        if any(r.get("skip") for r in rs):
            print("skipped: a split is empty")
            continue
        print("units: %d" % rs[0]["units"])
        ctrl = next(r for r in rs if r["label"] is None)
        res = {r["label"]: r for r in rs if r["label"] is not None}
        for lb in learners:
            r = res[lb]
            print("%-47s cpu %5.1f s  device %4.1f s  lam %-7g val %.6f%s"
                  % (lb, r["cpu"], r["dev"], r["lam"], r["val"],
                     "  AT GRID EDGE" if r["edge"] else ""))
        print("control lam fitted on validation: %g (val %.6f)"
              % (ctrl["lam"], ctrl["val"]))
        print("\nheld-out test, bits per byte (%d predictions):"
              % ctrl["n_test"])
        print("  %-47s %.6f" % ("control (count, rows sum to 1)",
                                ctrl["test"]))
        for lb in learners:
            print("  %-47s %.6f" % (lb, res[lb]["test"]))
        edge = edge or ctrl["edge"] or any(res[lb]["edge"] for lb in res)
        runs.append((salt, {lb: res[lb]["test"] for lb in learners},
                     ctrl["test"]))
    if not runs:
        print("INVALID: every split was empty")
        return 2
    print("\n=== learner minus control, test bits per byte ===")
    print("%-47s %s   mean    min    max" % ("", "  ".join(
        "%8s" % (s or "default") for s, _, _ in runs)))
    for lb in learners:
        d = [res[lb] - ctrl for _s, res, ctrl in runs]
        print("%-47s %s  %+.3f %+.3f %+.3f" % (
            lb, "  ".join("%+8.3f" % x for x in d), sum(d) / len(d),
            min(d), max(d)))
    cpu = sum(r.get("cpu", 0.0) for r in results)
    dev = sum(r["dev"] for r in results)
    print("\nresources: wall %.0f s, CPU %.0f core-s on %d processes, "
          "GPU device %.1f s. Energy is not readable here (no SMI power "
          "or RAPL under WSL2)." % (wall, cpu, procs, dev))
    return 2 if edge else 0


if __name__ == "__main__":
    raise SystemExit(main())
