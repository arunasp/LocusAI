"""Build a knowledge store: train one learner on the device, keep it.

  know.py CORPUS BINARY OUT [--salt S] [--learner LABEL] [--plain]

Trains LABEL (default: CLS with a metaplastic neocortex) on the train
split of CORPUS with BINARY (learn_device), fits lam on validation, and
writes OUT (the sparse stores) and OUT.json (encoder tables, lam, figures,
provenance). The readout is coincidence-gated (each row scaled by its
agreement with the others; tests/exp_match_gate.py) unless --plain, and
the store records which was used. Every file written is listed with its
size.
"""

import json
import os
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "py"))
sys.path.insert(0, os.path.join(HERE, "..", "tests"))

import exp_learn_gpu as G  # noqa: E402
import exp_learn_stream as X  # noqa: E402
from locus.encode import NgramEncoder, repo_files  # noqa: E402

DEFAULT = "cls, neocortex also metaplastic (1/n)"


def main(argv):
    if len(argv) < 3:
        print(__doc__)
        return 2
    corpus, binary, out, rest = argv[0], argv[1], argv[2], argv[3:]
    salt, label, gated = "", DEFAULT, True
    while rest:
        if rest[0] == "--salt" and len(rest) > 1:
            salt, rest = rest[1], rest[2:]
        elif rest[0] == "--learner" and len(rest) > 1:
            label, rest = rest[1], rest[2:]
        elif rest[0] == "--plain":
            gated, rest = False, rest[1:]
        else:
            print("unknown argument: %s" % rest[0])
            return 2
    match = [(lb, k, o) for lb, k, o in G.LEARNERS if lb == label]
    if not match:
        print("unknown learner: %s" % label)
        return 2
    _lb, kind, opts = match[0]
    parts = {"train": [], "val": [], "test": []}
    for path, data in repo_files(corpus):
        parts[X.split(path, salt)].append(data)
    enc = NgramEncoder(parts["train"])
    os.makedirs(os.path.dirname(os.path.abspath(out)), exist_ok=True)
    t0 = time.time()
    vb, tb, n_val, n_test = G.device_run(binary, enc, parts["train"],
                                         parts["val"], parts["test"],
                                         kind, opts, know=out, gated=gated)
    wall = time.time() - t0
    val, lam = min(zip(vb, G.LAMS))
    meta = {"learner": label, "readout": "gated" if gated else "plain",
            "corpus": os.path.abspath(corpus),
            "salt": salt, "orders": list(enc.orders), "heads": enc.heads,
            "tables": [list(t) for t in enc.tables], "units": enc.n,
            "lam": lam, "val_bpb": val, "test_bpb": tb[G.LAMS.index(lam)],
            "n_val": n_val, "n_test": n_test,
            "train_files": len(parts["train"]),
            "train_bytes": sum(map(len, parts["train"])),
            "made": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())}
    with open(out + ".json", "w") as fh:
        json.dump(meta, fh, indent=1)
    print("learner   %s, readout %s" % (
        label, "gated" if gated else "plain"))
    print("train     %d files, %d bytes; %d units" % (
        meta["train_files"], meta["train_bytes"], enc.n))
    print("lam       %g (val %.6f, test %.6f bits per byte)" % (
        lam, val, meta["test_bpb"]))
    print("device    %.1f s" % wall)
    for p in (out, out + ".json"):
        print("wrote     %s  %d bytes" % (p, os.path.getsize(p)))
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
