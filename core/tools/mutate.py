"""Break a stated design property on purpose and require a red test.

  mutate.py [--only SUBSTRING] [--list]

A green suite says the tests pass. It does not say they would CATCH the
property being broken -- and this repository has already shipped one
mechanism (the refresh that carries learning into the dynamics) whose
removal left every test green. So each claim below is a sentence from
the design documents, paired with an edit that makes it false and the
tests that should notice.

CAUGHT means the mutation turned a test red: the claim is held by
something. SURVIVED means the code can lose that property silently, and
the answer is a test, not a better suite average.

Every mutation is applied to a file, run, and REVERTED in a finally
block, with the original content compared byte for byte afterwards.
A mutation left behind would be far worse than the gap it measures.
"""

import hashlib
import os
import subprocess
import sys

CORE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PY = os.path.join(CORE, ".venv", "bin", "python3")
TESTS = os.path.join(CORE, "tests")

# (claim, file, old, new, test selection)
MUTATIONS = [
    ("Dispatcher.fires is exact membership -- no partial match",
     "py/locus/pathways.py",
     "        return cue in self._releasers\n",
     "        return True\n",
     ["test_locus", "test_constitution_wiring"]),

    ("commit_threshold scales inversely with reversibility",
     "py/locus/commit.py",
     "        return self.commit / max(reversibility, REVERSIBILITY_FLOOR)",
     "        return self.commit",
     ["test_locus", "test_constitution_wiring"]),

    ("required_signals scales inversely with reversibility",
     "py/locus/commit.py",
     "        scale = 1.0 / max(reversibility, REVERSIBILITY_FLOOR)",
     "        scale = 1.0",
     ["test_locus", "test_constitution_wiring"]),

    ("Consolidator.cost decays geometrically with confirmed reuse",
     "py/locus/procedural.py",
     "        return max(self.floor, DELIBERATE_COST * "
     "(self.discount ** reps))",
     "        return DELIBERATE_COST",
     ["test_locus", "test_constitution_wiring"]),

    ("spread decays per hop rather than reaching flat",
     "py/locus/graph.py",
     "        decay = self.decay if decay is None else decay",
     "        decay = 1.0",
     ["test_locus", "test_graph_constraint"]),

    ("prefetch_set excludes the origins it was seeded with",
     "py/locus/graph.py",
     "             if k not in seeds),",
     "             if True),",
     ["test_locus"]),

    ("NoveltyGate refuses to encode a familiar event",
     "py/locus/episodic.py",
     "        if error < self.threshold:\n            return False, 0.0",
     "        if False:\n            return False, 0.0",
     ["test_locus"]),

    ("a forbidden act is refused at any evidence level",
     "py/locus/constitution.py",
     "    def permits(self, action_class):",
     "    def permits(self, action_class):\n        return True",
     ["test_locus", "test_constitution_wiring", "test_assemble"]),

    ("an unnamed action class costs the MOST, not the least",
     "py/locus/constitution.py",
     "UNCLASSIFIED_REVERSIBILITY = 0.05",
     "UNCLASSIFIED_REVERSIBILITY = 1.0",
     ["test_locus", "test_constitution_wiring"]),

    ("the audited layer is handed a COPY of the trace",
     "py/locus/decisions.py",
     "        return tuple(self._records)",
     "        return self._records",
     ["test_constitution_wiring"]),

    ("a values file with an unknown key is refused, not ignored",
     "py/locus/assemble.py",
     "    if unknown:",
     "    if False:",
     ["test_assemble"]),

    # The zero-modulator guard USED to survive its own deletion, because
    # dw = rate * tag * 0.0 is zero anyway. A property resting on a
    # multiplication is not enforced, so the branch is now bound by a
    # test that watches whether the tags are walked at all.
    ("three-factor: the modulator carries magnitude and sign",
     "py/locus/plasticity.py",
     "            dw = self.rate * tag * modulator",
     "            dw = self.rate * tag",
     ["test_plasticity", "test_cycle"]),

    ("a zero modulator does not even walk the tags",
     "py/locus/plasticity.py",
     "        if modulator == 0.0:\n            return 0",
     "        if False:\n            return 0",
     ["test_plasticity"]),

    ("a non-finite modulator is refused, not propagated",
     "py/locus/plasticity.py",
     "        if modulator != modulator or modulator in (INF, -INF):",
     "        if False:",
     ["test_plasticity"]),

    ("promotion needs a winner set that RECURRED (C)",
     "src/store.c",
     "            && signature_recurred(s, active_signature(s)))",
     "            && 1)",
     ["c"]),

    # Mutating only the direct-direction test leaves the SYMMETRIC loop
    # below it, which still catches a declared pair -- so the choke point
    # is the function's own answer.
    ("incompatible classes may not be co-active, room or not (C)",
     "src/store.c",
     "    if (klass >= LOCUS_CLASS_MAX)\n        return 0;",
     "    if (1)\n        return 0;",
     ["c"]),

    ("Field refuses a parameter that is not a finite, non-negative rate",
     "py/locus/field.py",
     "            if value < 0.0:",
     "            if False:",
     ["test_field"]),

    ("Field refuses a non-finite injected amount",
     "py/locus/field.py",
     "        if amount != amount or amount in (INF, -INF):",
     "        if False:",
     ["test_field"]),

    ("a non-positive or non-finite lam is refused at the readout",
     "py/locus/learn.py",
     "    if lam != lam or lam in (INF, -INF) or lam <= 0.0:",
     "    if False:",
     ["test_learn"]),

    ("the uniform mass is spread over exactly BYTE_UNITS candidates",
     "py/locus/learn.py",
     "    return (max(drive.get(b, 0.0), 0.0) + lam / BYTE_UNITS) "
     "/ (pos + lam)",
     "    return (max(drive.get(b, 0.0), 0.0) + lam / 255) / (pos + lam)",
     ["test_learn"]),

    ("the constitution's costs cannot be reached from outside",
     "py/locus/constitution.py",
     "        self._costs = types.MappingProxyType(costs)",
     "        self._costs = dict(costs)",
     ["test_locus", "test_constitution_wiring", "test_assemble"]),
]


def run(tests):
    """True when the selection is GREEN. "c" runs the C suite, which
    has to be rebuilt first or it tests the previous binary -- a
    mutation of a .c file is invisible until it is compiled."""
    if tests == ["c"]:
        build = subprocess.run(["make", "build"], cwd=CORE,
                               capture_output=True, text=True)
        if build.returncode != 0:
            return False      # a mutation that will not compile is caught
        r = subprocess.run([os.path.join(CORE, "build", "test_store")],
                           cwd=CORE, capture_output=True, text=True)
        return r.returncode == 0
    r = subprocess.run([PY, "-m", "unittest"] + tests,
                       cwd=TESTS, capture_output=True, text=True)
    return r.returncode == 0


def main(argv):
    only = None
    if "--only" in argv:
        only = argv[argv.index("--only") + 1]
    if "--list" in argv:
        for claim, path, _o, _n, _t in MUTATIONS:
            print("  %-58s %s" % (claim, path))
        return 0

    survived, caught = [], []
    for claim, rel, old, new, tests in MUTATIONS:
        if only and only not in claim and only not in rel:
            continue
        path = os.path.join(CORE, rel)
        original = open(path).read()
        digest = hashlib.sha256(original.encode()).hexdigest()
        if original.count(old) != 1:
            print("STALE  %s\n       the mutation no longer matches %s -- "
                  "the code moved, so fix the table" % (claim, rel))
            survived.append(claim)
            continue
        try:
            with open(path, "w") as fh:
                fh.write(original.replace(old, new))
            green = run(tests)
        finally:
            with open(path, "w") as fh:
                fh.write(original)
        back = hashlib.sha256(open(path).read().encode()).hexdigest()
        if back != digest:
            print("ABORT: %s was not restored" % rel)
            return 2
        if green:
            survived.append(claim)
            print("SURVIVED  %s\n          (%s, tests: %s)"
                  % (claim, rel, ", ".join(tests)))
        else:
            caught.append(claim)
            print("caught    %s" % claim)

    print("\n%d caught, %d SURVIVED" % (len(caught), len(survived)))
    for claim in survived:
        print("  survived: %s" % claim)
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
