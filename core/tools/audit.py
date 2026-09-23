"""Two audits the standing rules ask for, run against the library.

  audit.py [--constants] [--inert]      (default: both)

CONSTANTS. doc/core/ROADMAP.md's standing constraints say no static
value where biology has dynamics, and that any value which cannot yet be
derived is recorded as initial state with its open question. This lists
every module-level numeric constant and numeric default in
core/py/locus, and marks the ones with NO nearby note saying so. A
number without that note is a design decision nobody declared.

INERT CODE. A mechanism that exists but nothing constructs is DECLARED,
not ACTIVE -- the distinction this project keeps rediscovering. This
lists public definitions no non-test file references, and separately the
classes no running path builds, which is the stronger form: tests that
construct a thing prove it works, not that anything uses it.

Reports only. Nothing here fails a build, because the right response to
most findings is a decision, not a fix.
"""

import ast
import os
import sys

# Resolved from this file, not from the working directory: run from
# core/ the relative form scanned nothing and every definition looked
# unused, which is a scan that lies rather than one that finds nothing.
CORE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
LIB = os.path.join(CORE, "py", "locus")
ROOTS = tuple(os.path.join(CORE, d) for d in ("py", "tools", "tests"))
FLAGS = ("initial", "open question", "not tuned", "placeholder", "until",
         "for now", "arbitrary", "default", "measured", "derived")


def sources(roots):
    out = {}
    for root in roots:
        for d, _, fs in os.walk(root):
            if "__pycache__" in d:
                continue
            for f in fs:
                if f.endswith(".py"):
                    p = os.path.join(d, f)
                    out[p] = open(p, errors="replace").read()
    return out


def near(lines, lineno, span=4):
    lo = max(0, lineno - span - 1)
    return " ".join(lines[lo:lineno + 1]).lower()


def constants(lib):
    rows = []
    for p in sorted(lib):
        src = lib[p]
        lines = src.splitlines()
        for n in ast.walk(ast.parse(src)):
            if (isinstance(n, ast.Assign) and n.col_offset == 0
                    and isinstance(n.targets[0], ast.Name)
                    and isinstance(n.value, ast.Constant)
                    and isinstance(n.value.value, (int, float))
                    and not isinstance(n.value.value, bool)):
                rows.append((p, n.lineno, n.targets[0].id, n.value.value,
                             any(f in near(lines, n.lineno) for f in FLAGS)))
            if isinstance(n, ast.FunctionDef) and n.args.defaults:
                args = n.args.args[-len(n.args.defaults):]
                for a, dv in zip(args, n.args.defaults):
                    if (isinstance(dv, ast.Constant)
                            and isinstance(dv.value, (int, float))
                            and not isinstance(dv.value, bool)):
                        rows.append(
                            (p, n.lineno, "%s(%s)" % (n.name, a.arg),
                             dv.value,
                             any(f in near(lines, n.lineno, 8)
                                 for f in FLAGS)))
    return rows


def inert(lib, everything):
    defs = []
    for p, src in lib.items():
        for n in ast.walk(ast.parse(src)):
            if (isinstance(n, (ast.FunctionDef, ast.ClassDef))
                    and not n.name.startswith("_")):
                defs.append((p, n.name, n.lineno,
                             isinstance(n, ast.ClassDef)))
    rows = []
    for p, name, line, is_class in defs:
        prod = sum(s.count(name) for q, s in everything.items()
                   if q != p and "/tests/" not in q)
        test = sum(s.count(name) for q, s in everything.items()
                   if "/tests/" in q)
        rows.append((os.path.basename(p), name, line, is_class, prod, test))
    return rows


def main(argv):
    want_c = "--constants" in argv or not argv
    want_i = "--inert" in argv or not argv
    lib = sources([LIB])
    everything = sources(ROOTS)

    if want_c:
        rows = constants(lib)
        bare = [r for r in rows if not r[4]]
        print("CONSTANTS: %d numeric values, %d with no note calling them "
              "initial state or open" % (len(rows), len(bare)))
        for p, line, name, value, _flag in sorted(bare, key=lambda r: r[0]):
            print("  %-16s %-30s %-12s %s:%d"
                  % (os.path.basename(p), name, value,
                     os.path.basename(p), line))
        print()

    if want_i:
        rows = inert(lib, everything)
        never = [r for r in rows if r[4] == 0 and r[5] == 0]
        testonly = [r for r in rows if r[4] == 0 and r[5] > 0]
        classes = [r for r in rows if r[3] and r[4] <= 1]
        print("INERT: %d public definitions, %d referenced nowhere, %d only "
              "by tests" % (len(rows), len(never), len(testonly)))
        for f, name, line, _c, _p, _t in sorted(never):
            print("  never used      %-16s %-26s %s:%d" % (f, name, f, line))
        for f, name, line, _c, _p, t in sorted(testonly):
            print("  tests only      %-16s %-26s %d references" % (f, name, t))
        print()
        print("DECLARED, NOT ACTIVE: classes no running path constructs")
        print("  (a hint, not a verdict: a class can appear here because "
              "it is RETURNED or RAISED rather than named)")
        for f, name, line, _c, p, t in sorted(classes):
            if t:
                print("  %-16s %-26s built only in tests" % (f, name))
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
