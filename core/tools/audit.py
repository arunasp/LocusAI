"""Two audits the standing rules ask for, across the whole repository.

  audit.py [--constants] [--inert] [--path P] [--all]
                                        (default: both, whole repo)

CONSTANTS. doc/core/ROADMAP.md's standing constraints say no static
value where biology has dynamics, and that any value which cannot yet be
derived is recorded as initial state with its open question. This finds
numeric constants and numeric defaults in Python, and `#define` and
`static const` values in C and C++, and marks the ones with NO nearby
note saying so. A number without that note is a design decision nobody
declared.

INERT CODE. A mechanism that exists but nothing constructs is DECLARED,
not ACTIVE -- the distinction this project keeps rediscovering. This
lists public definitions no non-test file references (Python functions
and classes, C functions declared in a header), and separately the
classes built only by tests, which is the stronger form: a test that
constructs a thing proves it works, not that anything uses it.

Scanned: every .py, .c, .h, .cpp under the repository except build
output, virtualenvs, caches and VENDORED headers -- a third party's
constants are not this project's design decisions. `--all` includes them
anyway; `--path P` restricts everything to one subtree.

Reports only, and never fails a build: the right response to most
findings is a decision, not a fix.
"""

import ast
import collections
import json
import os
import re
import sys

CORE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
REPO = os.path.dirname(CORE)
CODE = (".py", ".c", ".h", ".cpp")
SKIP = ("/build/", "/.venv/", "/__pycache__/", "/.git/", "/node_modules/",
        "/.rocm-include/", "/mock/", "/hf/", "/data/")
# A note DECLARES a value when it says what the value IS -- initial
# state, an open question, measured, derived, a bound. "default" was in
# this list and had to come out: almost every docstring mentioning a
# default satisfied it, so "(default 64)" counted as a declaration when
# it only documents the number rather than saying anything about where
# it came from or what would settle it.
# "neutral" earns its place beside the others: a value that is the
# identity of its operation -- a rate of 1.0 that scales nothing, a
# threshold of 0.0 that excludes nothing, an option off unless asked --
# is not a tuned number at all, and saying so is as complete an answer
# as "measured" or "a bound". It was used as a category in the substrate
# pass before it was a flag here, which is why that pass had to lean on
# other words in the same note.
FLAGS = ("initial", "open question", "not tuned", "placeholder", "until",
         "for now", "arbitrary", "measured", "derived", "neutral",
         "chosen so", "bound", "limit")
DEFINE = re.compile(r"^\s*#define\s+([A-Za-z_]\w*)\s+"
                    r"\(?(-?\d+\.?\d*(?:e-?\d+)?)\)?\s*(?:/\*|//|$)")
STATIC = re.compile(r"^\s*static\s+const\s+\w[\w ]*\s+([A-Za-z_]\w*)\s*=\s*"
                    r"\(?(-?\d+\.?\d*(?:e-?\d+)?)")
CFUNC = re.compile(r"^\w[\w \*]*\s\*?([a-z_][a-z0-9_]*)\s*\([^;]*\)\s*$")


def wanted(path, include_vendor=False):
    p = "/" + path.replace(os.sep, "/")
    if not path.endswith(CODE):
        return False
    return include_vendor or not any(s in p for s in SKIP)


def sources(root, include_vendor=False):
    out = {}
    for d, _, fs in os.walk(root):
        for f in fs:
            p = os.path.join(d, f)
            if wanted(p, include_vendor):
                out[p] = open(p, errors="replace").read()
    return out


def note(lines, lineno, extra=""):
    """The comment that BELONGS to the value on `lineno`.

    Attribution used to be proximity -- any of the five lines above --
    and proximity is not attachment. A note written for one constant
    silently declared a DIFFERENT one five lines away: adding a comment
    to `PROGRESS_SECONDS` in tools/hf_data.py marked `_request(tries=5)`
    as declared and dropped the undeclared count by one for no reason.
    A ratchet whose floor moves by accident is worse than no ratchet.

    So: the trailing comment on the value's own line, plus the
    CONTIGUOUS comment block immediately above it -- no blank line, no
    code line, in between -- plus `extra` for a function's own
    docstring, which is where a default's reasoning is usually written.
    """
    text = [extra, lines[lineno - 1]]
    i = lineno - 2
    while i >= 0:
        stripped = lines[i].strip()
        if stripped.startswith("#") or stripped.startswith("//"):
            text.append(stripped)
        elif stripped.startswith("*") or stripped.startswith("/*"):
            text.append(stripped)
            if stripped.startswith("/*"):
                break
        else:
            break
        i -= 1
    return " ".join(text).lower()


def py_constants(path, src):
    rows, lines = [], src.splitlines()
    try:
        tree = ast.parse(src)
    except SyntaxError:
        return rows
    for n in ast.walk(tree):
        if (isinstance(n, ast.Assign) and n.col_offset == 0
                and isinstance(n.targets[0], ast.Name)
                and isinstance(n.value, ast.Constant)
                and isinstance(n.value.value, (int, float))
                and not isinstance(n.value.value, bool)):
            rows.append((path, n.lineno, n.targets[0].id, n.value.value,
                         any(f in note(lines, n.lineno) for f in FLAGS)))
        if isinstance(n, ast.FunctionDef) and n.args.defaults:
            args = n.args.args[-len(n.args.defaults):]
            for a, dv in zip(args, n.args.defaults):
                if (isinstance(dv, ast.Constant)
                        and isinstance(dv.value, (int, float))
                        and not isinstance(dv.value, bool)):
                    doc = ast.get_docstring(n) or ""
                    rows.append((path, n.lineno,
                                 "%s(%s)" % (n.name, a.arg), dv.value,
                                 any(f in note(lines, n.lineno, doc)
                                     for f in FLAGS)))
    return rows


def c_constants(path, src):
    rows, lines = [], src.splitlines()
    for i, line in enumerate(lines, start=1):
        for pat in (DEFINE, STATIC):
            m = pat.match(line)
            if m:
                rows.append((path, i, m.group(1), m.group(2),
                             any(f in note(lines, i) for f in FLAGS)))
    return rows


def constants(files):
    rows = []
    for p, src in sorted(files.items()):
        rows += (py_constants(p, src) if p.endswith(".py")
                 else c_constants(p, src))
    return rows


def references(path, src):
    """Names this file actually USES, as code.

    Counting raw substrings made a name MENTIONED IN A COMMENT read as a
    caller, which silently cleared an inert definition -- a false "used"
    is worse than a false "unused", because nobody goes looking for it.
    For Python the names come from the parse tree; for C and C++, from
    the source with comments and string literals stripped.
    """
    names = collections.Counter()
    if path.endswith(".py"):
        try:
            tree = ast.parse(src)
        except SyntaxError:
            return names
        for n in ast.walk(tree):
            if isinstance(n, ast.Name):
                names[n.id] += 1
            elif isinstance(n, ast.Attribute):
                names[n.attr] += 1
            elif isinstance(n, (ast.FunctionDef, ast.ClassDef)):
                for d in n.decorator_list:
                    # A decorated definition is REGISTERED by that
                    # decorator -- an MCP tool, a handler -- so the name
                    # having no textual caller says nothing.
                    names["@" + n.name] += 1
            elif isinstance(n, ast.keyword) and n.arg:
                names[n.arg] += 1
    else:
        code = re.sub(r"/\*.*?\*/", " ", src, flags=re.S)
        code = re.sub(r"//[^\n]*", " ", code)
        code = re.sub(r'"(?:[^"\\]|\\.)*"', ' "" ', code)
        for word in re.findall(r"[A-Za-z_]\w*", code):
            names[word] += 1
    return names


def definitions(files):
    """(path, name, line, is_class) for things another file could call.

    Only from the library and the tools. A test file's own functions are
    not surface anyone is meant to call, and counting them made 458 of
    804 definitions look "referenced only by tests" -- true, and
    meaningless. Tests still count as REFERENCES, which is the whole
    point of separating declared from active.
    """
    out = []
    for p, src in files.items():
        if "/tests/" in p.replace(os.sep, "/"):
            continue
        if p.endswith(".py"):
            try:
                tree = ast.parse(src)
            except SyntaxError:
                continue
            for n in ast.walk(tree):
                if (isinstance(n, (ast.FunctionDef, ast.ClassDef))
                        and not n.name.startswith("_")):
                    out.append((p, n.name, n.lineno,
                                isinstance(n, ast.ClassDef)))
        elif p.endswith(".c"):  # headers declare, .c defines
            for i, line in enumerate(src.splitlines(), start=1):
                m = CFUNC.match(line)
                if m and not line.startswith("static"):
                    out.append((p, m.group(1), i, False))
    return out


def report(files, want_c, want_i):
    if want_c:
        rows = constants(files)
        bare = [r for r in rows if not r[4]]
        print("CONSTANTS: %d numeric values in %d files, %d with no note "
              "calling them initial state or open"
              % (len(rows), len(files), len(bare)))
        for p, line, name, value, _f in bare:
            print("  %-46s %-30s %s"
                  % (os.path.relpath(p, REPO) + ":" + str(line), name,
                     value))
        print()

    if want_i:
        defs = definitions(files)
        used = {p: references(p, src) for p, src in files.items()}
        print("INERT: %d public definitions" % len(defs))
        never, testonly, classes = [], [], []
        seen = set()
        for p, name, line, is_class in defs:
            if (p, name) in seen:
                continue      # one row per definition, not per parse hit
            seen.add((p, name))
            prod = sum(u[name] for q, u in used.items()
                       if q != p and "/tests/" not in q)
            test = sum(u[name] for q, u in used.items() if "/tests/" in q)
            registered = used[p]["@" + name]
            # A helper used only inside its own module is not inert; it
            # just is not surface. Without this the tool called its own
            # internals dead.
            own = used[p][name]
            where = os.path.relpath(p, REPO)
            if prod == 0 and test == 0 and own <= 0 and not registered:
                never.append((where, name, line))
            elif prod == 0 and test > 0:
                testonly.append((where, name, test))
            # prod == 0 with no test reference and own > 0 is an
            # internal helper: not surface, not inert, not reported.
            elif is_class and prod <= 1 and test:
                classes.append((where, name, test))
        # Deduplicate what is REPORTED, so the count and the rows agree
        # whatever the parse produced; two rows for one definition would
        # make the reader doubt the rest of the report.
        never = sorted(set(never))
        testonly = sorted(set(testonly))
        classes = sorted(set(classes))
        print("  never referenced: %d, referenced only by tests: %d"
              % (len(never), len(testonly)))
        for where, name, line in never:
            print("  never used   %-40s %s:%d" % (name, where, line))
        for where, name, test in testonly:
            print("  tests only   %-40s %s (%d)" % (name, where, test))
        print()
        print("DECLARED, NOT ACTIVE: classes built only by tests")
        print("  (a hint, not a verdict: a class can appear here because "
              "it is RETURNED or RAISED rather than named)")
        for where, name, test in classes:
            print("  %-40s %s" % (name, where))


BASELINE = os.path.join(REPO, "core", "audit-baseline.json")


def ratchet(files, write=False):
    """Fail when the count of UNDECLARED numeric values goes UP.

    doc/core/ROADMAP.md's standing constraint is no static value where
    biology has dynamics, and any value that cannot yet be derived is
    recorded as initial state with its open question. That rule has been
    stated in three places and restated in most sessions, and the count
    still grows -- `LOCUS_SEQ_HISTORY = 32` was added the same day the
    rule was quoted back. A rule nobody can fail to notice is not the
    same as a rule that cannot be broken.

    So this is a RATCHET, not a target: the existing values are a debt
    to be decided one at a time, and the only thing refused is ADDING to
    it. Declaring a value (a nearby note calling it initial state, an
    open question, measured, derived, a bound) lowers the number, and a
    lowered number is written back so the floor follows the work down.
    """
    rows = constants(files)
    bare = [r for r in rows if not r[4]]
    now = len(bare)
    base = now
    if os.path.exists(BASELINE):
        with open(BASELINE) as fh:
            base = json.load(fh).get("undeclared", now)
    if write or now < base:
        with open(BASELINE, "w") as fh:
            json.dump({"undeclared": now,
                       "note": "Undeclared numeric values. May fall, "
                               "never rise: see ratchet() in "
                               "tools/audit.py."}, fh, indent=2)
            fh.write("\n")
        print("audit-ratchet: baseline now %d (was %d)" % (now, base))
        return 0
    if now > base:
        print("audit-ratchet: FAIL -- %d undeclared numeric values, "
              "baseline %d." % (now, base))
        print("  A new static value needs a note saying what it is: "
              "initial state, an open question, measured, derived, or "
              "a bound. Or derive it and add none.")
        for where, line, name, value, _f in sorted(set(bare))[:0]:
            pass
        return 1
    print("audit-ratchet: %d undeclared, baseline %d -- ok" % (now, base))
    return 0


def main(argv):
    want_c = "--constants" in argv or not set(argv) & {"--constants",
                                                       "--inert"}
    want_i = "--inert" in argv or not set(argv) & {"--constants", "--inert"}
    root = REPO
    if "--path" in argv:
        root = os.path.join(REPO, argv[argv.index("--path") + 1])
    files = sources(root, include_vendor="--all" in argv)
    if "--ratchet" in argv:
        return ratchet(files, write="--accept" in argv)
    print("scanning %s (%d files)\n" % (os.path.relpath(root, REPO) or ".",
                                        len(files)))
    report(files, want_c, want_i)
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
