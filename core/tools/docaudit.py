"""Check the documents against the repository they describe.

  docaudit.py [--path P]        (default: the whole repository)

Docs drift silently. Nothing fails when a document names a file that was
renamed, a `make` target that no longer exists, or -- the costly one --
lists as NOT IMPLEMENTED a mechanism that was finished hours earlier,
which invites the work to be done twice.

Four checks, each mechanical and each reporting only what it can prove:

  PATHS     a path in backticks that looks like a repository file, and
            is not there
  TARGETS   a `make X` named in prose with no rule X in any Makefile
  SYMBOLS   a backticked identifier that looks like code from this
            project (locus_*, a .py module, Class.method) and is not
            defined anywhere in the source
  STALE     a "not implemented" claim whose subject IS now defined.
            A candidate, never a verdict: the claim may be about a
            different sense of the word, so each one is read by a human
            before it is believed.

It reports and never fails a build: most findings need a decision about
which side is wrong, and a check that edits prose on its own would be
the wrong instrument entirely.
"""

import os
import re
import sys

CORE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
REPO = os.path.dirname(CORE)
SKIP = ("/.git/", "/build/", "/.venv/", "/node_modules/", "/__pycache__/",
        "/.rocm-include/", "/hf/", "/data/")
DOCS = (".md",)
CODE = (".py", ".c", ".h", ".cpp")
TICKED = re.compile(r"`([^`\n]+)`")
MAKE = re.compile(r"`make ([a-z][a-z0-9-]*)[^`]*`")
RULE = re.compile(r"^([a-zA-Z][\w.-]*)\s*:(?!=)", re.M)
NOT_IMPL = re.compile(r"not (?:yet )?implemented", re.I)


def wanted(path):
    p = "/" + path.replace(os.sep, "/")
    return not any(s in p for s in SKIP)


def walk(root, exts):
    for d, _, fs in os.walk(root):
        for f in fs:
            p = os.path.join(d, f)
            if f.endswith(exts) and wanted(p):
                yield p


# Directories this repository owns. A path outside them belongs to
# somebody else -- an upstream source file, a system path, a package
# inside node_modules -- and this tool has nothing to say about it.
OURS = ("core/", "doc/", "tools/", "cicd/", "test/")
PLANNED = re.compile(r"to write|to be written|planned|does not exist yet",
                     re.I)


def looks_like_path(tok):
    if " " in tok or "<" in tok or tok.startswith(("http", "//", "~")):
        return False
    return "/" in tok and "." in tok.rsplit("/", 1)[-1]


def looks_like_symbol(tok):
    if " " in tok or "/" in tok:
        return False
    return (tok.startswith("locus_")
            or tok.endswith(".py")
            or bool(re.fullmatch(r"[A-Z]\w+\.\w+", tok or "")))


def main(argv):
    root = REPO
    if "--path" in argv:
        root = os.path.join(REPO, argv[argv.index("--path") + 1])

    source = {p: open(p, errors="replace").read()
              for p in walk(root, CODE)}
    blob = "\n".join(source.values())
    rules = set()
    for d, _, fs in os.walk(root):
        for f in fs:
            if f == "Makefile" and wanted(os.path.join(d, f)):
                rules |= set(RULE.findall(
                    open(os.path.join(d, f), errors="replace").read()))

    docs = sorted(walk(root, DOCS))
    bad_paths, bad_targets, bad_symbols, stale = [], [], [], []

    for doc in docs:
        text = open(doc, errors="replace").read()
        where = os.path.relpath(doc, REPO)
        for tok in TICKED.findall(text):
            tok = tok.strip()
            if looks_like_path(tok):
                cand = tok.lstrip("./")
                # Resolved against the repository, against core/, and
                # against the DOCUMENT'S OWN directory -- tools/README
                # naming `server/allowlist.txt` means the one beside it,
                # and reading that as missing was the tool's error, not
                # the document's.
                roots = (REPO, CORE, os.path.dirname(doc))
                if any(os.path.exists(os.path.join(r, cand))
                       for r in roots):
                    continue
                # Classified by the path AS WRITTEN. Resolving it
                # doc-relative first made `src/proxy.ts` in
                # tools/README -- a file in an upstream npm package --
                # look like tools/src/proxy.ts and therefore ours.
                if not cand.startswith(OURS):
                    continue      # somebody else's tree
                line = next((ln for ln in text.splitlines()
                             if tok in ln), "")
                if PLANNED.search(line):
                    continue      # named as not written yet, on purpose
                bad_paths.append((where, tok))
            elif looks_like_symbol(tok):
                name = tok.rstrip("()").split(".")[-1]
                if len(name) > 3 and name not in blob:
                    bad_symbols.append((where, tok))
        for target in MAKE.findall(text):
            if target not in rules:
                bad_targets.append((where, target))
        for n, line in enumerate(text.splitlines(), start=1):
            if not NOT_IMPL.search(line):
                continue
            for tok in TICKED.findall(line):
                name = tok.strip().rstrip("()").split(".")[-1]
                if len(name) > 3 and name in blob:
                    stale.append((where, n, tok))

    print("documents %d, source files %d, make rules %d\n"
          % (len(docs), len(source), len(rules)))
    for title, rows in (("PATHS that do not exist", bad_paths),
                        ("MAKE targets with no rule", bad_targets),
                        ("SYMBOLS not found in the source", bad_symbols)):
        print("%s: %d" % (title, len(rows)))
        for where, tok in sorted(set(rows)):
            print("  %-34s %s" % (where, tok))
        print()
    print("STALE candidates -- a not-implemented claim whose subject "
          "exists: %d" % len(stale))
    print("  (candidates only: read each before believing it)")
    for where, n, tok in sorted(set(stale)):
        print("  %-30s line %-5d %s" % (where, n, tok))
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
