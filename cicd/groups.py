"""List the named file sets and prove every path in them exists.

  groups.py MAKEFILE...

`make commit-verified GROUP=<name>` stages a named set instead of a
hand-typed list. A group is only worth having if it is TRUE, so this
reads the same Makefile the build reads -- not a second copy of the
lists, which would drift -- and reports any path that does not exist.

Exit 1 when a group names a missing file. A group that has quietly
stopped matching the tree is how a commit lands without the script it
calls, which is the failure the groups exist to prevent.

Standard library only.
"""

import os
import re
import sys

ASSIGN = re.compile(r"^GROUP_(\w+)\s*=\s*(.*)$")


MAKEFN = re.compile(r"\$\([^()]*(?:\([^()]*\)[^()]*)*\)")


def split_entries(rest):
    """Paths in a group line, with make functions kept whole.

    A naive split() breaks `$(wildcard doc/core/*.md)` into two tokens
    and then reports the second half as a missing file -- which it did,
    the first time this ran. A function call is one entry, expanded by
    make and not by us.
    """
    out = []
    for m in MAKEFN.finditer(rest):
        out.append(m.group(0))
    return out + MAKEFN.sub(" ", rest).split()


def read_groups(paths):
    """{name: [path, ...]} from GROUP_<name> assignments, honouring
    backslash continuations the way make does."""
    groups = {}
    for path in paths:
        if not os.path.exists(path):
            continue
        with open(path) as fh:
            lines = fh.read().splitlines()
        i = 0
        while i < len(lines):
            m = ASSIGN.match(lines[i])
            if not m:
                i += 1
                continue
            name, rest = m.group(1), m.group(2)
            while rest.endswith("\\") and i + 1 < len(lines):
                i += 1
                rest = rest[:-1] + " " + lines[i].strip()
            groups[name] = split_entries(rest)
            i += 1
    return groups


def main(argv):
    groups = read_groups(argv or ["Makefile"])
    if not groups:
        print("no GROUP_<name> assignments found")
        return 1
    bad = 0
    for name in sorted(groups):
        files = groups[name]
        # $(wildcard ...) and friends are resolved by make, not here;
        # reporting them as missing would be reporting on a string we
        # never expanded.
        unexpanded = [f for f in files if "$" in f]
        concrete = [f for f in files if f not in unexpanded]
        missing = [f for f in concrete if not os.path.exists(f)]
        bad += len(missing)
        mark = "MISSING" if missing else "ok"
        print("%-10s %-7s %d file(s)%s"
              % (name, mark, len(concrete),
                 ", %d expanded by make" % len(unexpanded)
                 if unexpanded else ""))
        for f in concrete:
            print("    %s %s" % ("!!" if f in missing else "  ", f))
    if bad:
        print()
        print("%d path(s) named by a group do not exist" % bad)
    return 1 if bad else 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
