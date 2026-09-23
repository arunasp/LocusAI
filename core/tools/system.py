"""Assemble the system from a values file and report what it permits.

  system.py [VALUES] [--cue NAME ...]      (default: core/values.json)

`make audit` reported Constitution, Dispatcher, DecisionTrace and
AssociativeGraph as constructed only by tests. This is the running path
that constructs them, so they stop being declared and start being
active. It prints what the file bought -- the cost of each declared
class, the categorical set, the incompatibilities -- then dispatches any
cues named on the command line and renders the audit trace, refusals
included.

Every path written is listed at the end. Nothing here writes to the
store on disk; the store it builds is in memory and discarded.
"""

import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                "..", "py"))

from locus.assemble import from_file            # noqa: E402
from locus.constitution import UNCLASSIFIED_REVERSIBILITY  # noqa: E402

DEFAULT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..",
                       "values.json")


def main(argv):
    path, cues = DEFAULT, []
    rest = list(argv)
    while rest:
        if rest[0] == "--cue" and len(rest) > 1:
            cues.append(rest[1])
            rest = rest[2:]
        elif rest[0].startswith("--"):
            print(__doc__)
            return 2
        else:
            path, rest = rest[0], rest[1:]

    system = from_file(path)
    print("values    %s" % os.path.relpath(path))
    print("costs     %d declared, an unnamed class costs %s"
          % (len(system.constitution.costs), UNCLASSIFIED_REVERSIBILITY))
    for name in sorted(system.constitution.costs):
        print("            %-14s %s" % (name,
                                        system.constitution.costs[name]))
    print("forbidden %s"
          % (", ".join(sorted(system.constitution.forbidden)) or "nothing"))
    for cls in sorted(system.graph.conflicts):
        print("conflict  %-14s not with %s"
              % (cls, ", ".join(sorted(system.graph.conflicts[cls]))))

    for cue in cues:
        out = system.dispatcher.act(cue)
        print("act       %-14s %s"
              % (cue, "REFUSED" if out.vetoed
                 else "%s via %s, stage %d"
                 % (out.action, out.source, int(out.stage))))

    print("trace     %d decisions, %d refused (%.0f%%)"
          % (len(system.trace), len(system.trace.vetoes()),
             100.0 * system.trace.rate()))
    for line in system.trace.render():
        print("            %s" % line)
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
