"""Assemble the system from a values file.

The mechanisms in constitution.py, decisions.py, pathways.py and
graph.py were each built and tested, and nothing constructed them:
DECLARED, not ACTIVE (`make audit`). This is the path that constructs
them, and it is also what doc/core/CONSTITUTION.md's list calls INGEST.

Values arrive from a FILE, read once, at construction:

- costs         action class -> reversibility, 1.0 undoable to 0 not
- forbidden     the categorical set, which no evidence clears
- conflicts     class -> classes it may not be co-active or linked with
- classes       trace key -> its class

Nothing here can be changed afterwards. The Constitution holds its two
mappings read-only, the graph holds its two read-only, and neither has a
method to add or remove. Changing values means editing the file and
building again, which is a deliberate act at a level above the one being
constrained -- the property that makes this a constraint rather than a
preference.

One DecisionTrace is shared by the dispatcher and the graph, so refusals
from either, and pairs the graph only noticed, land in one audit
surface.
"""

import collections
import json

from locus.commit import Cascade
from locus.constitution import Constitution
from locus.decisions import DecisionTrace
from locus.graph import AssociativeGraph
from locus.pathways import Dispatcher
from locus.procedural import Consolidator
from locus.store import Store

KEYS = ("costs", "forbidden", "conflicts", "classes", "releasers")

System = collections.namedtuple(
    "System", "constitution dispatcher graph trace store values")


def load_values(path):
    """Read a values file, refusing anything it does not understand.

    An unknown key is an error rather than a warning: a values file with
    a typo'd `forbiden` list would otherwise load as permitting
    everything, and silently.
    """
    with open(path) as fh:
        values = json.load(fh)
    if not isinstance(values, dict):
        raise ValueError("%s: values must be an object" % path)
    unknown = sorted(set(values) - set(KEYS))
    if unknown:
        raise ValueError("%s: unknown key(s): %s. Known: %s"
                         % (path, ", ".join(unknown), ", ".join(KEYS)))
    return values


def build(values, store=None, cascade=None):
    """Construct the system. Every constraint is fixed here and now."""
    trace = DecisionTrace()
    constitution = Constitution(costs=values.get("costs", {}),
                                forbidden=values.get("forbidden", ()))
    store = store if store is not None else Store()
    dispatcher = Dispatcher(store, Consolidator(store),
                            cascade=cascade or Cascade(),
                            constitution=constitution, trace=trace)
    graph = AssociativeGraph(classes=values.get("classes", {}),
                             conflicts=values.get("conflicts", {}),
                             trace=trace)
    for cue, spec in (values.get("releasers") or {}).items():
        dispatcher.add_releaser(cue, spec["key"], spec["action"],
                                action_class=spec.get("class"))
    return System(constitution, dispatcher, graph, trace, store, values)


def from_file(path, **kw):
    return build(load_values(path), **kw)
