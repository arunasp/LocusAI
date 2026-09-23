"""ctypes binding to the C trace store.

Deliberately ctypes rather than cffi/pybind: no build-time dependency and
no generated glue, which keeps the substrate frugal.
"""

import ctypes
import enum
import os


# Truth-value safeguards. All three of these enums have a 0-valued member,
# which as a plain IntEnum tests False -- and in every case 0 is the
# *ordinary* outcome, so the natural-looking `if result:` reads exactly
# backwards. Rather than renumber (0-is-success is the right C convention on
# the other side of the binding), each enum states its own truth semantics:
# where a boolean reading genuinely exists it is defined correctly, and where
# none exists the operation raises instead of quietly lying.


class Restabilize(enum.IntEnum):
    """Outcome of locus_restabilize().

    WRITTEN is 0 to match the C return convention, so it is falsy as an
    integer. __bool__ is therefore defined explicitly: the useful boolean
    question is "did the update land", not "is the value non-zero".
    """

    WRITTEN = 0
    GATED = 1
    ERROR = -1

    def __bool__(self):
        """True only when the update was actually written."""
        return self is Restabilize.WRITTEN


class Tier(enum.IntEnum):
    ACTIVE = 0
    EPISODIC = 1
    ARCHIVE = 2

    def __bool__(self):
        raise TypeError(
            "Tier has no truth value: ACTIVE is 0 and would test False, "
            "which is indistinguishable from the None returned for a "
            "missing trace. Compare explicitly (tier is Tier.ACTIVE), or "
            "test `is None` for absence."
        )


class Pathway(enum.IntEnum):
    DECLARATIVE = 0
    PROCEDURAL = 1
    INSTINCT = 2

    def __bool__(self):
        raise TypeError(
            "Pathway has no truth value: DECLARATIVE is 0 and would test "
            "False, which is indistinguishable from the None returned for "
            "a missing trace. Compare explicitly (path is "
            "Pathway.DECLARATIVE), or test `is None` for absence."
        )


class _Config(ctypes.Structure):
    _fields_ = [
        ("capacity", ctypes.c_int),
        ("active_k", ctypes.c_int),
        ("decay", ctypes.c_double),
        ("capture_floor", ctypes.c_double),
        ("capture_gain", ctypes.c_double),
        ("capture_window", ctypes.c_int),
        ("promote_after", ctypes.c_int),
        ("lease_max_ticks", ctypes.c_int),
        ("kwta_temp", ctypes.c_double),
        ("beta", ctypes.c_double),
        ("surprise_floor", ctypes.c_double),
        ("gate_tonic", ctypes.c_double),
        ("gate_threshold", ctypes.c_double),
        ("gate_decay", ctypes.c_double),
        ("seed", ctypes.c_uint32),
        # Mirrors LocusConfig exactly, including the trailing conflicts
        # table. _check_config_layout below verifies that at load: a
        # mismatch here is silent memory corruption, not an error.
        ("conflicts", ctypes.c_uint32 * 32),
    ]


class _View(ctypes.Structure):
    _fields_ = [
        ("key", ctypes.c_uint64),
        ("data", ctypes.c_void_p),
        ("len", ctypes.c_size_t),
        ("tier", ctypes.c_int),
        ("labile", ctypes.c_int),
        ("generation", ctypes.c_uint32),
        ("slot", ctypes.c_void_p),
    ]


class _Stats(ctypes.Structure):
    _fields_ = [
        ("requests", ctypes.c_uint64),
        ("hits", ctypes.c_uint64),
        ("misses", ctypes.c_uint64),
        ("prefetched", ctypes.c_uint64),
        ("prefetch_hits", ctypes.c_uint64),
        ("promotions", ctypes.c_uint64),
        ("evictions", ctypes.c_uint64),
        ("captures", ctypes.c_uint64),
        ("labile_lost", ctypes.c_uint64),
        ("leases_revoked", ctypes.c_uint64),
        ("updates_gated", ctypes.c_uint64),
        ("active_traces", ctypes.c_uint64),
        ("resident_traces", ctypes.c_uint64),
    ]


def _default_lib_path():
    here = os.path.dirname(os.path.abspath(__file__))
    return os.path.join(here, "..", "..", "build", "liblocus.so")


class LayoutMismatch(RuntimeError):
    """The ctypes mirror disagrees with the C struct it stands for."""


def _check_config_layout(lib, mirror=None):
    """Compare this module's _Config with the library's LocusConfig.

    Checked rather than trusted, because the failure it prevents is not an
    exception: a mirror one field short reads a neighbouring field's bytes
    and the store runs on values nobody set. Size catches a missing or
    added field, offsets catch reordering and padding.
    """
    mirror = mirror or _Config
    lib.locus_config_size.restype = ctypes.c_size_t
    lib.locus_config_layout.argtypes = [
        ctypes.POINTER(ctypes.c_size_t), ctypes.c_int]
    lib.locus_config_layout.restype = ctypes.c_int
    want_size = int(lib.locus_config_size())
    if ctypes.sizeof(mirror) != want_size:
        raise LayoutMismatch(
            "LocusConfig is %d bytes, this mirror is %d: a field was added, "
            "removed or resized without updating %s"
            % (want_size, ctypes.sizeof(mirror), __file__))
    n = lib.locus_config_layout(None, 0)
    buf = (ctypes.c_size_t * n)()
    lib.locus_config_layout(buf, n)
    names = [f[0] for f in mirror._fields_]
    if len(names) != n:
        raise LayoutMismatch(
            "LocusConfig has %d fields, this mirror has %d" % (n, len(names)))
    for name, want in zip(names, buf):
        got = getattr(mirror, name).offset
        if got != want:
            raise LayoutMismatch(
                "field %r sits at byte %d in LocusConfig and %d in this "
                "mirror" % (name, want, got))


def _bind(path):
    lib = ctypes.CDLL(os.path.abspath(path))
    lib.locus_config_default.restype = _Config
    lib.locus_store_create.argtypes = [ctypes.POINTER(_Config)]
    lib.locus_store_create.restype = ctypes.c_void_p
    lib.locus_store_destroy.argtypes = [ctypes.c_void_p]
    lib.locus_put.argtypes = [
        ctypes.c_void_p, ctypes.c_uint64, ctypes.c_int,
        ctypes.c_char_p, ctypes.c_size_t, ctypes.c_double,
    ]
    lib.locus_put.restype = ctypes.c_int
    for name in ("locus_lookup", "locus_retrieve"):
        fn = getattr(lib, name)
        fn.argtypes = [
            ctypes.c_void_p, ctypes.c_uint64, ctypes.POINTER(_View),
        ]
        fn.restype = ctypes.c_int
    lib.locus_restabilize.argtypes = [
        ctypes.c_void_p, ctypes.c_uint64, ctypes.c_char_p, ctypes.c_size_t,
    ]
    lib.locus_restabilize.restype = ctypes.c_int
    lib.locus_release.argtypes = [ctypes.c_void_p, ctypes.POINTER(_View)]
    lib.locus_prefetch.argtypes = [
        ctypes.c_void_p, ctypes.POINTER(ctypes.c_uint64), ctypes.c_size_t,
    ]
    lib.locus_prefetch.restype = ctypes.c_int
    lib.locus_tick.argtypes = [ctypes.c_void_p]
    lib.locus_note_salient.argtypes = [
        ctypes.c_void_p, ctypes.c_uint64, ctypes.c_double,
    ]
    lib.locus_reinforce.argtypes = [
        ctypes.c_void_p, ctypes.c_uint64, ctypes.c_double,
    ]
    lib.locus_reinforce.restype = ctypes.c_int
    lib.locus_gate.argtypes = [ctypes.c_void_p]
    lib.locus_gate.restype = ctypes.c_double
    lib.locus_excite.argtypes = [
        ctypes.c_void_p, ctypes.c_uint64, ctypes.c_double,
    ]
    lib.locus_excite.restype = ctypes.c_int
    lib.locus_stats.argtypes = [ctypes.c_void_p, ctypes.POINTER(_Stats)]
    for name in ("locus_trace_tier", "locus_trace_pathway",
                 "locus_trace_reps", "locus_trace_pinned"):
        fn = getattr(lib, name)
        fn.argtypes = [ctypes.c_void_p, ctypes.c_uint64]
        fn.restype = ctypes.c_int
    lib.locus_trace_activation.argtypes = [ctypes.c_void_p, ctypes.c_uint64]
    lib.locus_trace_activation.restype = ctypes.c_double
    lib.locus_trace_heat.argtypes = [ctypes.c_void_p, ctypes.c_uint64]
    lib.locus_trace_heat.restype = ctypes.c_double
    lib.locus_attend.argtypes = [ctypes.c_void_p, ctypes.c_uint64,
                                 ctypes.c_double]
    lib.locus_attend.restype = ctypes.c_int
    lib.locus_kwta.argtypes = [
        ctypes.POINTER(ctypes.c_double), ctypes.c_int, ctypes.c_int,
        ctypes.POINTER(ctypes.c_uint8),
    ]
    lib.locus_kwta_noisy.argtypes = [
        ctypes.POINTER(ctypes.c_double), ctypes.c_int, ctypes.c_int,
        ctypes.c_double, ctypes.POINTER(ctypes.c_uint32),
        ctypes.POINTER(ctypes.c_uint8),
    ]
    lib.locus_set_class.argtypes = [
        ctypes.c_void_p, ctypes.c_uint64, ctypes.c_uint8]
    lib.locus_set_class.restype = ctypes.c_int
    lib.locus_class_of.argtypes = [ctypes.c_void_p, ctypes.c_uint64]
    lib.locus_class_of.restype = ctypes.c_int
    # Last, so a mismatched mirror fails here rather than at the first
    # call that quietly reads the wrong bytes.
    _check_config_layout(lib)
    return lib


class Lease:
    """Context manager enforcing the release-exactly-once contract."""

    def __init__(self, store, view):
        self._store = store
        self._view = view

    @property
    def data(self):
        if not self._view.data:
            return b""
        return ctypes.string_at(self._view.data, self._view.len)

    @property
    def tier(self):
        return Tier(self._view.tier)

    @property
    def labile(self):
        return bool(self._view.labile)

    def __enter__(self):
        return self

    def __exit__(self, *_exc):
        self.release()
        return False

    def release(self):
        if self._view.slot:
            self._store._lib.locus_release(
                self._store._handle, ctypes.byref(self._view)
            )


class Store:
    """Tiered, lease-protected trace store."""

    def __init__(self, lib_path=None, **overrides):
        self._lib = _bind(lib_path or _default_lib_path())
        cfg = self._lib.locus_config_default()
        for key, value in overrides.items():
            if not hasattr(cfg, key):
                raise KeyError("unknown config field: %s" % key)
            setattr(cfg, key, value)
        self._handle = self._lib.locus_store_create(ctypes.byref(cfg))
        if not self._handle:
            raise RuntimeError("locus_store_create failed")
        self.config = cfg

    def close(self):
        if self._handle:
            self._lib.locus_store_destroy(self._handle)
            self._handle = None

    def __enter__(self):
        return self

    def __exit__(self, *_exc):
        self.close()
        return False

    def put(self, key, data, pathway=Pathway.DECLARATIVE, salience=1.0):
        payload = data if isinstance(data, bytes) else str(data).encode()
        rc = self._lib.locus_put(
            self._handle, key, int(pathway), payload, len(payload), salience
        )
        return rc == 0

    def _acquire(self, fn, key):
        view = _View()
        if fn(self._handle, key, ctypes.byref(view)) != 0:
            return None
        return Lease(self, view)

    def lookup(self, key):
        """Non-destabilising read. Fast and procedural pathways use this."""
        return self._acquire(self._lib.locus_lookup, key)

    def retrieve(self, key):
        """Declarative recall. Leaves the trace labile."""
        return self._acquire(self._lib.locus_retrieve, key)

    def restabilize(self, key, data=None):
        """Stabilise a labile trace.

        The trace stabilises either way; the return says whether the update
        was written or discarded because modulatory tone was too low.
        """
        payload = None
        if data is not None:
            payload = data if isinstance(data, bytes) else str(data).encode()
        return Restabilize(self._lib.locus_restabilize(
            self._handle, key, payload, len(payload) if payload else 0
        ))

    def prefetch(self, keys):
        keys = list(keys)
        if not keys:
            return
        arr = (ctypes.c_uint64 * len(keys))(*keys)
        self._lib.locus_prefetch(self._handle, arr, len(keys))

    def tick(self):
        self._lib.locus_tick(self._handle)

    def note_salient(self, key, strength):
        self._lib.locus_note_salient(self._handle, key, strength)

    def reinforce(self, key, surprise=0.0):
        """Register a re-application carrying its prediction error."""
        return self._lib.locus_reinforce(self._handle, key, surprise)

    def gate(self):
        """Current modulatory tone: tonic baseline plus phasic component."""
        return self._lib.locus_gate(self._handle)

    def attend(self, key, amount):
        """Task-driven drive, exempt from the refractory depression.

        `excite` is the unsolicited path and stays depressed.
        """
        return self._lib.locus_attend(self._handle, key, amount) == 0

    def excite(self, key, amount):
        return self._lib.locus_excite(self._handle, key, amount) == 0

    def stats(self):
        out = _Stats()
        self._lib.locus_stats(self._handle, ctypes.byref(out))
        return {f[0]: getattr(out, f[0]) for f in _Stats._fields_}

    def tier(self, key):
        rc = self._lib.locus_trace_tier(self._handle, key)
        return None if rc < 0 else Tier(rc)

    def pathway(self, key):
        rc = self._lib.locus_trace_pathway(self._handle, key)
        return None if rc < 0 else Pathway(rc)

    def reps(self, key):
        return self._lib.locus_trace_reps(self._handle, key)

    def pinned(self, key):
        return self._lib.locus_trace_pinned(self._handle, key) == 1

    def activation(self, key):
        return self._lib.locus_trace_activation(self._handle, key)

    def heat(self, key):
        """Usage history, which outlives activation. Read-only."""
        return self._lib.locus_trace_heat(self._handle, key)

    def kwta(self, activations, k, temp=0.0, seed=1):
        """Competition over an activation vector.

        temp 0 is deterministic top-k; above 0 the winners are drawn from
        the softmax over activations rather than ranked.
        """
        n = len(activations)
        arr = (ctypes.c_double * n)(*activations)
        win = (ctypes.c_uint8 * n)()
        if temp > 0.0:
            rng = ctypes.c_uint32(seed)
            self._lib.locus_kwta_noisy(arr, n, k, temp, ctypes.byref(rng), win)
        else:
            self._lib.locus_kwta(arr, n, k, win)
        return [bool(w) for w in win]
