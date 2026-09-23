/* locus.h — LocusAI memory substrate: tiered, lease-protected trace store.
 *
 * Storage semantics deliberately differ from a cache:
 *   - tiers hold representations, not identical copies of one blob;
 *   - retrieval destabilises the trace (read path is also a write path);
 *   - the active set is bounded by competition (kWTA), never by bytes.
 */
#ifndef LOCUS_H
#define LOCUS_H

#include <stddef.h>
#include <stdint.h>

/* Bitmask width for trace classes: a class id is a bit position, and 0
 * means unclassified. */
/* A BOUND set by the representation: conflicts are a uint32_t bitmask,
 * so 32 is the width of that word rather than a judgement about how
 * many action classes a constitution needs. */
#define LOCUS_CLASS_MAX 32

/* Moments of active-set history kept for the promotion trigger. A habit
 * forms from a WINNER SEQUENCE that recurs, so the store has to remember
 * which sets it has been in. 32 is an initial value, not a measured one:
 * the open question is how far back a recurrence should still count,
 * which wants an answer from the data rather than from this header. */
#define LOCUS_SEQ_HISTORY 32

#ifdef __cplusplus
extern "C" {
#endif

/* Residency class, not a speed ranking. ACTIVE is the only tier where an
 * activity-maintained state can exist, because it is the only one that lives
 * where the compute does. */
typedef enum {
    LOCUS_TIER_ACTIVE = 0,   /* device-local, activity-maintained */
    LOCUS_TIER_EPISODIC = 1, /* host memory, sparse one-shot traces */
    LOCUS_TIER_ARCHIVE = 2   /* backing store, transformed/statistical */
} LocusTier;

/* Pathway drives residency policy. Non-declarative pathways cannot tolerate a
 * miss -- a habit that needs a slow fetch has lost the property that made it a
 * habit -- so they pin. */
typedef enum {
    LOCUS_PATH_DECLARATIVE = 0,
    LOCUS_PATH_PROCEDURAL = 1,
    LOCUS_PATH_INSTINCT = 2
} LocusPathway;

/* locus_restabilize() outcomes. */
typedef enum {
    LOCUS_RESTABILIZED = 0, /* trace stabilised, update written */
    LOCUS_GATED = 1,        /* stabilised, update discarded: tone too low */
    LOCUS_RESTABILIZE_ERR = -1
} LocusRestabilizeResult;

typedef uint64_t LocusKey;

typedef struct LocusStore LocusStore;

/* Lease contract:
 *   - after a successful lookup/retrieve the caller MUST release exactly once;
 *   - a leased slot is never evicted, and prefetch may never break a lease;
 *   - on failure the view is cleared and must not be released;
 *   - retrieve() leaves the trace LABILE: locus_restabilize() must follow
 *     before the trace is durable again. An unrestabilised trace decays.
 *
 * Leases EXPIRE. A lease held for more than cfg.lease_max_ticks is revoked by
 * locus_tick(), which is what stops a leaked lease from deadlocking encoding.
 * A revoked view must not be dereferenced; releasing one is safe and is a
 * no-op, detected via the generation stamp. Callers needing a trace for longer
 * than the expiry window must re-acquire it rather than hold across ticks.
 */
typedef struct {
    LocusKey key;
    const void *data;
    size_t len;
    LocusTier tier;
    int labile;
    uint32_t generation; /* slot stamp; mismatch means the view is stale */
    void *slot;          /* opaque lease handle; NULL when the view is cleared */
} LocusView;

typedef struct {
    uint64_t requests;
    uint64_t hits;
    uint64_t misses;
    uint64_t prefetched;
    uint64_t prefetch_hits;
    uint64_t promotions;
    uint64_t evictions;
    uint64_t captures;       /* weak traces rescued by a nearby salient event */
    uint64_t labile_lost;
    uint64_t leases_revoked; /* leases expired by tick, freeing their slot */
    uint64_t updates_gated;  /* re-stabilisations that discarded their update */
    uint64_t active_traces;
    uint64_t resident_traces;
} LocusStats;

typedef struct {
    int capacity;         /* total traces the store may hold */
    int active_k;         /* kWTA width: max simultaneously ACTIVE traces */
    double decay;         /* per-tick multiplicative heat decay, 0<d<1 */
    double capture_floor; /* min tag strength eligible for capture */
    double capture_gain;  /* fraction of a salient event's strength captured */
    int capture_window;   /* ticks a tag stays eligible */
    int promote_after;    /* successful re-applications before pinning */
    int lease_max_ticks;  /* ticks before a held lease is revoked */

    /* Competition is noisy: winners are drawn rather than ranked. 0 collapses
     * to deterministic top-k, which is what tests asserting an exact winner
     * set use. Higher values widen how often a weaker trace wins. */
    double kwta_temp;

    /* Repetition alone is not the promotion trigger. A re-application only
     * counts toward a habit when its outcome was unsurprising -- prediction
     * error at or below this floor. */
    double surprise_floor;

    /* Modulatory tone gating whether a re-stabilisation may overwrite the
     * trace it read. tonic is the baseline; salient events add a phasic
     * component that decays back toward it. Default tonic sits above
     * threshold, so the gate is open until a real modulator drives it. */
    double gate_tonic;
    double gate_threshold;
    double gate_decay;

    uint32_t seed; /* 0 selects a fixed default; runs are reproducible */

    /* Incompatibility between trace classes. conflicts[c] is a bitmask of
     * the classes that may not be ACTIVE alongside class c, so kWTA bounds
     * the active set by class as well as by activation: a strong trace no
     * longer drags an incompatible one into the same moment.
     *
     * It lives in the config because it is fixed at construction. The layer
     * it constrains selects traces; it does not get to say which selections
     * are permitted, which is what makes this a constraint rather than a
     * preference (doc/core/CONSTITUTION.md).
     *
     * All zero by default: nothing conflicts with anything and selection
     * behaves exactly as before. */
    uint32_t conflicts[LOCUS_CLASS_MAX];
} LocusConfig;

/* Defaults grounded in the design notes: kWTA width in the Cowan range,
 * capture window short, promotion threshold deliberately low so the
 * procedural path is observable early. */
LocusConfig locus_config_default(void);

/* Layout of LocusConfig, so a binding can verify its own mirror instead of
 * trusting it. A Python ctypes Structure that disagrees with this struct is
 * silent memory corruption, not an error, so the binding checks at load.
 *
 * locus_config_size() is sizeof(LocusConfig). locus_config_layout() fills
 * `out` with each field's byte offset in declaration order and returns the
 * number of fields, or the count alone when out is NULL or max is too
 * small. Offsets catch reordering and padding; size catches a missing or
 * added field. */
size_t locus_config_size(void);
int locus_config_layout(size_t *out, int max);

LocusStore *locus_store_create(const LocusConfig *cfg);
void locus_store_destroy(LocusStore *s);

/* Encode a trace. Novelty gating happens above this layer; put() is
 * unconditional. Returns 0 on success, -1 on capacity exhaustion. */
int locus_put(LocusStore *s, LocusKey key, LocusPathway path,
              const void *data, size_t len, double salience);

/* Non-destabilising read. Used by the fast and procedural pathways, which
 * must not pay a reconsolidation cost. Returns 0 on hit. */
int locus_lookup(LocusStore *s, LocusKey key, LocusView *v);

/* Destabilising read: declarative recall. Marks the trace labile. */
int locus_retrieve(LocusStore *s, LocusKey key, LocusView *v);

/* Re-stabilise after retrieve(). The trace always stabilises; whether the
 * supplied payload REPLACES it depends on modulatory tone at this moment.
 * Returns LOCUS_RESTABILIZED when the update was written, LOCUS_GATED when it
 * was discarded, LOCUS_RESTABILIZE_ERR when there was nothing labile. */
int locus_restabilize(LocusStore *s, LocusKey key, const void *data,
                      size_t len);

void locus_release(LocusStore *s, LocusView *v);

/* Advisory. Never evicts a leased slot, never reports failure as an error. */
int locus_prefetch(LocusStore *s, const LocusKey *keys, size_t n);

/* Drive one tick: decay heat and tone, expire capture tags, revoke expired
 * leases, drop unrestabilised traces, enforce the kWTA active-set bound. */
void locus_tick(LocusStore *s);

/* Record a salient event at a trace. Strong events rescue weakly-tagged
 * neighbours inside the capture window (synaptic tagging and capture) and
 * raise the phasic component of modulatory tone. */
void locus_note_salient(LocusStore *s, LocusKey key, double strength);

/* Register a re-application, with the prediction error of its outcome.
 * Promotion requires both enough re-applications AND a run of them whose
 * outcome had stopped being surprising: pass |delta| for the step. Returns
 * the new re-application count, or -1 if absent. */
int locus_reinforce(LocusStore *s, LocusKey key, double surprise);

/* Raise activation directly; used by spreading activation from above. */
int locus_excite(LocusStore *s, LocusKey key, double amount);

void locus_stats(const LocusStore *s, LocusStats *out);

/* Current modulatory tone (tonic + phasic). */
double locus_gate(const LocusStore *s);

/* Introspection for the Python layer and tests. Returns -1 if absent. */
int locus_trace_tier(const LocusStore *s, LocusKey key);
int locus_trace_pathway(const LocusStore *s, LocusKey key);
int locus_trace_reps(const LocusStore *s, LocusKey key);
double locus_trace_activation(const LocusStore *s, LocusKey key);
int locus_trace_pinned(const LocusStore *s, LocusKey key);

/* Deterministic k-winners-take-all over an activation vector. Exposed because
 * the same competition bounds the active set and the separation stage. */
void locus_kwta(const double *act, int n, int k, uint8_t *winners);

/* Trace class, 0..LOCUS_CLASS_MAX-1, where 0 means unclassified. Returns 0
 * on success, -1 if the key is absent or the class is out of range. */
int locus_set_class(LocusStore *s, LocusKey key, uint8_t klass);

/* Class of a trace, or -1 if the key is absent. */
int locus_class_of(const LocusStore *s, LocusKey key);

/* Stochastic variant: perturbs each activation by Gumbel noise scaled by
 * temp, then takes the top k, which draws k winners from the softmax over
 * activations without replacement. temp <= 0 falls through to locus_kwta().
 * rng is advanced in place, so a caller holding a seed gets a reproducible
 * sequence. */
void locus_kwta_noisy(const double *act, int n, int k, double temp,
                      uint32_t *rng, uint8_t *winners);

#ifdef __cplusplus
}
#endif

#endif /* LOCUS_H */
