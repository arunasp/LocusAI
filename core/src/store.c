/* store.c — LocusAI tiered trace store. */

#include "locus.h"

#include <math.h>
#include <stdlib.h>
#include <string.h>

/* Ticks a retrieved trace may stay labile before it is counted as lost. */
#define LOCUS_LABILE_TICKS 2

/* Activation below which an unpinned declarative trace demotes a tier. */
#define LOCUS_DEMOTE_FLOOR 0.05

/* Heat decays slower than activation: usage history outlives the moment. */
#define LOCUS_HEAT_DAMP 0.25

/* Stack-allocated working set for noisy competition; larger stores fall back
 * to deterministic selection rather than allocating on the tick path. */
#define LOCUS_KWTA_MAX 256

typedef struct {
    LocusKey key;
    void *data;
    size_t len;
    LocusTier tier;
    LocusPathway path;
    double activation;
    double heat;
    double tag;
    int tag_ticks;
    int reps;
    int calm_run; /* consecutive unsurprising re-applications */
    int pinned;
    int labile;
    int labile_ticks;
    int leases;
    int lease_ticks;
    int prefetched;
    int used;
    uint32_t generation; /* bumped on reuse and on lease revocation */
} Slot;

struct LocusStore {
    Slot *slots;
    int capacity;
    LocusConfig cfg;
    LocusStats stats;
    double phasic; /* decaying component of modulatory tone */
    uint32_t rng;
};

LocusConfig locus_config_default(void)
{
    LocusConfig c;
    c.capacity = 1024;
    c.active_k = 4; /* interference-bounded, not byte-bounded */
    c.decay = 0.85;
    c.capture_floor = 0.1;
    c.capture_gain = 0.5;
    c.capture_window = 3;
    c.promote_after = 3;
    c.lease_max_ticks = 8;
    c.kwta_temp = 0.1;
    c.surprise_floor = 0.1;
    c.gate_tonic = 1.0;
    c.gate_threshold = 0.5;
    c.gate_decay = 0.5;
    c.seed = 0;
    return c;
}

static uint32_t xorshift32(uint32_t *state)
{
    uint32_t x = *state;
    x ^= x << 13;
    x ^= x >> 17;
    x ^= x << 5;
    *state = x;
    return x;
}

/* Uniform in (0,1); never returns exactly 0 or 1, so log() stays finite. */
static double next_unit(uint32_t *state)
{
    return ((double)(xorshift32(state) >> 8) + 0.5) / 16777216.0;
}

static int find(const LocusStore *s, LocusKey key)
{
    for (int i = 0; i < s->capacity; i++)
        if (s->slots[i].used && s->slots[i].key == key)
            return i;
    return -1;
}

static void clear_view(LocusView *v)
{
    if (v)
        memset(v, 0, sizeof(*v));
}

/* Reuse invalidates every outstanding view of this slot. */
static void slot_free(Slot *t)
{
    uint32_t generation = t->generation + 1;
    free(t->data);
    memset(t, 0, sizeof(*t));
    t->generation = generation;
}

/* Lowest-activation unpinned, unleased slot. Competition, not size. */
static int weakest(const LocusStore *s)
{
    int worst = -1;
    for (int i = 0; i < s->capacity; i++) {
        const Slot *t = &s->slots[i];
        if (!t->used || t->pinned || t->leases > 0)
            continue;
        if (worst < 0 || t->activation < s->slots[worst].activation)
            worst = i;
    }
    return worst;
}

LocusStore *locus_store_create(const LocusConfig *cfg)
{
    LocusStore *s = calloc(1, sizeof(*s));
    if (!s)
        return NULL;
    s->cfg = cfg ? *cfg : locus_config_default();
    if (s->cfg.capacity < 1 || s->cfg.active_k < 1 || s->cfg.decay <= 0.0 ||
        s->cfg.decay >= 1.0) {
        free(s);
        return NULL;
    }
    s->capacity = s->cfg.capacity;
    s->slots = calloc((size_t)s->capacity, sizeof(Slot));
    if (!s->slots) {
        free(s);
        return NULL;
    }
    s->rng = s->cfg.seed ? s->cfg.seed : 0x9e3779b9u;
    s->phasic = 0.0;
    return s;
}

void locus_store_destroy(LocusStore *s)
{
    if (!s)
        return;
    for (int i = 0; i < s->capacity; i++)
        if (s->slots[i].used)
            free(s->slots[i].data);
    free(s->slots);
    free(s);
}

int locus_put(LocusStore *s, LocusKey key, LocusPathway path,
              const void *data, size_t len, double salience)
{
    if (!s || (!data && len))
        return -1;

    int i = find(s, key);
    if (i < 0) {
        for (i = 0; i < s->capacity && s->slots[i].used; i++)
            ;
        if (i == s->capacity) {
            i = weakest(s);
            if (i < 0)
                return -1;
            slot_free(&s->slots[i]);
            s->stats.evictions++;
        }
    } else {
        slot_free(&s->slots[i]);
    }

    Slot *t = &s->slots[i];
    if (len) {
        t->data = malloc(len);
        if (!t->data)
            return -1;
        memcpy(t->data, data, len);
    }
    t->key = key;
    t->len = len;
    t->path = path;
    t->activation = salience;
    t->heat = salience;
    t->tag = salience;
    t->tag_ticks = s->cfg.capture_window;
    t->used = 1;

    if (path == LOCUS_PATH_INSTINCT) {
        t->pinned = 1;
        t->tier = LOCUS_TIER_ACTIVE;
    } else {
        t->tier = LOCUS_TIER_EPISODIC;
    }
    return 0;
}

static int acquire(LocusStore *s, LocusKey key, LocusView *v, int destabilise)
{
    clear_view(v);
    if (!s || !v)
        return -1;
    s->stats.requests++;

    int i = find(s, key);
    if (i < 0) {
        s->stats.misses++;
        return -1;
    }

    Slot *t = &s->slots[i];
    s->stats.hits++;
    if (t->prefetched) {
        s->stats.prefetch_hits++;
        t->prefetched = 0;
    }
    t->activation += 1.0;
    t->heat += 1.0;
    if (t->leases == 0)
        t->lease_ticks = 0;
    t->leases++;

    if (destabilise) {
        t->labile = 1;
        t->labile_ticks = LOCUS_LABILE_TICKS;
    }

    v->key = key;
    v->data = t->data;
    v->len = t->len;
    v->tier = t->tier;
    v->labile = t->labile;
    v->generation = t->generation;
    v->slot = t;
    return 0;
}

int locus_lookup(LocusStore *s, LocusKey key, LocusView *v)
{
    return acquire(s, key, v, 0);
}

int locus_retrieve(LocusStore *s, LocusKey key, LocusView *v)
{
    return acquire(s, key, v, 1);
}

double locus_gate(const LocusStore *s)
{
    return s ? s->cfg.gate_tonic + s->phasic : 0.0;
}

int locus_restabilize(LocusStore *s, LocusKey key, const void *data,
                      size_t len)
{
    if (!s)
        return LOCUS_RESTABILIZE_ERR;
    int i = find(s, key);
    if (i < 0)
        return LOCUS_RESTABILIZE_ERR;

    Slot *t = &s->slots[i];
    if (!t->labile)
        return LOCUS_RESTABILIZE_ERR;

    /* The trace stabilises either way. Tone decides only whether what was
     * read back may overwrite what was stored: consolidating a CHANGE is the
     * neuromodulator-gated step, not surviving retrieval. */
    int gated = locus_gate(s) < s->cfg.gate_threshold;

    if (data && !gated) {
        void *fresh = len ? malloc(len) : NULL;
        if (len && !fresh)
            return LOCUS_RESTABILIZE_ERR;
        if (len)
            memcpy(fresh, data, len);
        free(t->data);
        t->data = fresh;
        t->len = len;
    } else if (data && gated) {
        s->stats.updates_gated++;
    }

    t->labile = 0;
    t->labile_ticks = 0;
    return gated ? LOCUS_GATED : LOCUS_RESTABILIZED;
}

void locus_release(LocusStore *s, LocusView *v)
{
    if (!s || !v || !v->slot) {
        clear_view(v);
        return;
    }
    Slot *t = (Slot *)v->slot;
    /* A revoked or recycled slot no longer owes this view a decrement. */
    if (t->generation == v->generation && t->leases > 0) {
        t->leases--;
        if (t->leases == 0)
            t->lease_ticks = 0;
    }
    clear_view(v);
}

int locus_prefetch(LocusStore *s, const LocusKey *keys, size_t n)
{
    if (!s || !keys)
        return 0; /* advisory: never an error */
    for (size_t j = 0; j < n; j++) {
        int i = find(s, keys[j]);
        if (i < 0)
            continue;
        s->slots[i].prefetched = 1;
        s->stats.prefetched++;
    }
    return 0;
}

void locus_note_salient(LocusStore *s, LocusKey key, double strength)
{
    if (!s || strength <= 0.0)
        return;

    /* Phasic tone: the same salience signal that rescues weak traces also
     * opens the gate on consolidating changes. */
    s->phasic += strength;

    int origin = find(s, key);
    if (origin >= 0) {
        s->slots[origin].activation += strength;
        s->slots[origin].heat += strength;
        s->slots[origin].tag = strength;
        s->slots[origin].tag_ticks = s->cfg.capture_window;
    }

    /* Synaptic tagging and capture: a weakly-tagged trace inside the window
     * is rescued by an unrelated strong event nearby in time. */
    for (int i = 0; i < s->capacity; i++) {
        Slot *t = &s->slots[i];
        if (!t->used || i == origin)
            continue;
        if (t->tag_ticks <= 0 || t->tag < s->cfg.capture_floor)
            continue;
        t->activation += strength * s->cfg.capture_gain;
        t->heat += strength * s->cfg.capture_gain;
        s->stats.captures++;
    }
}

int locus_reinforce(LocusStore *s, LocusKey key, double surprise)
{
    if (!s)
        return -1;
    int i = find(s, key);
    if (i < 0)
        return -1;

    Slot *t = &s->slots[i];
    t->reps++;
    t->heat += 1.0;

    /* An outcome that still surprises breaks the run: repetition is necessary
     * for a habit but not sufficient. What promotes is a sequence that has
     * stopped producing prediction error. */
    if (fabs(surprise) <= s->cfg.surprise_floor)
        t->calm_run++;
    else
        t->calm_run = 0;

    /* Repetition-count promotion IS tier promotion: the procedural
     * consolidator and the residency manager are one mechanism. */
    if (!t->pinned && t->reps >= s->cfg.promote_after &&
        t->calm_run >= s->cfg.promote_after) {
        t->pinned = 1;
        t->path = LOCUS_PATH_PROCEDURAL;
        t->tier = LOCUS_TIER_ACTIVE;
        s->stats.promotions++;
    }
    return t->reps;
}

int locus_excite(LocusStore *s, LocusKey key, double amount)
{
    if (!s)
        return -1;
    int i = find(s, key);
    if (i < 0)
        return -1;
    s->slots[i].activation += amount;
    return 0;
}

/* Bound the unpinned active set by competition. Pinned traces are required
 * residents and are exempt; they do not consume kWTA width. Selection is
 * noisy, so a marginally weaker contender sometimes holds its place. */
static void enforce_kwta(LocusStore *s)
{
    int idx[LOCUS_KWTA_MAX];
    /* Zeroed so -Wmaybe-uninitialized can see the whole array is defined;
     * only the first n entries are ever read. */
    double act[LOCUS_KWTA_MAX] = { 0 };
    uint8_t win[LOCUS_KWTA_MAX];
    int n = 0;

    for (int i = 0; i < s->capacity && n < LOCUS_KWTA_MAX; i++) {
        Slot *t = &s->slots[i];
        if (!t->used || t->pinned || t->tier != LOCUS_TIER_ACTIVE)
            continue;
        idx[n] = i;
        act[n] = t->activation;
        n++;
    }
    if (n <= s->cfg.active_k)
        return;

    locus_kwta_noisy(act, n, s->cfg.active_k, s->cfg.kwta_temp, &s->rng, win);
    for (int j = 0; j < n; j++)
        if (!win[j])
            s->slots[idx[j]].tier = LOCUS_TIER_EPISODIC;
}

void locus_tick(LocusStore *s)
{
    if (!s)
        return;
    const double heat_decay =
        1.0 - (1.0 - s->cfg.decay) * LOCUS_HEAT_DAMP;

    s->phasic *= s->cfg.gate_decay;
    if (s->phasic < 1e-9)
        s->phasic = 0.0;

    for (int i = 0; i < s->capacity; i++) {
        Slot *t = &s->slots[i];
        if (!t->used)
            continue;

        t->activation *= s->cfg.decay;
        t->heat *= heat_decay;

        if (t->tag_ticks > 0 && --t->tag_ticks == 0)
            t->tag = 0.0;

        /* Leases expire. Without this a leaked lease pins its slot forever
         * and encoding deadlocks once every slot is held. lease_ticks is
         * written only here, never by the layer the expiry constrains. */
        if (t->leases > 0 && ++t->lease_ticks > s->cfg.lease_max_ticks) {
            t->leases = 0;
            t->lease_ticks = 0;
            t->generation++;
            s->stats.leases_revoked++;
        }

        if (t->labile && --t->labile_ticks <= 0) {
            /* Never re-stabilised: the trace weakens rather than persisting
             * unchanged. Retrieval without restabilisation is lossy. */
            t->activation *= 0.5;
            t->labile = 0;
            s->stats.labile_lost++;
        }

        if (t->pinned)
            continue;

        if (t->activation >= 1.0 && t->tier == LOCUS_TIER_EPISODIC)
            t->tier = LOCUS_TIER_ACTIVE;
        else if (t->activation < LOCUS_DEMOTE_FLOOR &&
                 t->tier != LOCUS_TIER_ARCHIVE && t->leases == 0)
            t->tier = LOCUS_TIER_ARCHIVE;
    }
    enforce_kwta(s);
}

void locus_stats(const LocusStore *s, LocusStats *out)
{
    if (!s || !out)
        return;
    *out = s->stats;
    out->active_traces = 0;
    out->resident_traces = 0;
    for (int i = 0; i < s->capacity; i++) {
        if (!s->slots[i].used)
            continue;
        out->resident_traces++;
        if (s->slots[i].tier == LOCUS_TIER_ACTIVE)
            out->active_traces++;
    }
}

int locus_trace_tier(const LocusStore *s, LocusKey key)
{
    int i = s ? find(s, key) : -1;
    return i < 0 ? -1 : (int)s->slots[i].tier;
}

int locus_trace_pathway(const LocusStore *s, LocusKey key)
{
    int i = s ? find(s, key) : -1;
    return i < 0 ? -1 : (int)s->slots[i].path;
}

int locus_trace_reps(const LocusStore *s, LocusKey key)
{
    int i = s ? find(s, key) : -1;
    return i < 0 ? -1 : s->slots[i].reps;
}

double locus_trace_activation(const LocusStore *s, LocusKey key)
{
    int i = s ? find(s, key) : -1;
    return i < 0 ? -1.0 : s->slots[i].activation;
}

int locus_trace_pinned(const LocusStore *s, LocusKey key)
{
    int i = s ? find(s, key) : -1;
    return i < 0 ? -1 : s->slots[i].pinned;
}

void locus_kwta(const double *act, int n, int k, uint8_t *winners)
{
    if (!act || !winners || n <= 0)
        return;
    memset(winners, 0, (size_t)n);
    if (k <= 0)
        return;
    if (k > n)
        k = n;

    for (int w = 0; w < k; w++) {
        int best = -1;
        for (int i = 0; i < n; i++) {
            if (winners[i])
                continue;
            if (best < 0 || act[i] > act[best])
                best = i;
        }
        if (best < 0)
            return;
        winners[best] = 1;
    }
}

void locus_kwta_noisy(const double *act, int n, int k, double temp,
                      uint32_t *rng, uint8_t *winners)
{
    double noisy[LOCUS_KWTA_MAX];

    if (!act || !winners || n <= 0)
        return;
    if (temp <= 0.0 || !rng || n > LOCUS_KWTA_MAX) {
        locus_kwta(act, n, k, winners);
        return;
    }

    /* Gumbel-top-k: perturbing by -temp*log(-log(u)) and taking the top k
     * draws k winners from the softmax over activations without replacement,
     * so competition is genuinely a draw rather than a ranking. */
    for (int i = 0; i < n; i++) {
        double u = next_unit(rng);
        noisy[i] = act[i] - temp * log(-log(u));
    }
    locus_kwta(noisy, n, k, winners);
}
