/* store.c — LocusAI tiered trace store. */

#include "locus.h"

#include <stddef.h>

#include <math.h>
#include <stdlib.h>
#include <string.h>

/* Every double below crosses into the store from a CALLER. A NaN that
 * gets in is unrecoverable and silent: it lands in activation or heat,
 * every comparison against it is false, so the trace neither wins nor
 * loses selection predictably, and the store is persistent state. The
 * existing `strength <= 0.0` guard did NOT catch it -- NaN fails that
 * test too -- which is why this is a check of its own rather than a
 * range test doing double duty. */
static int usable(double v)
{
    return !isnan(v) && !isinf(v);
}

/* Ticks a retrieved trace may stay labile before it is counted as lost.
 * An INITIAL VALUE, not measured: reconsolidation in the literature is
 * a window of hours, and a tick here is not a unit of time, so the open
 * question is what a tick should correspond to before this can be set
 * from anything but convenience. */
#define LOCUS_LABILE_TICKS 2

/* Heat decays slower than activation: usage history outlives the moment.
 * The RATIO is the claim and it is structural; the 0.25 itself is an
 * INITIAL VALUE and the open question is what sets the separation
 * between the two timescales. */
#define LOCUS_HEAT_DAMP 0.25

/* Stack-allocated working set for noisy competition; larger stores fall back
 * to deterministic selection rather than allocating on the tick path.
 * A BOUND on stack use, not a claim about the substrate: it limits how
 * many contenders the noisy path can hold, and a larger store degrades
 * to the deterministic path rather than failing. */
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
    uint8_t klass; /* trace class; 0 is unclassified */
    uint32_t generation; /* bumped on reuse and on lease revocation */
} Slot;

struct LocusStore {
    Slot *slots;
    int capacity;
    LocusConfig cfg;
    LocusStats stats;
    double phasic; /* decaying component of modulatory tone */
    uint32_t rng;
    /* Ring of recent active-set signatures, newest at seq_at. */
    uint64_t seq[LOCUS_SEQ_HISTORY];
    int seq_at;
    int seq_len;
};

/* Signature of the active set: the keys of every ACTIVE slot, in slot
 * order, hashed. Two moments with the same winners have the same
 * signature; one extra or missing winner changes it. */
static uint64_t active_signature(const LocusStore *s)
{
    uint64_t h = 1469598103934665603ULL; /* FNV-1a offset basis */
    int any = 0;
    for (int i = 0; i < s->capacity; i++) {
        const Slot *t = &s->slots[i];
        if (!t->used || t->tier != LOCUS_TIER_ACTIVE)
            continue;
        uint64_t k = (uint64_t)t->key;
        any = 1;
        for (int b = 0; b < 8; b++) {
            h ^= (k >> (b * 8)) & 0xff;
            h *= 1099511628211ULL;
        }
    }
    /* An EMPTY active set is not a winner sequence. Left as a signature
     * it would recur constantly and promote things that never won
     * anything. */
    return any ? h : 0;
}

/* Has this exact set of winners been active in an EARLIER moment? */
static int signature_recurred(const LocusStore *s, uint64_t sig)
{
    int seen = 0;
    if (!sig)
        return 0;
    for (int i = 0; i < s->seq_len; i++)
        if (s->seq[i] == sig && ++seen >= 2)
            return 1;
    return 0;
}

static void remember_active_set(LocusStore *s)
{
    uint64_t sig = active_signature(s);
    if (!sig)
        return;
    s->seq[s->seq_at] = sig;
    s->seq_at = (s->seq_at + 1) % LOCUS_SEQ_HISTORY;
    if (s->seq_len < LOCUS_SEQ_HISTORY)
        s->seq_len++;
}

size_t locus_config_size(void)
{
    return sizeof(LocusConfig);
}

int locus_config_layout(size_t *out, int max)
{
    static const size_t off[] = {
        offsetof(LocusConfig, capacity),
        offsetof(LocusConfig, active_k),
        offsetof(LocusConfig, decay),
        offsetof(LocusConfig, capture_floor),
        offsetof(LocusConfig, capture_gain),
        offsetof(LocusConfig, capture_window),
        offsetof(LocusConfig, promote_after),
        offsetof(LocusConfig, lease_max_ticks),
        offsetof(LocusConfig, kwta_temp),
        offsetof(LocusConfig, beta),
        offsetof(LocusConfig, surprise_floor),
        offsetof(LocusConfig, gate_tonic),
        offsetof(LocusConfig, gate_threshold),
        offsetof(LocusConfig, gate_decay),
        offsetof(LocusConfig, seed),
        offsetof(LocusConfig, conflicts),
    };
    const int n = (int)(sizeof(off) / sizeof(*off));
    if (out && max >= n)
        for (int i = 0; i < n; i++)
            out[i] = off[i];
    return n;
}

LocusConfig locus_config_default(void)
{
    LocusConfig c;
    c.capacity = 1024;
    c.active_k = 4; /* interference-bounded, not byte-bounded */
    for (int i = 0; i < LOCUS_CLASS_MAX; i++)
        c.conflicts[i] = 0; /* nothing conflicts until something says so */
    c.decay = 0.85;
    c.capture_floor = 0.1;
    c.capture_gain = 0.5;
    c.capture_window = 3;
    c.promote_after = 3;
    c.lease_max_ticks = 8;
    c.kwta_temp = 0.1;
    c.beta = 0.1;
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
    if (!s || (!data && len) || !usable(salience))
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
    t->klass = 0; /* a reused slot must not inherit the last trace's class */
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
    if (!s || !usable(strength) || strength <= 0.0)
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
    if (!s || !usable(surprise))
        return -1;
    int i = find(s, key);
    if (i < 0)
        return -1;

    Slot *t = &s->slots[i];
    t->reps++;
    t->heat += 1.0;

    /* An outcome that still surprises breaks the run: repetition is necessary
     * for a habit but not sufficient. What promotes is a sequence that has
     * stopped producing prediction error.
     *
     * And the sequence is the WINNER SET, not this trace alone. A calm
     * re-application inside a set of winners the store has never been in
     * before is a new situation that happened to go well, which is not a
     * habit; it counts only once that set RECURS. Without this, repetition
     * alone promotes, which is what doc/core/ROADMAP.md stage 1 names. */
    if (fabs(surprise) <= s->cfg.surprise_floor
            && signature_recurred(s, active_signature(s)))
        t->calm_run++;
    else if (fabs(surprise) > s->cfg.surprise_floor)
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
    if (!s || !usable(amount))
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
/* Does this class conflict with any already admitted? The relation is used
 * symmetrically: a caller that declares only one direction still gets the
 * exclusion, because co-activation has no direction. */
static int conflicts_with(const LocusStore *s, uint8_t klass, uint32_t held)
{
    if (klass >= LOCUS_CLASS_MAX)
        return 0;
    if (s->cfg.conflicts[klass] & held)
        return 1;
    for (int c = 0; c < LOCUS_CLASS_MAX; c++)
        if ((held & (1u << c)) && (s->cfg.conflicts[c] & (1u << klass)))
            return 1;
    return 0;
}

/* Total activation of the traces a trace competes with: used, not
 * pinned, not archived. Computed per tick rather than carried, because
 * a carried total is a second copy of the state it summarises. */
/* COMPETITION IS WITHIN A PATHWAY, NOT ACROSS THEM. One pool means a
 * saturated focal load silences everything else -- measured: seven
 * declarative winners and nothing left for anything watching the
 * surroundings. Biology separates the maps instead: interference is
 * strongest between SIMILAR representations (Desimone & Duncan 1995),
 * and a goal-directed set runs alongside a stimulus-driven monitor
 * rather than starving it (Corbetta & Shulman 2002). The pathways are
 * already the maps here: declarative, procedural, instinct.
 *
 * So each pathway carries its own capacity window, and a background
 * pathway keeps its own however busy the foreground is. */
static double competing_total(const LocusStore *s, LocusPathway pathway)
{
    /* PRIORITY BETWEEN POOLS IS AN INTERRUPT, NOT A RANKING. Phasic
     * noradrenaline is a network reset: it interrupts ongoing activity
     * and reorganises which network holds the output (Bouret & Sara
     * 2005; Dayan & Yu 2006, "a neural interrupt signal for unexpected
     * events"). Tonic level sets the regime -- moderate while engaged
     * and filtering, high while uncommitted and responsive to
     * unanticipated change -- so one scalar decides how porous the
     * pools are, and that scalar already exists here as tone.
     *
     * So the phasic component IS the coupling: at baseline tone each
     * pathway competes alone, and while a salient event holds tone
     * above baseline the pools couple and the strongest trace across
     * them wins. The reset is transient by construction, because
     * phasic decays back (gate_decay) without anything resetting it.
     *
     * NOT IMPLEMENTED, deliberately: whether breakthrough should
     * require the event to MATCH current top-down settings. Contingent
     * capture (Folk et al. 1992) says yes, the additional-singleton
     * results say no, and load theory's account of what a busy
     * foreground filters out has recent replication failures. An
     * undisputed interrupt is worth building; a disputed condition on
     * it is not. */
    /* The coupling lives in the SETTLING pass only. Adding it here too
     * survived its own mutation -- removing it changed no test -- and
     * an unbound term is inert whatever it looks like it does: within
     * one pathway it scales every trace and the boundary alike, so the
     * comparison it feeds is unchanged. */
    double own = 0.0;
    for (int i = 0; i < s->capacity; i++) {
        const Slot *t = &s->slots[i];
        if (t->used && !t->pinned && t->tier != LOCUS_TIER_ARCHIVE
            && t->path == pathway)
            own += t->activation;
    }
    return own;
}

/* What a trace is WORTH against the rest, which is what selection reads.
 * The stored activation is never divided -- normalising the state itself
 * would compound every tick. Same shape as Field.step's divisive term. */
static double effective(const LocusStore *s, const Slot *t, double total)
{
    double others = total - t->activation;
    if (others < 0.0)
        others = 0.0;
    return t->activation / (1.0 + s->cfg.beta * others);
}

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
        /* Ranked against its OWN pathway: the noisy draw below picks
         * among peers, and a declarative trace is not a peer of an
         * instinct one. */
        act[n] = effective(s, t, competing_total(s, t->path));
        n++;
    }
    /* Room in the active set is not permission to co-occur: an
     * incompatible pair must be separated even when the count is under
     * the bound, which is the whole point of bounding by class as well.
     * With no conflicts declared this is the old early return. */
    int any_conflict = 0;
    for (int c = 0; c < LOCUS_CLASS_MAX; c++)
        if (s->cfg.conflicts[c]) {
            any_conflict = 1;
            break;
        }
    if (n <= s->cfg.active_k && !any_conflict)
        return;
    int width = s->cfg.active_k < n ? s->cfg.active_k : n;

    locus_kwta_noisy(act, n, width, s->cfg.kwta_temp, &s->rng, win);

    /* Activation decides who competes; class decides who may co-occur.
     * Winners are admitted strongest first, and a winner incompatible with
     * one already admitted loses its place rather than displacing it --
     * otherwise the bound would be on count alone, which is what
     * doc/core/CONSTITUTION.md names as missing. */
    uint32_t held = 0;
    for (int done = 0; done < n; done++) {
        int best = -1;
        for (int j = 0; j < n; j++)
            if (win[j] == 1 && (best < 0 || act[j] > act[best]))
                best = j;
        if (best < 0)
            break;
        uint8_t klass = s->slots[idx[best]].klass;
        if (conflicts_with(s, klass, held))
            win[best] = 0; /* admitted rival already holds an incompatible class */
        else {
            win[best] = 2; /* admitted */
            if (klass < LOCUS_CLASS_MAX)
                held |= 1u << klass;
        }
    }

    for (int j = 0; j < n; j++)
        if (win[j] != 2)
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

    }

    /* RESIDENCY IS A RANK, NOT A LEVEL. This was a fixed floor of 0.05,
     * which means something different in a store of 72,886 units and one
     * of 191,912 -- the same size-dependence measured in the sleep
     * trigger, where the interval between consolidations grows 57x
     * across the learning curve. A trace is not weak in the abstract; it
     * is weak COMPARED WITH WHAT IT COMPETES WITH, which is the header's
     * own position that capacity is interference rather than bytes.
     *
     * So the boundary is the mean activation of the unpinned traces that
     * are still resident -- computed from the store each tick, carrying
     * no constant of its own, and moving with the population the way
     * synaptic scaling normalises against a neuron's own inputs. An
     * empty population demotes nothing, since a mean of nothing is not a
     * boundary. */
    double won[3] = { 0.0, 0.0, 0.0 };

    /* WHO WINS, settled strongest-first, PER PATHWAY. Each candidate is weighed
     * against the activation already admitted, so admitting a winner
     * raises the bar for the next -- which is what makes the set size
     * a consequence of interference rather than of active_k. */
    uint8_t promoted[LOCUS_KWTA_MAX] = { 0 };
    uint8_t done[3] = { 0, 0, 0 };
    for (int round = 0; round < s->capacity; round++) {
        int best = -1;
        for (int i = 0; i < s->capacity && i < LOCUS_KWTA_MAX; i++) {
            const Slot *t = &s->slots[i];
            if (!t->used || t->pinned || promoted[i])
                continue;
            if (t->tier == LOCUS_TIER_ARCHIVE || done[t->path])
                continue;
            if (best < 0 || t->activation > s->slots[best].activation)
                best = i;
        }
        if (best < 0)
            break;
        const Slot *b = &s->slots[best];
        LocusPathway p = b->path;
        /* Coupled by the same phasic term: during an interrupt the
         * winners of every pool raise the bar for the next candidate,
         * so a salient trace can take a slot the foreground held. */
        double against = won[p];
        for (int j = 0; j < 3; j++)
            if (j != (int)p)
                against += s->phasic * won[j];
        if (b->activation / (1.0 + s->cfg.beta * against) < 1.0) {
            /* This pathway is full; the others are still open. */
            done[p] = 1;
            continue;
        }
        promoted[best] = 1;
        won[p] += b->activation;
    }

    double sum = 0.0, lo = 0.0, hi = 0.0;
    int live = 0;
    for (int i = 0; i < s->capacity; i++) {
        const Slot *t = &s->slots[i];
        if (t->used && !t->pinned && t->tier != LOCUS_TIER_ARCHIVE) {
            double e = effective(s, t,
                                 competing_total(s, t->path));
            if (!live || e < lo)
                lo = e;
            if (!live || e > hi)
                hi = e;
            sum += e;
            live++;
        }
    }
    double boundary = live ? sum / live : 0.0;
    /* A UNIFORM POPULATION OUTCOMPETES NOBODY, and this is not a
     * refinement -- without it the store archives ITSELF. sum/live is
     * not exactly the common value when live does not divide the sum
     * exactly, so eight identical traces can each sit one ulp below
     * their own mean and every one of them demotes. Measured: eight
     * traces at activation 1.8424 all archived, while four identical
     * traces survived because dividing by four is exact.
     *
     * Comparing spread instead of trusting the mean needs no epsilon:
     * if the strongest and the weakest are the same, no trace is weak
     * relative to what it competes with. */
    int spread = live && hi > lo;

    for (int i = 0; i < s->capacity; i++) {
        Slot *t = &s->slots[i];
        if (!t->used || t->pinned)
            continue;
        /* Promotion is competed, and competition SETTLES: a trace is
         * measured against the winners so far, strongest first, not
         * against every other candidate at once. Measured against all
         * at once, eight equally cued traces each fell below the unit
         * and NONE promoted -- a crowd silencing itself, which is the
         * opposite of what lateral inhibition does. Settling gives the
         * seven that fit and refuses the eighth. */
        /* SAME SCALE AS THE BOUNDARY, which is the whole point of a
         * rank rule. This read the winners' accumulated total while
         * the boundary was the mean of population-relative values --
         * two denominators, one comparison, and it ARCHIVED THE
         * STRONGEST TRACE: measured, 3.542 archived while 2.55 stayed
         * active. `won` belongs to promotion, where the bar rises as
         * winners are admitted; residency is relative to the
         * population. */
        double e = effective(s, t, competing_total(s, t->path));
        if (t->tier == LOCUS_TIER_EPISODIC && promoted[i])
            t->tier = LOCUS_TIER_ACTIVE;
        else if (spread && e < boundary &&
                 t->tier != LOCUS_TIER_ARCHIVE && t->leases == 0)
            t->tier = LOCUS_TIER_ARCHIVE;
    }
    enforce_kwta(s);
    remember_active_set(s);
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

double locus_trace_heat(const LocusStore *s, LocusKey key)
{
    if (!s)
        return 0.0;
    int i = find(s, key);
    return i < 0 ? 0.0 : s->slots[i].heat;
}

int locus_trace_pinned(const LocusStore *s, LocusKey key)
{
    int i = s ? find(s, key) : -1;
    return i < 0 ? -1 : s->slots[i].pinned;
}

int locus_set_class(LocusStore *s, LocusKey key, uint8_t klass)
{
    if (!s || klass >= LOCUS_CLASS_MAX)
        return -1;
    for (int i = 0; i < s->capacity; i++)
        if (s->slots[i].used && s->slots[i].key == key) {
            s->slots[i].klass = klass;
            return 0;
        }
    return -1;
}

int locus_class_of(const LocusStore *s, LocusKey key)
{
    if (!s)
        return -1;
    for (int i = 0; i < s->capacity; i++)
        if (s->slots[i].used && s->slots[i].key == key)
            return (int)s->slots[i].klass;
    return -1;
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
