/* test_store.c — behavioural tests for the LocusAI trace store. */

#include "locus.h"

#include <stdio.h>
#include <string.h>

static int failures;
static int checks;

static void ok(int cond, const char *what)
{
    checks++;
    if (!cond) {
        failures++;
        printf("  FAIL %s\n", what);
    }
}

static void test_lease_and_lookup(void)
{
    LocusConfig cfg = locus_config_default();
    LocusStore *s = locus_store_create(&cfg);
    LocusView v;

    ok(locus_put(s, 1, LOCUS_PATH_DECLARATIVE, "abc", 3, 0.5) == 0, "put");
    ok(locus_lookup(s, 1, &v) == 0, "lookup hit");
    ok(v.len == 3 && memcmp(v.data, "abc", 3) == 0, "payload intact");
    ok(v.labile == 0, "lookup does not destabilise");
    locus_release(s, &v);
    ok(v.slot == NULL, "release clears view");

    ok(locus_lookup(s, 99, &v) != 0, "miss");
    ok(v.slot == NULL, "failed view cleared");

    LocusStats st;
    locus_stats(s, &st);
    ok(st.hits == 1 && st.misses == 1, "hit/miss counted");
    locus_store_destroy(s);
}

static void test_retrieve_destabilises(void)
{
    LocusConfig cfg = locus_config_default();
    LocusStore *s = locus_store_create(&cfg);
    LocusView v;

    locus_put(s, 1, LOCUS_PATH_DECLARATIVE, "old", 3, 1.0);
    ok(locus_retrieve(s, 1, &v) == 0, "retrieve hit");
    ok(v.labile == 1, "retrieve marks labile");
    locus_release(s, &v);

    ok(locus_restabilize(s, 1, "new", 3) == 0, "restabilise");
    ok(locus_restabilize(s, 1, NULL, 0) != 0, "double restabilise rejected");

    ok(locus_lookup(s, 1, &v) == 0, "reread");
    ok(memcmp(v.data, "new", 3) == 0, "recall rewrote the trace");
    locus_release(s, &v);
    locus_store_destroy(s);
}

static void test_labile_loss(void)
{
    LocusConfig cfg = locus_config_default();
    LocusStore *s = locus_store_create(&cfg);
    LocusView v;

    locus_put(s, 1, LOCUS_PATH_DECLARATIVE, "x", 1, 1.0);
    locus_retrieve(s, 1, &v);
    locus_release(s, &v);

    double before = locus_trace_activation(s, 1);
    for (int i = 0; i < 3; i++)
        locus_tick(s);

    LocusStats st;
    locus_stats(s, &st);
    ok(st.labile_lost == 1, "unrestabilised trace counted lost");
    ok(locus_trace_activation(s, 1) < before, "and weakened");
    locus_store_destroy(s);
}

static void test_kwta_bound(void)
{
    LocusConfig cfg = locus_config_default();
    cfg.active_k = 3;
    cfg.kwta_temp = 0.0; /* assert an exact winner set */
    LocusStore *s = locus_store_create(&cfg);

    for (LocusKey k = 1; k <= 8; k++)
        locus_put(s, k, LOCUS_PATH_DECLARATIVE, "y", 1, 2.0 + (double)k);
    locus_tick(s);

    LocusStats st;
    locus_stats(s, &st);
    ok(st.active_traces == 3, "active set bounded by competition");
    ok(locus_trace_tier(s, 8) == LOCUS_TIER_ACTIVE, "strongest stays active");
    ok(locus_trace_tier(s, 1) != LOCUS_TIER_ACTIVE, "weakest demoted");
    locus_store_destroy(s);
}

static void test_pinned_exempt(void)
{
    LocusConfig cfg = locus_config_default();
    cfg.active_k = 1;
    cfg.kwta_temp = 0.0;
    LocusStore *s = locus_store_create(&cfg);

    locus_put(s, 1, LOCUS_PATH_INSTINCT, "i", 1, 0.1);
    for (LocusKey k = 2; k <= 5; k++)
        locus_put(s, k, LOCUS_PATH_DECLARATIVE, "d", 1, 5.0);
    for (int i = 0; i < 20; i++)
        locus_tick(s);

    ok(locus_trace_tier(s, 1) == LOCUS_TIER_ACTIVE, "instinct never demotes");
    ok(locus_trace_pinned(s, 1) == 1, "instinct pinned");
    locus_store_destroy(s);
}

static void test_promotion(void)
{
    LocusConfig cfg = locus_config_default();
    cfg.promote_after = 3;
    LocusStore *s = locus_store_create(&cfg);

    locus_put(s, 1, LOCUS_PATH_DECLARATIVE, "h", 1, 0.2);
    ok(locus_reinforce(s, 1, 0.0) == 1, "first use");
    ok(locus_trace_pinned(s, 1) == 0, "not yet a habit");
    locus_reinforce(s, 1, 0.0);
    locus_reinforce(s, 1, 0.0);

    ok(locus_trace_pinned(s, 1) == 1, "promoted after repetition");
    ok(locus_trace_pathway(s, 1) == LOCUS_PATH_PROCEDURAL, "now procedural");
    ok(locus_trace_tier(s, 1) == LOCUS_TIER_ACTIVE, "and resident");

    for (int i = 0; i < 30; i++)
        locus_tick(s);
    ok(locus_trace_tier(s, 1) == LOCUS_TIER_ACTIVE, "habit survives disuse");
    locus_store_destroy(s);
}

static void test_tagging_and_capture(void)
{
    LocusConfig cfg = locus_config_default();
    LocusStore *s = locus_store_create(&cfg);

    locus_put(s, 1, LOCUS_PATH_DECLARATIVE, "weak", 4, 0.2);
    double before = locus_trace_activation(s, 1);

    locus_put(s, 2, LOCUS_PATH_DECLARATIVE, "strong", 6, 0.2);
    locus_note_salient(s, 2, 4.0);

    ok(locus_trace_activation(s, 1) > before, "weak trace captured");

    LocusStats st;
    locus_stats(s, &st);
    ok(st.captures == 1, "capture counted");

    /* Outside the window there is no rescue. */
    locus_put(s, 3, LOCUS_PATH_DECLARATIVE, "late", 4, 0.2);
    for (int i = 0; i < cfg.capture_window + 1; i++)
        locus_tick(s);
    double cold = locus_trace_activation(s, 3);
    locus_note_salient(s, 2, 4.0);
    ok(locus_trace_activation(s, 3) == cold, "expired tag not captured");
    locus_store_destroy(s);
}

static void test_prefetch_advisory(void)
{
    LocusConfig cfg = locus_config_default();
    LocusStore *s = locus_store_create(&cfg);
    LocusView v;
    LocusKey want[] = { 1, 42 };

    locus_put(s, 1, LOCUS_PATH_DECLARATIVE, "p", 1, 0.5);
    ok(locus_prefetch(s, want, 2) == 0, "absent key is not an error");

    locus_lookup(s, 1, &v);
    locus_release(s, &v);

    LocusStats st;
    locus_stats(s, &st);
    ok(st.prefetched == 1, "only present key prefetched");
    ok(st.prefetch_hits == 1, "prefetch hit credited");
    locus_store_destroy(s);
}

static void test_lease_blocks_eviction(void)
{
    LocusConfig cfg = locus_config_default();
    cfg.capacity = 2;
    LocusStore *s = locus_store_create(&cfg);
    LocusView v;

    locus_put(s, 1, LOCUS_PATH_DECLARATIVE, "a", 1, 0.1);
    locus_put(s, 2, LOCUS_PATH_DECLARATIVE, "b", 1, 9.0);
    locus_lookup(s, 1, &v); /* lease the weakest: it would lose on merit */

    /* The lease is absolute for its own slot; competition then decides among
     * the eligible, so the stronger unleased trace is what goes. */
    ok(locus_put(s, 3, LOCUS_PATH_DECLARATIVE, "c", 1, 1.0) == 0, "put fits");
    ok(locus_trace_tier(s, 1) >= 0, "leased slot survived");
    ok(locus_trace_tier(s, 2) < 0, "weakest eligible evicted instead");
    locus_release(s, &v);

    /* With no lease held, competition alone decides. Key 1 was accessed and
     * an access strengthens, so the never-accessed trace is now the weakest. */
    ok(locus_trace_activation(s, 1) > locus_trace_activation(s, 3),
       "access strengthened the retrieved trace");
    ok(locus_put(s, 4, LOCUS_PATH_DECLARATIVE, "d", 1, 5.0) == 0, "put fits");
    ok(locus_trace_tier(s, 3) < 0, "never-accessed trace evicted");
    ok(locus_trace_tier(s, 1) >= 0, "accessed trace retained on merit");
    locus_store_destroy(s);
}

static void test_kwta_primitive(void)
{
    double act[6] = { 0.1, 0.9, 0.4, 0.8, 0.2, 0.7 };
    uint8_t win[6];

    locus_kwta(act, 6, 3, win);
    ok(win[1] && win[3] && win[5], "top three win");
    ok(!win[0] && !win[2] && !win[4], "rest lose");

    locus_kwta(act, 6, 0, win);
    ok(!win[1], "k=0 selects nothing");
}


static void test_lease_expiry_frees_the_slot(void)
{
    LocusConfig cfg = locus_config_default();
    cfg.capacity = 1;
    cfg.lease_max_ticks = 3;
    LocusStore *s = locus_store_create(&cfg);
    LocusView v;

    locus_put(s, 1, LOCUS_PATH_DECLARATIVE, "held", 4, 1.0);
    locus_lookup(s, 1, &v); /* leaked: never released */

    ok(locus_put(s, 2, LOCUS_PATH_DECLARATIVE, "new", 3, 1.0) != 0,
       "leased slot blocks encoding while the lease is live");

    for (int i = 0; i <= cfg.lease_max_ticks; i++)
        locus_tick(s);

    LocusStats st;
    locus_stats(s, &st);
    ok(st.leases_revoked == 1, "expired lease revoked by tick");
    ok(locus_put(s, 2, LOCUS_PATH_DECLARATIVE, "new", 3, 1.0) == 0,
       "encoding no longer deadlocked");

    /* Releasing a revoked view is safe and must not corrupt the new tenant. */
    locus_release(s, &v);
    ok(locus_trace_tier(s, 2) >= 0, "stale release did not disturb the slot");
    locus_store_destroy(s);
}

static void test_release_after_reuse_is_a_noop(void)
{
    LocusConfig cfg = locus_config_default();
    cfg.capacity = 1;
    LocusStore *s = locus_store_create(&cfg);
    LocusView v, w;

    locus_put(s, 1, LOCUS_PATH_DECLARATIVE, "a", 1, 1.0);
    locus_lookup(s, 1, &v);
    locus_release(s, &v);          /* slot now unleased */
    locus_lookup(s, 1, &v);        /* lease again */
    locus_release(s, &v);
    locus_put(s, 1, LOCUS_PATH_DECLARATIVE, "b", 1, 1.0); /* reuse, gen bump */

    locus_lookup(s, 1, &w);
    locus_release(s, &v); /* stale view from before the reuse */
    locus_release(s, &w);
    ok(locus_put(s, 2, LOCUS_PATH_DECLARATIVE, "c", 1, 5.0) == 0,
       "stale release did not underflow the lease count");
    locus_store_destroy(s);
}

static void test_surprise_gates_promotion(void)
{
    LocusConfig cfg = locus_config_default();
    cfg.promote_after = 3;
    LocusStore *s = locus_store_create(&cfg);

    locus_put(s, 1, LOCUS_PATH_DECLARATIVE, "h", 1, 0.2);
    for (int i = 0; i < 5; i++)
        locus_reinforce(s, 1, 0.9); /* still surprising */
    ok(locus_trace_reps(s, 1) == 5, "repetitions counted");
    ok(locus_trace_pinned(s, 1) == 0,
       "repetition alone does not promote while outcomes surprise");

    for (int i = 0; i < 3; i++)
        locus_reinforce(s, 1, 0.0); /* prediction error gone */
    ok(locus_trace_pinned(s, 1) == 1, "promoted once outcomes stopped surprising");
    locus_store_destroy(s);
}

static void test_surprise_breaks_the_run(void)
{
    LocusConfig cfg = locus_config_default();
    cfg.promote_after = 3;
    LocusStore *s = locus_store_create(&cfg);

    locus_put(s, 1, LOCUS_PATH_DECLARATIVE, "h", 1, 0.2);
    locus_reinforce(s, 1, 0.0);
    locus_reinforce(s, 1, 0.0);
    locus_reinforce(s, 1, 0.8); /* surprise resets the calm run */
    ok(locus_trace_pinned(s, 1) == 0, "a surprising outcome breaks the run");
    locus_reinforce(s, 1, 0.0);
    locus_reinforce(s, 1, 0.0);
    ok(locus_trace_pinned(s, 1) == 0, "run must restart from zero");
    locus_reinforce(s, 1, 0.0);
    ok(locus_trace_pinned(s, 1) == 1, "promoted after a fresh calm run");
    locus_store_destroy(s);
}

static void test_gate_blocks_the_update_not_the_trace(void)
{
    LocusConfig cfg = locus_config_default();
    cfg.gate_tonic = 0.0; /* below threshold: nothing to open the gate */
    LocusStore *s = locus_store_create(&cfg);
    LocusView v;

    locus_put(s, 1, LOCUS_PATH_DECLARATIVE, "old", 3, 1.0);
    locus_retrieve(s, 1, &v);
    locus_release(s, &v);

    ok(locus_restabilize(s, 1, "new", 3) == LOCUS_GATED, "update gated");
    ok(locus_lookup(s, 1, &v) == 0, "trace still stabilised and readable");
    ok(memcmp(v.data, "old", 3) == 0, "prior content retained");
    locus_release(s, &v);

    LocusStats st;
    locus_stats(s, &st);
    ok(st.updates_gated == 1, "gated update counted");
    locus_store_destroy(s);
}

static void test_salience_opens_the_gate(void)
{
    LocusConfig cfg = locus_config_default();
    cfg.gate_tonic = 0.0;
    LocusStore *s = locus_store_create(&cfg);
    LocusView v;

    locus_put(s, 1, LOCUS_PATH_DECLARATIVE, "old", 3, 1.0);
    locus_note_salient(s, 1, 2.0); /* phasic tone lifts above threshold */
    ok(locus_gate(s) >= cfg.gate_threshold, "tone raised by salience");

    locus_retrieve(s, 1, &v);
    locus_release(s, &v);
    ok(locus_restabilize(s, 1, "new", 3) == LOCUS_RESTABILIZED, "update written");

    locus_lookup(s, 1, &v);
    ok(memcmp(v.data, "new", 3) == 0, "salient recall rewrote the trace");
    locus_release(s, &v);
    locus_store_destroy(s);
}

static void test_noisy_kwta_is_a_draw_not_a_ranking(void)
{
    double act[4] = { 1.0, 1.05, 1.1, 1.15 };
    uint8_t win[4];
    uint32_t rng = 12345u;
    int weakest_won = 0;

    for (int trial = 0; trial < 200; trial++) {
        locus_kwta_noisy(act, 4, 2, 0.5, &rng, win);
        if (win[0])
            weakest_won++;
    }
    ok(weakest_won > 0, "close contenders sometimes win under noise");
    ok(weakest_won < 200, "and do not always win");

    locus_kwta_noisy(act, 4, 2, 0.0, &rng, win);
    ok(win[2] && win[3], "temp 0 collapses to deterministic top-k");
}

static void test_noisy_kwta_is_reproducible(void)
{
    double act[5] = { 0.2, 0.4, 0.6, 0.8, 1.0 };
    uint8_t a[5], b[5];
    uint32_t r1 = 999u, r2 = 999u;

    locus_kwta_noisy(act, 5, 2, 0.7, &r1, a);
    locus_kwta_noisy(act, 5, 2, 0.7, &r2, b);
    ok(memcmp(a, b, sizeof(a)) == 0, "same seed gives the same draw");
}

int main(void)
{
    printf("locus store tests\n");
    test_lease_and_lookup();
    test_retrieve_destabilises();
    test_labile_loss();
    test_kwta_bound();
    test_pinned_exempt();
    test_promotion();
    test_tagging_and_capture();
    test_prefetch_advisory();
    test_lease_blocks_eviction();
    test_kwta_primitive();
    test_lease_expiry_frees_the_slot();
    test_release_after_reuse_is_a_noop();
    test_surprise_gates_promotion();
    test_surprise_breaks_the_run();
    test_gate_blocks_the_update_not_the_trace();
    test_salience_opens_the_gate();
    test_noisy_kwta_is_a_draw_not_a_ranking();
    test_noisy_kwta_is_reproducible();

    printf("%d checks, %d failures\n", checks, failures);
    return failures ? 1 : 0;
}
