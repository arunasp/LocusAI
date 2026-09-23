"""Property tests for the simultaneous-update field.

The first test is the load-bearing one. Everything else here is ordinary
dynamics checking; SIMULTANEITY is the claim that distinguishes this from
the staged pipeline it replaces, and the only way to test it is to
permute the order units are computed in and require the result to be
bit-identical. An accidental in-place update -- writing into the same
list being read -- turns the field into a sequential sweep whose outcome
depends on index order, and it would otherwise look completely normal.

The rest exist because each corresponds to a defect already measured in
this project and reported as a result before being caught:

  no_dieoff          purely subtractive inhibition killed the whole field
                     in 11-19 steps at every beta > 0 tested. Shunting
                     must not be able to do that at any strength.
  gain_has_effect    global divisive normalisation erased the loop gain
                     so completely that three values of g produced
                     byte-identical output, voiding that whole sweep.
  graded_response    a constant offset plus a hard top-k gives a step,
                     and the step was mistaken for dynamics. Response to
                     injection size must be graded, not binary.
  injection_is_event re-applying a boost every step is a CLAMP, not
                     persistence; an earlier run measured a clamp and
                     called it a retained goal.
  active_is_a_read   the active set must not be a stage with side
                     effects, and must keep the sub-threshold values.
  share_not_logslope share is dimensionless and defined even when the
                     field level drifts; the log-slope it replaces never
                     settled and produced a retracted estimate.
"""

import os
import random
import sys
import unittest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "py"))
sys.path.insert(0, os.path.dirname(__file__))

import watchdog as wd  # noqa: E402
from locus.field import INF, Field  # noqa: E402
from locus.plasticity import Plasticity  # noqa: E402
from fixtures import ring_kernel            # noqa: E402


class Simultaneity(unittest.TestCase):

    def test_update_order_cannot_matter(self):
        # THE test. If any unit reads another unit's UPDATED value, the
        # result depends on the order of computation and the field is a
        # sequential sweep wearing a parallel name.
        n = 24
        k = ring_kernel(n, 1)
        base = [0.03 * (i + 1) for i in range(n)]
        orders = [list(range(n)), list(reversed(range(n)))]
        shuffled = list(range(n))
        random.Random(99).shuffle(shuffled)
        orders.append(shuffled)

        results = []
        for order in orders:
            f = Field(n, k)
            f.set_state(base)
            for _ in range(5):
                f.step(order=order)
            results.append(list(f.a))

        for other in results[1:]:
            for j in range(n):
                self.assertEqual(
                    results[0][j], other[j],
                    msg="state %d differs by update order: %r vs %r -- "
                        "the update is sequential, not simultaneous"
                        % (j, results[0][j], other[j]),
                )

    def test_step_does_not_alias_previous_state(self):
        # A returned state that IS the internal list would let a caller
        # corrupt the field, and would also mean the update wrote in
        # place.
        f = Field(6, ring_kernel(6, 2))
        f.inject(0, 1.0)
        out = f.step()
        out[0] = 999.0
        self.assertNotEqual(f.a[0], 999.0)


class Dynamics(unittest.TestCase):

    def test_no_dieoff_at_any_inhibition(self):
        # Subtractive inhibition emptied the field entirely. Shunting
        # divides the drive instead, so it must never be able to.
        n = 20
        k = ring_kernel(n, 3)
        for beta in (0.0, 0.05, 0.2, 1.0, 5.0):
            f = Field(n, k, beta=beta)
            f.set_state([0.05] * n)
            f.inject(0, 1.0)
            for _ in range(40):
                pass
            for _ in range(40):
                f.step()
            self.assertGreater(
                f.total(), 0.0,
                msg="field died at beta=%s -- inhibition is subtracting, "
                    "not shunting" % beta,
            )

    def test_gain_has_effect(self):
        # Three values of g once produced byte-identical output.
        n = 20
        k = ring_kernel(n, 4)
        finals = []
        for g in (0.2, 0.5, 0.9):
            f = Field(n, k, g=g)
            f.inject(0, 1.0)
            for _ in range(20):
                f.step()
            finals.append(round(f.total(), 9))
        self.assertEqual(len(set(finals)), 3,
                         "lateral gain is inert: %r" % (finals,))

    def test_self_excitation_has_effect(self):
        n = 20
        k = ring_kernel(n, 5)
        finals = []
        for rho in (0.5, 1.0, 1.5):
            f = Field(n, k, rho=rho)
            f.inject(0, 1.0)
            for _ in range(20):
                f.step()
            finals.append(round(f.share(0), 9))
        self.assertEqual(len(set(finals)), 3,
                         "self-excitation is inert: %r" % (finals,))

    def test_graded_response_to_injection(self):
        # Not a step function. A hard top-k produced one and it was
        # reported as a property of the dynamics.
        n = 20
        k = ring_kernel(n, 6)
        shares = []
        for amount in (0.05, 0.1, 0.2, 0.4, 0.8):
            f = Field(n, k)
            f.set_state([0.02] * n)
            f.inject(0, amount)
            for _ in range(10):
                f.step()
            shares.append(round(f.share(0), 6))
        self.assertGreater(len(set(shares)), 2,
                           "response is binary, not graded: %r" % (shares,))
        for a, b in zip(shares, shares[1:]):
            self.assertLessEqual(
                a, b + 1e-9,
                "more injection gave less share: %r" % (shares,))

    def test_leak_only_decays(self):
        # With no lateral drive and no self-excitation, leak must be the
        # only term and the state must fall monotonically.
        f = Field(8, ring_kernel(8, 7), rho=0.0, g=0.0, leak=0.5)
        f.inject(3, 1.0)
        prev = f.a[3]
        for _ in range(6):
            f.step()
            self.assertLess(f.a[3], prev + 1e-12)
            prev = f.a[3]
        self.assertLess(prev, 0.2, "leak did not decay the state")

    def test_injection_is_an_event_not_a_clamp(self):
        # Nothing re-applies an injection. Measuring a clamp and calling
        # it persistence is an error already made here.
        f = Field(12, ring_kernel(12, 8), rho=0.0, g=0.0, leak=0.9)
        f.inject(5, 1.0)
        start = f.a[5]
        for _ in range(10):
            f.step()
        self.assertLess(f.a[5], start * 0.5,
                        "state held without support -- something is "
                        "re-injecting")


class Reads(unittest.TestCase):

    def test_active_is_a_read(self):
        f = Field(10, ring_kernel(10, 9))
        f.inject(0, 1.0)
        f.step()
        before = list(f.a)
        steps_before = f.steps
        got = f.active(0.0)
        self.assertEqual(f.a, before, "active() mutated the field")
        self.assertEqual(f.steps, steps_before, "active() advanced time")
        self.assertTrue(all(isinstance(t, tuple) for t in got))

    def test_active_keeps_sub_threshold_values_available(self):
        # A hard top-k would discard these, and they are what an
        # eligibility trace and a deferral confidence read.
        n = 16
        f = Field(n, ring_kernel(n, 10))
        f.set_state([0.01 * (i + 1) for i in range(n)])
        f.step()
        high = f.active(0.05)
        self.assertLess(len(high), n, "threshold selected everything")
        self.assertEqual(len(f.a), n, "field lost states")
        self.assertTrue(any(v <= 0.05 for v in f.a),
                        "no sub-threshold values remain to read")

    def test_share_is_defined_and_dimensionless(self):
        n = 12
        f = Field(n, ring_kernel(n, 11))
        self.assertEqual(f.share(0), 0.0, "empty field should give 0")
        f.inject(0, 1.0)
        f.step()
        s = sum(f.share(i) for i in range(n))
        self.assertAlmostEqual(s, 1.0, places=9,
                               msg="shares do not sum to 1")

    def test_cycle_runs_fast_period_and_reports_history(self):
        n = 10
        f = Field(n, ring_kernel(n, 12))
        f.inject(0, 1.0)
        hist = f.cycle(fast=7)
        self.assertEqual(len(hist), 7, "cycle did not run 7 fast steps")
        self.assertEqual(f.steps, 7)
        self.assertEqual(f.cycles, 1)
        with self.assertRaises(ValueError):
            f.cycle(fast=0)


class BiasedCompetition(unittest.TestCase):
    """Goal maintenance as a SEPARATE circuit biasing the pool.

    Biology keeps these apart: pattern completion is recurrent attractor
    dynamics inside a pool, while a goal is sustained by another
    population and projects a biasing input in. The earlier measurement
    that found an injected node inside its own attractor only 14% of the
    time was asking a competitor whether it survived competition -- the
    expected answer, and not a test of whether a goal can steer.
    """

    def test_bias_steers_which_attractor_forms(self):
        n = 24
        k = ring_kernel(n, 20)

        def settle(bias_at=None, strength=0.0):
            f = Field(n, k, beta=0.3)
            f.set_state([0.02] * n)
            f.inject(0, 1.0)
            if bias_at is not None:
                b = [0.0] * n
                b[bias_at] = strength
                f.set_bias(b)
            for _ in range(80):
                f.step()
            return [f.share(i) for i in range(n)]

        free = settle()
        steered = settle(bias_at=12, strength=0.5)
        moved = max(abs(a - b) for a, b in zip(free, steered))
        self.assertGreater(
            moved, 1e-3,
            "bias changed nothing -- top-down input is not reaching the "
            "competition",
        )
        self.assertGreater(
            steered[12], free[12],
            "the biased state did not gain share relative to unbiased",
        )

    def test_bias_is_persistent_not_consumed(self):
        # inject() is an event; set_bias() is a standing input. If the
        # bias were consumed this would decay like an injection.
        # Checked below a_max so this cannot pass by saturating.
        n = 12
        f = Field(n, ring_kernel(n, 21), rho=0.0, g=0.0, leak=0.9)
        b = [0.0] * n
        b[4] = 1.0
        f.set_bias(b)
        for _ in range(30):
            f.step()
        held = f.a[4]
        self.assertGreater(
            held, 0.1,
            "a standing bias decayed away -- it is being consumed like "
            "an injection",
        )
        self.assertLess(held, f.a_max * 0.9,
                        "saturated: this would pass without the bias "
                        "being persistent at all")
        f.clear_bias()
        for _ in range(30):
            f.step()
        self.assertLess(f.a[4], held * 0.5,
                        "clearing the bias did not release the state")

    def test_bias_competes_rather_than_overriding(self):
        # A bias that ignored normalisation would be a clamp wearing a
        # different name. TWO distinct properties, and conflating them
        # is what made the first version of this test wrong.
        #
        # (a) ABSOLUTE activation must FALL as inhibition rises. That is
        #     what says the top-down input is being divided like every
        #     other input rather than bypassing the competition.
        # (b) SHARE RISES, and that is correct rather than a failure.
        #     The biased unit carries the largest drive, so raising beta
        #     suppresses every OTHER unit proportionally more. Sharper
        #     competition, larger winner -- the mechanism that produces
        #     winner-take-all at all.
        #
        # The first version asserted share would fall. It also read
        # 10.0, 10.0, 10.0 -- a_max at every level -- because a bias of
        # 1.0 into a field starting at 0.02 saturates regardless, so it
        # was measuring the CEILING. Third instrument-not-model defect
        # of the session, which is why watchdog.in_responsive_range now
        # exists and is used here.
        n = 20
        k = ring_kernel(n, 22)
        betas = (0.05, 1.0, 20.0)
        absolute, shares = [], []
        for beta in betas:
            f = Field(n, k, beta=beta)
            f.set_state([0.02] * n)
            b = [0.0] * n
            b[7] = 0.05
            f.set_bias(b)
            for _ in range(40):
                f.step()
            absolute.append(f.a[7])
            shares.append(f.share(7))

        status, detail = wd.in_responsive_range(absolute, lo=0.0,
                                                hi=Field(1, [[1.0]]).a_max)
        self.assertEqual(status, "pass",
                         "activation is pinned, so the sweep measures a "
                         "bound rather than the bias: %s" % detail)
        self.assertGreater(
            absolute[0], absolute[-1],
            "absolute activation did not fall with inhibition -- the "
            "bias is bypassing normalisation: %r" % (absolute,))
        self.assertLess(
            shares[0], shares[-1],
            "share did not rise with inhibition -- divisive "
            "normalisation is not sharpening the competition: %r"
            % (shares,))

    def test_zero_bias_is_a_no_op(self):
        # Regression: adding the pathway must not change behaviour when
        # it is unused.
        n = 16
        k = ring_kernel(n, 23)
        a = Field(n, k)
        a.inject(0, 1.0)
        b = Field(n, k)
        b.inject(0, 1.0)
        b.set_bias([0.0] * n)
        for _ in range(20):
            a.step()
            b.step()
        for j in range(n):
            self.assertEqual(a.a[j], b.a[j],
                             "an all-zero bias changed the dynamics")

    def test_bias_length_is_checked(self):
        f = Field(4, ring_kernel(4, 24))
        with self.assertRaises(ValueError):
            f.set_bias([0.1, 0.2])


class PrecisionUnderBias(unittest.TestCase):
    """A bias must steer, never overrule. If top-down input can win
    regardless of bottom-up evidence, the system sees what it expects
    rather than what is there -- which is hallucination, not attention.

    The hazard is measured, not hypothetical: at beta=20 a biased unit
    took 92% of the field. The biological operating range found here is
    beta between 0.1 and 0.5 (participation ratio 2.93% down to 0.54%,
    against cortex at 1-2%), so these tests run in that range and check
    that evidence still wins there.

    Evidence is SUSTAINED, not injected once. That is the fair
    comparison: the world keeps providing sensory input, and a standing
    bias against a one-shot injection is a race a persistent signal
    always eventually wins. Comparing a transient against a sustained
    input would measure the asymmetry, not the arbitration.
    """

    def _settle(self, n, k, evidence_at, evidence, bias_at, bias,
                beta, steps=60):
        f = Field(n, k, beta=beta)
        f.set_state([0.02] * n)
        b = [0.0] * n
        if bias_at is not None:
            b[bias_at] = bias
        f.set_bias(b)
        for _ in range(steps):
            if evidence_at is not None:
                f.inject(evidence_at, evidence)
            f.step()
        return f

    def test_evidence_outweighs_equal_bias(self):
        # Equal strength, opposite targets: evidence must win. If a bias
        # of the same magnitude as the evidence already dominates, there
        # is no headroom at all.
        n = 24
        k = ring_kernel(n, 30)
        for beta in (0.1, 0.3, 0.5):
            f = self._settle(n, k, evidence_at=3, evidence=0.2,
                             bias_at=15, bias=0.2, beta=beta)
            ev, bi = f.share(3), f.share(15)
            self.assertGreater(
                ev, bi,
                "at beta=%s an equal bias beat the evidence (%.4f vs "
                "%.4f) -- top-down input is overruling, not steering"
                % (beta, ev, bi),
            )

    def test_bias_must_exceed_evidence_to_override(self):
        # Find the crossover and require it above parity. A crossover
        # BELOW 1.0 would mean a weaker expectation beats stronger
        # evidence, which is the failure this class exists for.
        n = 24
        k = ring_kernel(n, 31)
        evidence = 0.2
        crossover = None
        ratios = [0.25, 0.5, 0.75, 1.0, 1.5, 2.0, 3.0, 5.0]
        shares = []
        for r in ratios:
            f = self._settle(n, k, evidence_at=3, evidence=evidence,
                             bias_at=15, bias=evidence * r, beta=0.3)
            shares.append((r, f.share(3), f.share(15)))
            if crossover is None and f.share(15) > f.share(3):
                crossover = r
        status, detail = wd.in_responsive_range(
            [b for _, _, b in shares], lo=0.0)
        self.assertEqual(status, "pass",
                         "bias share is pinned, so the sweep measures a "
                         "bound rather than arbitration: %s" % detail)
        self.assertIsNotNone(
            crossover,
            "bias never overrode evidence even at 5x -- the sweep does "
            "not reach the crossover, so it is unmeasured: %r" % (shares,))
        self.assertGreater(
            crossover, 1.0,
            "a bias WEAKER than the evidence overrode it at ratio %s: "
            "%r" % (crossover, shares),
        )

    def test_derived_bias_cannot_point_at_unlearned_structure(self):
        # THE fix for a measured failure. A hand-set bias at double
        # strength on an unsupported state beat sustained evidence on a
        # supported pattern, 0.588 against 0.406 -- expectation
        # manufacturing structure the substrate never formed.
        #
        # The correction is not a parameter. A bias should be DRIVEN
        # FROM OUTSIDE: learned from outcomes, so it can only point
        # where outcomes have actually built something. A state with no
        # learned incoming weight then gets exactly zero by
        # construction, and no tuning can make it otherwise.
        n = 20
        p = Plasticity(rate=1.0)
        # Learning happens only where activity co-occurred AND an
        # outcome arrived: states 0-3 together, nothing at 17.
        pattern = [0.0] * n
        for i in range(4):
            pattern[i] = 1.0
        p.observe(pattern)
        p.consolidate(1.0)

        bias = p.projected_bias(n, sources=range(4), strength=0.4)
        self.assertGreater(max(bias), 0.0, "nothing was learned at all")
        self.assertEqual(
            bias[17], 0.0,
            "a derived bias pointed at a state with no learned weight "
            "-- expectation can still manufacture structure",
        )
        for j in range(4, n):
            self.assertEqual(bias[j], 0.0,
                             "bias leaked to unlearned state %d" % j)

    def test_bad_outcome_suppresses_the_bias_it_built(self):
        # Suppression needs no separate rule: a bad outcome is a
        # negative modulator, which depresses the same tagged pairs,
        # which flips the derived bias negative.
        n = 12
        pattern = [0.0] * n
        for i in range(3):
            pattern[i] = 1.0

        good = Plasticity(rate=1.0)
        good.observe(pattern)
        good.consolidate(1.0)
        bad = Plasticity(rate=1.0)
        bad.observe(pattern)
        bad.consolidate(-1.0)

        gb = good.projected_bias(n, sources=range(3), strength=0.4)
        bb = bad.projected_bias(n, sources=range(3), strength=0.4)
        self.assertGreater(max(gb), 0.0)
        self.assertLess(min(bb), 0.0,
                        "a bad outcome did not produce a negative bias")

    def test_derived_bias_steers_the_field_it_learned_from(self):
        # Having established it cannot point anywhere unlearned, it must
        # still actually steer where it CAN point -- otherwise the
        # guarantee is bought by making the mechanism inert.
        n = 20
        k = ring_kernel(n, 32)
        p = Plasticity(rate=1.0)
        pattern = [0.0] * n
        for i in (8, 9, 10):
            pattern[i] = 1.0
        p.observe(pattern)
        p.consolidate(1.0)
        bias = p.projected_bias(n, sources=(8, 9, 10), strength=0.2)

        free = self._settle(n, k, evidence_at=0, evidence=0.2,
                            bias_at=None, bias=0.0, beta=0.3)
        f = Field(n, k, beta=0.3)
        f.set_state([0.02] * n)
        f.set_bias(bias)
        for _ in range(60):
            f.inject(0, 0.2)
            f.step()
        biased_region = sum(f.share(i) for i in (8, 9, 10))
        free_region = sum(free.share(i) for i in (8, 9, 10))
        self.assertGreater(
            biased_region, free_region,
            "a derived bias did not steer toward what it learned "
            "(%.4f vs %.4f) -- the guarantee was bought by making the "
            "mechanism inert" % (biased_region, free_region),
        )


class SpontaneousActivity(unittest.TestCase):
    """Noise as the mechanism a kernel restart was standing in for.

    Cortex is never silent. A uniform restart mixed into the kernel was
    tried instead and deleted: inert under measurement
    (`param_has_effect` gave 1 distinct value of 6, twice, across two
    wirings), absent from biology (input arrives on specific afferents,
    which `inject` models), and it silently removed a correct refusal by
    making every row carry mass so nothing was ever absorbing.

    The decisive difference is WHERE it acts. Noise drives the FIELD, so
    it never invents connectivity and never makes an absorbing partition
    look admissible -- which is what the last test here asserts.
    """

    def test_zero_noise_is_a_no_op(self):
        n = 16
        k = ring_kernel(n, 70)
        a = Field(n, k, noise=0.0)
        b = Field(n, k)
        a.set_state([0.02] * n)
        b.set_state([0.02] * n)
        for _ in range(20):
            a.step()
            b.step()
        self.assertEqual(a.a, b.a,
                         "noise=0 changed the dynamics")

    def test_noise_has_an_effect(self):
        # The check the deleted parameter failed.
        n = 16
        k = ring_kernel(n, 71)
        totals = []
        for nz in (0.0, 0.001, 0.01, 0.05):
            f = Field(n, k, noise=nz, seed=7)
            f.set_state([0.02] * n)
            for _ in range(40):
                f.step()
            totals.append(f.total())
        status, detail = wd.param_has_effect(totals)
        self.assertEqual(status, "pass",
                         "noise is inert: %s" % detail)
        for x, y in zip(totals, totals[1:]):
            self.assertGreater(y, x - 1e-12,
                               "more noise gave less activity: %r"
                               % (totals,))

    def test_simultaneity_survives_noise(self):
        # Samples are drawn up front in fixed index order, so a
        # permuted WRITE order cannot change which unit gets which
        # sample. Without that the noise would silently reintroduce
        # order dependence into an update that must be simultaneous.
        n = 12
        k = ring_kernel(n, 72)
        order = list(range(n))
        random.Random(1).shuffle(order)
        a = Field(n, k, noise=0.01, seed=3)
        b = Field(n, k, noise=0.01, seed=3)
        a.set_state([0.05] * n)
        b.set_state([0.05] * n)
        for _ in range(6):
            a.step()
            b.step(order=order)
        for j in range(n):
            self.assertEqual(a.a[j], b.a[j],
                             "noise made the update order-dependent "
                             "at state %d" % j)

    def test_seed_is_reproducible_and_distinguishing(self):
        n = 12
        k = ring_kernel(n, 73)

        def run(seed):
            f = Field(n, k, noise=0.02, seed=seed)
            f.set_state([0.03] * n)
            for _ in range(15):
                f.step()
            return list(f.a)

        self.assertEqual(run(5), run(5), "same seed diverged")
        self.assertNotEqual(run(5), run(6),
                            "different seeds gave identical noise")

    def test_noise_keeps_a_starved_field_alive(self):
        # Leak alone drives a field with no drive to zero. Spontaneous
        # activity is what stops any state trapping or losing all
        # activation forever -- the job the restart was meant to do.
        n = 10
        k = ring_kernel(n, 74)
        quiet = Field(n, k, rho=0.0, g=0.0, leak=0.9, noise=0.0)
        noisy = Field(n, k, rho=0.0, g=0.0, leak=0.9, noise=0.01,
                      seed=11)
        for f in (quiet, noisy):
            f.set_state([0.5] * n)
            for _ in range(60):
                f.step()
        self.assertLess(quiet.total(), 1e-6,
                        "leak did not starve the quiet field")
        self.assertGreater(noisy.total(), 1e-6,
                           "noise did not sustain any activity")

    def test_noise_cannot_make_a_kernel_admissible(self):
        # THE distinction that justified deleting the restart. Noise
        # acts on the field; the kernel is untouched, so an absorbing
        # complement stays absorbing and a coarse level over it still
        # correctly does not exist.
        n = 8
        k = [[0.0] * n for _ in range(n)]
        for i in range(n):
            k[i][i] = 1.0
        f = Field(n, k, noise=0.05, seed=2)
        f.set_state([0.1] * n)
        for _ in range(20):
            f.step()
        for i in range(n):
            self.assertEqual(
                f.kernel[i][i], 1.0,
                "noise modified the kernel at state %d -- it must "
                "drive the field only" % i)
            self.assertAlmostEqual(sum(f.kernel[i]), 1.0, places=12)


class NumbersAreCheckedNotAssumed(unittest.TestCase):
    """Shape was validated and the parameters were not, so an
    out-of-range number changed the MEANING of the dynamics rather than
    failing. Found by generalising the zero-modulator case: a guarantee
    resting on a numeric value is not a guarantee.
    """

    def kernel(self):
        return ring_kernel(8, 3)

    def test_a_negative_beta_is_refused(self):
        # denom = 1 + beta * others reaches ZERO at beta < 0, and below
        # it the drive flips sign and the rectifier floors the field.
        with self.assertRaises(ValueError):
            Field(8, self.kernel(), beta=-0.5)

    def test_every_rate_must_be_finite(self):
        for name in ("rho", "g", "beta", "leak", "dt", "a_max",
                     "floor", "noise"):
            for bad in (float("nan"), float("inf")):
                with self.assertRaises(ValueError, msg="%s=%r"
                                       % (name, bad)):
                    Field(8, self.kernel(), **{name: bad})

    def test_dt_and_a_max_must_be_positive(self):
        with self.assertRaises(ValueError):
            Field(8, self.kernel(), dt=0.0)
        with self.assertRaises(ValueError):
            Field(8, self.kernel(), a_max=0.0)

    def test_a_non_finite_injection_is_refused(self):
        # Evidence enters here; NaN would spread through the kernel on
        # the next step and every later figure would be NaN with no
        # failing step to point at.
        f = Field(8, self.kernel())
        with self.assertRaises(ValueError):
            f.inject(0, float("nan"))
        self.assertFalse(any(v != v for v in f.a))

    def test_the_denominator_cannot_reach_zero(self):
        # The property those guards protect, stated directly.
        f = Field(8, self.kernel(), beta=2.0)
        for i in range(8):
            f.inject(i, 1.0)
        for _ in range(20):
            f.step()
            self.assertTrue(all(v == v and abs(v) != INF for v in f.a))


class Construction(unittest.TestCase):

    def test_kernel_shape_is_checked(self):
        with self.assertRaises(ValueError):
            Field(4, [[0.25] * 4] * 3)
        with self.assertRaises(ValueError):
            Field(4, [[0.25] * 3] * 4)

    def test_set_state_length_is_checked(self):
        f = Field(4, ring_kernel(4, 13))
        with self.assertRaises(ValueError):
            f.set_state([0.1, 0.2])


if __name__ == "__main__":
    unittest.main()
