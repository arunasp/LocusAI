"""Single-pass inhibition: can the per-area sums skip their own pass?

`Field.step` needs each area's total activation before it can update any
unit. On a GPU that is a reduction, and gfx1100 has no grid-wide barrier,
so it costs a separate launch per tick.

The sums are taken over the state at entry to the step, which is exactly
what the previous step wrote. `Carried` accumulates them during that write
and adjusts them in `inject` and `set_state`, so no step re-reduces.

`Lagged` is the negative control: it uses sums one step older, the shift
Single-Pass mHC applies. The experiment is valid only if the control is
detected; a comparison that cannot see a real lag says nothing about
`Carried`.

Run: python3 tests/exp_singlepass.py   (or `make singlepass`)
"""

import os
import random
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "py"))
sys.path.insert(0, os.path.dirname(__file__))

from locus.field import Field            # noqa: E402
from test_cycle import ring_kernel       # noqa: E402

THRESHOLD = 0.05


class Carried(Field):
    """Area sums accumulated during the previous write."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._tot = self._sums(self.a)

    def _sums(self, a):
        t = [0.0] * len(self.areas)
        for i, v in enumerate(a):
            t[self._area_of[i]] += v
        return t

    def inject(self, index, amount):
        before = self.a[index]
        super().inject(index, amount)
        self._tot[self._area_of[index]] += self.a[index] - before

    def set_state(self, values):
        super().set_state(values)
        self._tot = self._sums(self.a)

    def step(self, order=None):
        a = self.a
        drive = self._drive(a)
        spont = ([self.noise * self._rng.random() for _ in range(self.n)]
                 if self.noise > 0.0 else None)
        area_total = self._tot
        internal = 0.0
        nxt = [0.0] * self.n
        new_tot = [0.0] * len(self.areas)
        for j in (range(self.n) if order is None else order):
            aj = a[j]
            others = area_total[self._area_of[j]] - aj
            denom = 1.0 + self.beta * others
            net = (self.rho * aj + self.g * drive[j] + self.bias[j]) / denom
            if spont is not None:
                net += spont[j]
            internal += abs(net)
            v = aj + self.dt * (net - self.leak * aj)
            v = 0.0 if v < 0.0 else min(v, self.a_max)
            nxt[j] = v
            new_tot[self._area_of[j]] += v
        ext = self._pending_input
        self._pending_input = 0.0
        share = (ext / (ext + internal)) if (ext + internal) > 0 else 0.0
        self._input_share = ((1.0 - self.input_alpha) * self._input_share
                             + self.input_alpha * share)
        self.a = nxt
        self._tot = new_tot
        self.steps += 1
        return list(nxt)


class Lagged(Carried):
    """Negative control: sums from one step older."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._old = list(self._tot)

    def step(self, order=None):
        cur = self._tot
        self._tot = self._old
        out = super().step(order)
        self._old = cur
        return out


def run(cls, n, seed, beta, areas, noise, steps=300, every=25):
    f = cls(n, ring_kernel(n, seed), beta=beta, areas=areas, noise=noise,
            seed=seed)
    rnd = random.Random(seed + 1)
    for t in range(steps):
        if t % every == 0:
            f.inject(rnd.randrange(n), 0.2 + rnd.random())
        f.step()
    return f.a


def support(a):
    return {i for i, v in enumerate(a) if v > THRESHOLD}


def compare(cls):
    cases = exact = mismatch = 0
    worst = 0.0
    for n in (16, 32, 64):
        for seed in range(5):
            for beta in (0.3, 1.0, 3.0):
                for split in (1, 4):
                    areas = (None if split == 1 else
                             [list(range(i, n, split)) for i in range(split)])
                    for noise in (0.0, 0.01):
                        a = run(Field, n, seed, beta, areas, noise)
                        b = run(cls, n, seed, beta, areas, noise)
                        worst = max(worst,
                                    max(abs(x - y) for x, y in zip(a, b)))
                        cases += 1
                        exact += (a == b)
                        mismatch += (support(a) != support(b))
    return cases, exact, worst, mismatch


def main():
    cases, exact, worst, mismatch = compare(Carried)
    print("carried  cases=%d bit_identical=%d worst_abs_diff=%.3e "
          "support_mismatch=%d" % (cases, exact, worst, mismatch))
    c_cases, _, c_worst, c_mismatch = compare(Lagged)
    print("lagged   cases=%d worst_abs_diff=%.3e support_mismatch=%d "
          "(negative control)" % (c_cases, c_worst, c_mismatch))
    if c_mismatch == 0:
        print("INVALID: the control was not detected")
        return 2
    if mismatch:
        print("FAIL: carried sums changed a settled support")
        return 1
    print("PASS: carried sums preserve every settled support")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
