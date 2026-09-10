#!/usr/bin/env python3
"""Verify caps.cpp and persist.cpp branching against a mocked HIP runtime.

WHY THIS EXISTS
caps.cpp decides what to report based on what the runtime exposes: optional
struct fields that may be absent, attributes that may be unsupported, error
codes that distinguish "not supported" from "unsupported limit", and whether
a launch actually covered the grid. Those are branches, and a branch that
has only been read is not verified. Running them needs neither a GPU nor
hipcc, so they belong in a stage that runs everywhere -- including a CI
worker with no device access, where the real `caps` target can only SKIP.

WHAT IT DOES NOT DO
It says nothing about whether the GPU works. Faked runtime, serial "launch",
host pointers. It catches "reports the wrong thing" before the code reaches
the machine; the real run is still the only evidence about the device.

Coverage follows the four-state rule -- present, variant-present, ABSENT,
and present-but-BROKEN -- because the last two are the ones usually missing:
  absent           : MOCK_OMIT_OPTIONAL (fields not in the struct at all),
                     MOCK_PHYS_ATTR_FAIL, MOCK_DEVICE_COUNT=0
  present-but-broken: MOCK_FUNCATTR_FAIL, MOCK_PARTIAL_LAUNCH,
                     MOCK_LIMIT_UNSUPPORTED

Report vocabulary is this project's ([PASS]/[FAIL]/RESULT:), not a generic
mock-library banner, so pipeline output stays greppable the same way.
"""

import os
import shutil
import subprocess
import sys
import tempfile

HERE = os.path.dirname(os.path.abspath(__file__))
GPU_DIR = os.path.dirname(HERE)
CAPS = os.path.join(GPU_DIR, "caps.cpp")
PERSIST = os.path.join(GPU_DIR, "persist.cpp")
MOCK_INC = os.path.join(GPU_DIR, "mock")

CXX = os.environ.get("CXX", "c++")

# Injecting non-temporal availability rather than requiring clang: the
# builtins are clang-only, so on a host with only g++ the PRESENT branch
# would otherwise be untestable. caps.cpp routes both through one macro pair
# precisely so this is possible -- see the injection-point note there.
NT_PRESENT = [
    "-DLOCUS_HAS_NT=1",
    "-DLOCUS_NT_LOAD(p)=(*(p))",
    "-DLOCUS_NT_STORE(v,p)=(*(p)=(v))",
]

results = []


def scenario(name, defines, want_exit, must_contain=(), must_not_contain=(),
             source=None, env=None):
    """Compile one probe under a mocked environment, run it, assert."""
    tmp = tempfile.mkdtemp(prefix="capsmock-")
    try:
        exe = os.path.join(tmp, "caps")
        log = os.path.join(tmp, "caps.log")
        cmd = [CXX, "-std=c++17", "-Wall", "-Wextra", "-I", MOCK_INC,
               "-o", exe, source or CAPS] + list(defines)
        cc = subprocess.run(cmd, capture_output=True, text=True)
        if cc.returncode != 0:
            # The FIRST "error:" line, not the last stderr line: the tail of
            # a g++ diagnostic is caret art and names nothing.
            errs = [ln for ln in cc.stderr.splitlines() if "error:" in ln]
            results.append((name, False, "did not compile: " +
                            (errs[0].strip() if errs else "no error: line")))
            return
        runenv = dict(os.environ)
        if env:
            runenv.update(env)
        run = subprocess.run([exe, log], capture_output=True, text=True,
                             env=runenv)
        text = run.stdout
        # The log must exist and match stdout's content, because the caller
        # reads the log rather than the terminal -- a probe whose log went
        # missing has produced no evidence at all.
        logged = ""
        if os.path.exists(log):
            with open(log) as fh:
                logged = fh.read()

        problems = []
        # The probe reports its own artifact size. That claim must match the
        # filesystem, not merely be present: it read 0 bytes on the first
        # run because the log was still buffered, and a substring assertion
        # would have passed that happily.
        art = [ln for ln in text.splitlines() if ln.startswith("[artifact]")]
        if not art:
            problems.append("no [artifact] line -- probe did not"
                            " enumerate its output")
        else:
            try:
                claimed = int(art[0].split()[-2])
                actual = os.path.getsize(log)
                if claimed != actual:
                    problems.append(f"artifact size claimed {claimed},"
                                    f" actual {actual}")
                if claimed == 0:
                    problems.append("artifact size reported as 0")
            except (ValueError, IndexError, OSError) as exc:
                problems.append(f"artifact line unparseable: {exc}")
        nonzero_ok = want_exit == -1 and run.returncode != 0
        if run.returncode != want_exit and not nonzero_ok:
            problems.append(f"exit {run.returncode}, wanted "
                            f"{'nonzero' if want_exit == -1 else want_exit}")
        if not logged:
            problems.append("log file empty or missing")
        elif "RESULT:" not in logged:
            problems.append("log has no RESULT line")
        for needle in must_contain:
            if needle not in text:
                problems.append(f"missing {needle!r}")
        for needle in must_not_contain:
            if needle in text:
                problems.append(f"unexpectedly present {needle!r}")
        results.append((name, not problems, "; ".join(problems) or "ok"))
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


# -- 1. the healthy case, fallback NT path (what g++ gives us) --------------
scenario("healthy, NT fallback", [], 0,
         must_contain=["RESULT: PASS", "built with: LOCUS_HAS_NT=0",
                       "nontemporal FALLBACK", "physical/logical ratio"])

# -- 2. the healthy case with NT builtins injected as present --------------
scenario("healthy, NT present (injected)", NT_PRESENT, 0,
         must_contain=["RESULT: PASS", "built with: LOCUS_HAS_NT=1",
                       "nontemporal builtins"],
         must_not_contain=["nontemporal FALLBACK"])

# -- 3. ABSENT: no device at all -------------------------------------------
# Two extra assertions so this cannot pass for the wrong reason: the output
# must NAME the missing thing, and something that should still have run must
# have run.
scenario("absent: no HIP device", ["-DMOCK_DEVICE_COUNT=0"], -1,
         must_contain=["[1] device enumeration", "no device", "RESULT: FAIL"],
         must_not_contain=["[3] memory hierarchy"])

# -- 4. ABSENT: runtime does not expose the newer struct fields ------------
# The whole point of the detector: an older ROCm must yield a report saying
# "not exposed", not a compile error.
scenario("absent: optional fields not in struct",
         ["-DMOCK_OMIT_OPTIONAL=1"], 0,
         must_contain=["RESULT: PASS",
                       "persistingL2CacheMaxSize            : -1",
                       "regsPerMultiprocessor               : -1"])

# -- 5. ABSENT: the physical-CU attribute is unsupported -------------------
# Reported, not asserted, so the probe must still pass overall.
scenario("absent: PhysicalMultiProcessorCount", ["-DMOCK_PHYS_ATTR_FAIL=1"], 0,
         must_contain=["RESULT: PASS", "hipErrorNotSupported"],
         must_not_contain=["physical/logical ratio"])

# -- 6. BROKEN: launch covered only part of the grid ----------------------
scenario("broken: partial launch", ["-DMOCK_PARTIAL_LAUNCH=1"], -1,
         must_contain=["wrong, first at index", "RESULT: FAIL"])

# -- 7. BROKEN: hipFuncGetAttributes refuses ------------------------------
scenario("broken: hipFuncGetAttributes", ["-DMOCK_FUNCATTR_FAIL=1"], -1,
         must_contain=["hipFuncGetAttributes", "hipErrorNotSupported",
                       "RESULT: FAIL"])

# -- 8. BROKEN: a limit is refused ----------------------------------------
# Expected real behaviour on AMD, so it must be reported without failing.
scenario("broken: limit unsupported", ["-DMOCK_LIMIT_UNSUPPORTED=1"], 0,
         must_contain=["RESULT: PASS", "hipErrorUnsupportedLimit"])

# -- 9. VARIANT: 64-wide wavefront (CDNA-shaped device) -------------------
# Guards against the probe hardcoding this machine.
scenario("variant: warpSize 64", ["-DMOCK_WARP_SIZE=64"], 0,
         must_contain=["RESULT: PASS",
                       "warpSize                            : 64"])

# -- 10. BROKEN: nonsense warp size and unusable LDS ----------------------
scenario("broken: warpSize 7", ["-DMOCK_WARP_SIZE=7"], -1,
         must_contain=["warpSize is 32 or 64", "RESULT: FAIL"])
scenario("broken: LDS 1 KiB", ["-DMOCK_LDS=1024"], -1,
         must_contain=["LDS per block >= 16 KiB", "RESULT: FAIL"])


# ==========================================================================
# persist.cpp -- the gated probe. Its gate and its reporting are what the
# mock can verify; the barrier-dependent behaviour and the actual watchdog
# risk are real-device-only and deliberately not asserted here.
# ==========================================================================
P = dict(source=PERSIST)

# -- 11. the gate itself: unset means SKIP, and nothing long may run -------
scenario("persist: gate unset -> SKIP", [], 0,
         must_contain=["[SKIP] LOCUS_GPU_STRESS is not 1", "RESULT: SKIP"],
         must_not_contain=["[2] LDS state", "wall duration"], **P)

# -- 12. gate set to something other than 1 must NOT run either -----------
scenario("persist: gate=0 -> SKIP", [], 0,
         must_contain=["RESULT: SKIP"],
         must_not_contain=["wall duration"],
         env={"LOCUS_GPU_STRESS": "0"}, **P)

# -- 13. healthy gated run -------------------------------------------------
scenario("persist: healthy", [], 0,
         must_contain=["RESULT: PASS", "every LDS slot survived every tick",
                       "cooperative launch accepted at full grid",
                       "wall duration"],
         env={"LOCUS_GPU_STRESS": "1", "LOCUS_GPU_STRESS_TICKS": "7"}, **P)

# -- 14. the tick count must come from the environment, not a default -----
scenario("persist: tick count honoured", [], 0,
         must_contain=["ticks  : 7"],
         must_not_contain=["ticks  : 1000"],
         env={"LOCUS_GPU_STRESS": "1", "LOCUS_GPU_STRESS_TICKS": "7"}, **P)

# -- 15. ABSENT: cooperative launch unsupported -> skip that section only --
scenario("persist: absent cooperative launch",
         ["-DMOCK_COOP_UNSUPPORTED=1"], 0,
         must_contain=["RESULT: PASS",
                       "[SKIP] device reports cooperative launch unsupported"],
         env={"LOCUS_GPU_STRESS": "1", "LOCUS_GPU_STRESS_TICKS": "7"}, **P)

# -- 16. BROKEN: cooperative launch rejected at full grid -----------------
scenario("persist: cooperative rejected", ["-DMOCK_COOP_REJECT=1"], -1,
         must_contain=["rejected", "RESULT: FAIL",
                       "the resident grid must be sized down"],
         env={"LOCUS_GPU_STRESS": "1", "LOCUS_GPU_STRESS_TICKS": "7"}, **P)

# -- 17. BROKEN: occupancy query fails -> must stop, not guess a grid -----
scenario("persist: occupancy query fails", ["-DMOCK_OCC_FAIL=1"], -1,
         must_contain=["occupancy query returned a grid bound",
                       "RESULT: FAIL"],
         must_not_contain=["[2] LDS state"],
         env={"LOCUS_GPU_STRESS": "1", "LOCUS_GPU_STRESS_TICKS": "7"}, **P)

# -- 18. BROKEN: part of the grid never ran, so LDS state is missing ------
scenario("persist: partial launch loses LDS", ["-DMOCK_PARTIAL_LAUNCH=1"], -1,
         must_contain=["slots wrong", "RESULT: FAIL"],
         env={"LOCUS_GPU_STRESS": "1", "LOCUS_GPU_STRESS_TICKS": "7"}, **P)


failed = [r for r in results if not r[1]]
print("\n=== gpu probe mock verification (caps.cpp + persist.cpp) ===")
for name, ok, detail in results:
    print(f"  [{'PASS' if ok else 'FAIL'}] {name:38s} {detail}")
print(f"\n{len(results) - len(failed)}/{len(results)} scenarios passed")
print(f"RESULT: {'ALL PASS' if not failed else 'FAIL'}")
sys.exit(1 if failed else 0)
