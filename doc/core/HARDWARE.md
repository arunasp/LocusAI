# Hardware

Device facts this design depends on, and what follows from them. Everything
here was read from the machine rather than from documentation — the values
that matter are the ones the compiled code can see.

## The device

Read via `rocminfo` and confirmed against the HIP runtime from inside the
project's own container:

| Property | Value |
|---|---|
| Agent | `gfx1100` — AMD Radeon RX 7900 XT |
| Compute units | 84 (`rocminfo`) / 42 reported by HIP as `multiProcessorCount` |
| Wavefront | 32 (RDNA3) |
| Fast f16 | yes |
| Vector registers | 196,608 x 32-bit per WGP = 768 KiB (192 KiB per SIMD) |
| LDS per workgroup | 64 KB |
| L1 | 32 KB per CU (`rocminfo` only) |
| L2 | 6 MiB (`l2CacheSize` and `rocminfo` agree) |
| L3 / Infinity Cache | 80 MB (81920 KB, `rocminfo` only) |
| Cacheline | 64 bytes |
| VRAM pool | ~19.96 GiB, coarse-grained |
| Max waves per CU | 32 |
| XNACK | disabled |
| Coherent host access | false |
| Max clock | 2025 MHz |
| ISAs | `amdgcn-amd-amdhsa--gfx1100`, `gfx11-generic` |

Backend is **ROCm/HIP**, not Vulkan or DirectML.

## The compute-unit count depends on who is asked

`rocminfo` reports 84 compute units; HIP's `multiProcessorCount` reports 42
for the same device. Both are correct and neither is a rounding error.

On RDNA, two CUs are paired into a **work-group processor**, and it is the
WGP that HIP surfaces as a "multiprocessor". 42 WGPs x 2 = 84 CUs.

This matters for occupancy arithmetic rather than trivia: a kernel's
waves-per-multiprocessor budget is against the WGP, so using 84 there
overestimates available parallelism by exactly a factor of two. Take the
number from the HIP runtime when sizing a launch, and from `rocminfo` when
describing the hardware.

**There is no HIP route to the 84, measured 2026-09-10.**
`hipDeviceAttributePhysicalMultiProcessorCount` is documented as "all
available physical compute units" and looked like the obvious way to get it,
but on this platform it returns **42**, identical to
`hipDeviceAttributeMultiprocessorCount`. Both queries succeed; they simply
agree. So `rocminfo` remains the only source for the CU count, and a probe
that wants both numbers has to read two tools.

## Known runtime warning

A clean probe run still emits `Resource leak detected by SharedSignalPool,
2 Signals leaked` on exit. It comes from the ROCm runtime's own teardown,
not from the probe's allocations — every `hipMalloc` is freed and the
result verifies. Recorded rather than dismissed: it is unexplained, and a
second, different leak count later would otherwise look normal.

## Reaching the device

The host runs WSL2, so the GPU arrives through `/dev/dxg` rather than a
`/dev/dri` node.

**Two things below stopped being sufficient when the host moved to ROCm 7.14
on 2026-08-18, and both were silent until `make gpu` was next run on
2026-09-10.** Mounting `/opt/rocm` read-only is no longer enough on its own:

1. That tree's `bin`, `lib` and `include` are **update-alternatives
   symlinks into `/etc/alternatives`**, which the container does not mount,
   so every one of them dangles and `hipcc` is unreachable at the canonical
   path even though the tree is present. Docker resolves a symlink used *as*
   a mount source but does not rewrite symlinks *inside* a mounted
   directory, so no mount option fixes this. The resolved
   `/opt/rocm/core-<ver>/...` paths work, which is why `core/Makefile`
   discovers `ROCM_PATH` by looking for a real `bin/hipcc` rather than
   assuming `/opt/rocm`. Do not mount `/etc/alternatives` to "fix" it: that
   would overlay the container's own alternatives and break its `cc`/`g++`.
2. **The host's ROCm 7.14 ships no HIP headers at all.**
   `amdrocm7.14-gfx1100` was installed to repair Ollama's *runtime* path and
   carries `hipcc`, `amdclang++` and LLVM but not the SDK. Since this
   container mounts ROCm from the host by design, it inherits that gap
   exactly. The headers come from `amdrocm-runtime-dev7.14` (494 KB,
   installs to the real `/opt/rocm/core-7.14/include/hip`), extracted into
   `core/gpu/.rocm-include` and passed with `-isystem`. That tree is
   gitignored and version-pinned: regenerate it after any ROCm move rather
   than trusting it.

**Why not source ROCm from a container image instead**, checked against
Docker Hub's `rocm` org on 2026-09-10 so it is not re-surveyed: AMD stopped
publishing a thin dev image at 7.x. The 6.4 line has a plain tag at 0.96 GiB
and a `-complete` at ~4 GiB, but 7.14 and 10.0 ship **`-full` only**, 7.40
to 7.70 GiB compressed — roughly 18-20 GiB extracted, against 24.7 GiB free
on `/data/disk`. The thin images that do exist (`rocm-terminal:latest` at
1.02 GiB, `dev-ubuntu-24.04:6.4` at 0.96 GiB) are ROCm **6.4**, and 6.4
headers against a 7.14 runtime is precisely the skew `caps.cpp`'s field
detection exists to *survive* rather than detect. So the 494 KB package
extraction is not a stopgap for a better source — it is the only option that
is both thin and version-exact, because it comes from the same package set
the host actually runs.

**The staleness is therefore handled by assertion, not by choice of source.**
Both sides publish the same triple — the host at
`$ROCM_PATH/share/hip/version` and the headers at `hip/hip_version.h` — so
`make hip-header-check` compares them and fails loudly on a mismatch. It is
a prerequisite of `gpu`, `caps` and `persist`, so skewed headers cannot
reach a compile; it skips when the extracted tree is not in use, since a
host with a complete SDK has nothing to skew. Mutation-tested rather than
assumed: perturbing the header's patch level turns it red and blocks the
build.

The rest of the working configuration, established by adding exactly
what each failure reported missing:

- device `/dev/dxg` — mode `crw-rw-rw-`, so no group membership and no
  `--privileged`
- `/usr/lib/wsl` mounted read-only — holds `libdxcore.so` and the D3D12
  libraries ROCm links against on WSL; the whole tree rather than `lib/`
  alone, because driver blobs under `drivers/` resolve through it
- `/opt/rocm` mounted read-only rather than installed — no multi-gigabyte
  image layer, and the container's ROCm matches the host's by construction
- `LD_LIBRARY_PATH=/opt/rocm/lib:/usr/lib/wsl/lib`
- `/opt/rocm/bin` prepended to `PATH` — an allowlist that resolves binaries
  through `PATH` will otherwise report a mounted tool as not installed
- image packages: `libelf1`, `libnuma1`, `libdrm2` for ROCm's runtime, and
  `g++` because HIP is C++ and `hipcc` is a clang++ driver

`core/gpu/probe.cpp` and `make -C core gpu` exist to verify this path before
any real kernel does, so a later failure is unambiguously the kernel's.

## Design consequences

**Wavefront 32 sets the natural competition granularity.** An active set
bounded near four sits far below one wavefront, so competition over the
active set is essentially free and needs no special handling. The stage
where wavefront-aligned work actually pays is **separation**, which competes
over a much larger population — that is where kernel effort belongs, not in
kWTA over the active set.

**XNACK disabled means no page-fault-driven unified memory.** Host and
device transfers are explicit. The tier model must therefore move data
deliberately rather than relying on demand paging — which suits this design:
residency becomes a real decision the substrate makes, rather than something
the driver hides.

**80 MB of Infinity Cache is the locality budget.** A trace store whose
active working set fits inside it is in a different performance regime from
one that does not, which makes working-set size a design target rather than
an outcome.

**The register file is larger than L2, which inverts the usual assumption.**
42 WGPs x 768 KiB is 32.25 MiB of vector registers against 6 MiB of L2. The
innermost tier has over five times the capacity of the one outside it, so
"spill to the next level down" is not a size ladder here -- a working set
that fits in registers has more room than one that merely fits in L2.

**Cooperative launch is unsupported on this device, measured 2026-09-10:
`cooperativeLaunch` reads 0.** This closes a route the tier model might
otherwise have taken. LDS does not survive a kernel launch, so an
activity-maintained ACTIVE tier that literally lives where the compute is
would need a resident kernel -- and without cooperative launch a grid-wide
barrier is illegal, so the whole-grid form of that is unavailable. Per-block
persistence remains possible. `gpu/persist.cpp` probes this and skips its
cooperative section rather than failing.

**The memory hierarchy is not runtime-configurable, and one query lies.**
`hipDeviceGetLimit` accepts only `hipLimitStackSize` and
`hipLimitMallocHeapSize` (`hipLimitPrintfFifoSize` returns
`hipErrorUnsupportedLimit`), and `hipLimit_t` has no persisting-cache member
at all -- so `hipAccessPolicyWindow` exists with nothing to arm it, and
`accessPolicyMaxWindowSize` reads 0 even though `persistingL2CacheMaxSize`
advertises the whole 6 MiB. `hipDeviceSetCacheConfig` returns `hipSuccess`
here despite AMD's docs saying it is unimplemented, i.e. it is a silently
accepted no-op. Placement and eviction are hardware decisions: the only
software lever is *negative* (non-temporal / no-allocate hints), so a
hot/cold split has to work by SIZING a structure to stay resident, never by
placing it. And **DRAM bandwidth cannot be derived from HIP on this
platform**: `memoryBusWidth` 320 with `memoryClockRate` 1.25 GHz yields
50 or 100 GB/s depending on how the field is read, against a real ~800 GB/s.
The field is unreliable under GPU-PV; use the card's published figure.

## Fixed-function reuse

Assessed for whether hardware built for something else can serve a substrate
mechanism.

**Confirmed usable.** The VCN media engine for asynchronous event-stream
ingestion, off the main compute path. Infinity Cache and LDS for locality.
RDNA3 AI accelerators for the dense tensor operations in three-factor
learning, kWTA and completion.

**Unconfirmed — needs direct verification, not assumption.** Ray
accelerators for nearest-attractor search: on RDNA3 these are embedded in
the CU rather than being a separate co-processor as on some other vendors'
parts, so the benefit is smaller even if the capability is exposed at all.
No confirmed equivalent of a dedicated optical-flow engine.

**Negative.** Codec transform hardware is codec-locked and not reusable for
general work. The DCT remains a useful dense-tensor *algorithm* for the
separation stage — GPU-native decorrelation, and a reasonable analogue of
retinal efficient coding — but it runs on the normal compute path, not on
fixed-function silicon.
