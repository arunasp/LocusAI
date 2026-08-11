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
| Compute units | 84, 2 SIMDs each, 6 shader engines |
| Wavefront | 32 (RDNA3) |
| Fast f16 | yes |
| VRAM pool | ~19.96 GiB, coarse-grained |
| L3 / Infinity Cache | 80 MB |
| LDS per workgroup | 64 KB |
| Max waves per CU | 32 |
| XNACK | disabled |
| Coherent host access | false |
| Max clock | 2025 MHz |
| ISAs | `amdgcn-amd-amdhsa--gfx1100`, `gfx11-generic` |

Backend is **ROCm/HIP**, not Vulkan or DirectML.

## Reaching the device

The host runs WSL2, so the GPU arrives through `/dev/dxg` rather than a
`/dev/dri` node. The working configuration, established by adding exactly
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
