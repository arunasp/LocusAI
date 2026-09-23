"""Babble S streams on the device, and prove it matches the CPU exactly.

  babble_gpu.py STORE BINARY [--streams S] [--bytes N] [--seed TEXT]
                [--verify K] [--out FILE]

Packs a written store into the flat form core/gpu/babble_device.cpp
reads, runs it, and writes the generated streams.

WHY A PACKER AND NOT A LOADER IN C++: the store's sparse rows already
have a reader in py/locus/knowledge.py, and that reader is the oracle
for every figure in this project. A second parser in C++ would be a
second definition of what a store IS, free to drift from the first
without any test noticing.

VERIFICATION IS BYTE-FOR-BYTE, not statistical. `--verify K` regenerates
the first K streams in Python -- through Knowledge.drive(), the same
call exp_babble.py uses -- with splitmix64 reimplemented here so the two
draw the same numbers, and compares the bytes. A distributional check
("both look like babble") would pass while the kernel computed something
subtly different, which is the whole failure mode worth catching.

ALIGNMENT IS THE PACKER'S JOB: the C++ side casts straight into the
buffer, so every double array has to start on an 8-byte boundary. The
byte arrays are padded here rather than copied there.
"""

import os
import struct
import subprocess
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "py"))

from locus.knowledge import Knowledge, TOP_ONLY, BYTE_UNITS  # noqa: E402

MASK = (1 << 64) - 1


def mix64(state):
    """splitmix64, the kernel's generator, in Python."""
    state = (state + 0x9E3779B97F4A7C15) & MASK
    z = state
    z = ((z ^ (z >> 30)) * 0xBF58476D1CE4E5B9) & MASK
    z = ((z ^ (z >> 27)) * 0x94D049BB133111EB) & MASK
    return state, (z ^ (z >> 31)) & MASK


def next_unit(state):
    state, z = mix64(state)
    return state, (z >> 11) * (1.0 / 9007199254740992.0)


def pad8(blob):
    return blob + b"\0" * ((8 - len(blob) % 8) % 8)


def csr(store, units):
    """rowptr/byte/weight for `store`, as the C++ side expects them."""
    rowptr = struct.pack("<%dq" % (units + 1), *store.rowptr[:units + 1])
    nnz = store.rowptr[units]
    byte = bytes(store.byte[:nnz])
    weight = struct.pack("<%dd" % nnz, *store.weight[:nnz])
    return nnz, rowptr, byte, weight


def pack(know, streams, seed_text, steps, seeds, dump=0):
    """`dump` 0 is the NEUTRAL default: the device emits its working
    quantities only when the host is going to check them, so an
    ordinary run carries no diagnostic cost.
    """
    enc = know.enc
    out = [struct.pack("<iii", enc.n, len(enc.tables), max(enc.orders))]
    for k, _h, seed, off, size in enc.tables:
        out.append(struct.pack("<iIii", k, seed, off, size))
    cortex = know.stores[0]
    hippo = None
    for s in know.stores[1:]:
        if s.flags & TOP_ONLY:
            hippo = s
    out.append(struct.pack("<iii", 1 if know.gated else 0,
                           1 if hippo else 0, streams))
    # dump + one pad int: the double that follows must stay 8-aligned,
    # because the C++ side casts straight into this buffer.
    out.append(struct.pack("<ii", dump, 0))
    out.append(struct.pack("<d", know.lam))
    out.append(struct.pack("<qq", len(seed_text), steps))
    for store in [cortex] + ([hippo] if hippo else []):
        nnz, rowptr, byte, weight = csr(store, enc.n)
        out.append(struct.pack("<q", nnz))
        out.append(rowptr)
        out.append(pad8(byte))
        out.append(weight)
    out.append(pad8(seed_text))
    out.append(struct.pack("<%dQ" % streams, *seeds))
    return b"".join(out)


def cpu_stream(know, seed_text, steps, rng_seed):
    """The same stream in Python, through the reader that defines it."""
    buf = bytearray(seed_text)
    state = rng_seed
    for _ in range(steps):
        drive = know.drive(bytes(buf), len(buf) - 1)
        total = sum(v for v in drive.values() if v > 0.0) + know.lam
        state, u = next_unit(state)
        r = u * total
        acc = 0.0
        pick = BYTE_UNITS - 1
        for b in range(BYTE_UNITS):
            v = drive.get(b, 0.0)
            acc += (v if v > 0.0 else 0.0) + know.lam / BYTE_UNITS
            if acc >= r:
                pick = b
                break
        buf.append(pick)
    return bytes(buf[len(seed_text):])


def generate(store_path, binary, streams, steps, seed_text=b"Once upon",
             know=None, workdir="/tmp"):
    """`streams` babble streams from `store_path`, generated on device.

    The importable form of what main() does, so an experiment does not
    have to shell out and re-parse its own output. Seeds are 1..streams,
    matching the CPU babbler's convention, so a stream generated here
    and one generated there are the same stream.
    """
    know = know or Knowledge(store_path)
    seeds = [1 + i for i in range(streams)]
    blob = pack(know, streams, seed_text, steps, seeds)
    inp = os.path.join(workdir, "babgen.in")
    res = os.path.join(workdir, "babgen.bin")
    with open(inp, "wb") as fh:
        fh.write(blob)
    try:
        r = subprocess.run([binary, inp, res], capture_output=True,
                           text=True)
        if r.returncode != 0:
            raise SystemExit("FAIL: %s exited %d: %s"
                             % (binary, r.returncode, r.stderr))
        with open(res, "rb") as fh:
            data = fh.read()
    finally:
        for path in (inp, res):
            if os.path.exists(path):
                os.unlink(path)
    return [data[i * steps:(i + 1) * steps] for i in range(streams)]


def main(argv):
    if len(argv) < 2:
        print(__doc__)
        return 2
    store, binary, rest = argv[0], argv[1], argv[2:]
    streams, steps, verify, out, dump = 8, 4096, 0, None, 0
    seed_text = b"Once upon a time"
    while rest:
        if rest[0] == "--streams" and len(rest) > 1:
            streams, rest = int(rest[1]), rest[2:]
        elif rest[0] == "--bytes" and len(rest) > 1:
            steps, rest = int(rest[1]), rest[2:]
        elif rest[0] == "--seed" and len(rest) > 1:
            seed_text, rest = rest[1].encode(), rest[2:]
        elif rest[0] == "--dump" and len(rest) > 1:
            dump, rest = int(rest[1]), rest[2:]
        elif rest[0] == "--verify" and len(rest) > 1:
            verify, rest = int(rest[1]), rest[2:]
        elif rest[0] == "--out" and len(rest) > 1:
            out, rest = rest[1], rest[2:]
        else:
            print("unknown argument: %s" % rest[0])
            return 2

    know = Knowledge(store)
    seeds = [1 + i for i in range(streams)]
    blob = pack(know, streams, seed_text, steps, seeds, dump)
    inp = (out or "babble") + ".in"
    res = (out or "babble") + ".bin"
    with open(inp, "wb") as fh:
        fh.write(blob)
    r = subprocess.run([binary, inp, res], capture_output=True, text=True)
    os.unlink(inp)
    if r.returncode != 0:
        print("FAIL: %s exited %d: %s" % (binary, r.returncode, r.stderr))
        return 3
    with open(res, "rb") as fh:
        data = fh.read()
    print("device    %d streams x %d bytes -> %s" % (streams, steps, res))
    print(r.stderr.strip())

    # PRECISION: the device decided, the host rules. Compare the
    # quantities rather than the byte -- two babbles can differ at byte 0
    # for a reason as small as one row being gated differently, and the
    # byte alone cannot say which.
    if dump:
        import array
        with open(res + ".dbg", "rb") as fh:
            raw = array.array("d")
            raw.frombytes(fh.read())
        buf = bytearray(seed_text)
        state = seeds[0]
        for step in range(dump):
            base = step * (BYTE_UNITS + 2)
            dev = list(raw[base:base + BYTE_UNITS])
            dev_total = raw[base + BYTE_UNITS]
            dev_u = raw[base + BYTE_UNITS + 1]
            drive = know.drive(bytes(buf), len(buf) - 1)
            cpu = [max(drive.get(b, 0.0), 0.0) for b in range(BYTE_UNITS)]
            cpu_total = sum(cpu) + know.lam
            worst = max(abs(a - b) for a, b in zip(dev, cpu))
            nz_dev = sum(1 for v in dev if v != 0.0)
            nz_cpu = sum(1 for v in cpu if v != 0.0)
            print("step %d  worst |dev-cpu| %.3e   non-zero dev %d cpu %d"
                  % (step, worst, nz_dev, nz_cpu))
            print("        total dev %.9f cpu %.9f   uniform %.9f"
                  % (dev_total, cpu_total, dev_u))
            state, u = next_unit(state)
            print("        host draws %.9f  %s"
                  % (u, "same" if abs(u - dev_u) < 1e-15 else "DIFFERENT"))
            r = u * cpu_total
            acc, pick = 0.0, BYTE_UNITS - 1
            for b in range(BYTE_UNITS):
                acc += cpu[b] + know.lam / BYTE_UNITS
                if acc >= r:
                    pick = b
                    break
            print("        host picks %r, device picked %r"
                  % (bytes([pick]), bytes([data[step]])))
            buf.append(data[step])

    ok = True
    for i in range(min(verify, streams)):
        got = data[i * steps:(i + 1) * steps]
        want = cpu_stream(know, seed_text, steps, seeds[i])
        same = got == want
        ok = ok and same
        if same:
            print("stream %d  IDENTICAL to the CPU babbler (%d bytes)"
                  % (i, steps))
        else:
            n = next((j for j in range(min(len(got), len(want)))
                      if got[j] != want[j]), None)
            print("stream %d  DIFFERS at byte %s" % (i, n))
            print("  device %r" % got[max(0, (n or 0) - 20):(n or 0) + 20])
            print("  cpu    %r" % want[max(0, (n or 0) - 20):(n or 0) + 20])
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
