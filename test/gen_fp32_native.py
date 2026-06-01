#!/usr/bin/env python3
"""gen_fp32_data_golden.py — regenerate fp32 input + fp32 golden for n1s16 ops.

SMM_ISA v2.0.0d makes ALL I/O fp32. This regenerates, per op:
  * <kernel>_input.txt        fresh fp32 random (op-appropriate range), 8-lane
                              little-endian elem-reversed (GTX_DDR_REVERSE_MODE=elem)
  * <kernel>_fp32_golden.txt  fp32 reference via the vendor numpy math
                              (compare_all_ops.compute_scalar — already computes in
                              fp32; we only swap the fp16 input decode / output for fp32)

The reference op formulas are reused verbatim from compare_all_ops; we monkeypatch
its hex→array decoder to read fp32 so the same code path yields an fp32 golden.

Run a kernel afterwards with GTX_DDR_REVERSE_MODE=elem (DDR/L2/L1/L0 all fp32) and
byte-compare the dump against <kernel>_fp32_golden.txt.

Usage: python3 test/gen_fp32_data_golden.py [op ...]   (default: all in OP_CONFIG)
"""
import os
import sys

import numpy as np

REPO = "/mnt/e/14_NIGHTLY/pyspike"
VENDOR_TEST = os.path.join(REPO, "vendor", "gtx_cpp_reference", "test")
sys.path.insert(0, VENDOR_TEST)
import compare_all_ops as cao  # noqa: E402

WORD = 8  # fp32 lanes per 32-byte bus word

# --- op input ranges (mirror generate_n1s16_tests positivity/nonzero needs) ---
_POSITIVE = {"sqrt", "rsqrt", "log"}
_NONZERO_B = {"div"}


def _enc_fp32_elem(vals):
    """logical fp32 -> 8-lane elem-reversed little-endian bytes."""
    v = np.asarray(vals, dtype="<f4").astype(np.float32)
    pad = (-len(v)) % WORD
    if pad:
        v = np.concatenate([v, np.zeros(pad, np.float32)])
    return v.reshape(-1, WORD)[:, ::-1].reshape(-1).astype("<f4").tobytes()


def _dec_fp32_elem(hex_lines):
    """hex lines -> logical fp32 (undo 8-lane elem reversal)."""
    raw = b"".join(bytes.fromhex(l.strip()) for l in hex_lines
                   if l.strip() and not l.strip().startswith("@"))
    v = np.frombuffer(raw, dtype="<f4").astype(np.float32).copy()
    m = (len(v) // WORD) * WORD
    if m:
        v[:m] = v[:m].reshape(-1, WORD)[:, ::-1].reshape(-1)
    return v


def _hex_lines(byts, addr):
    h = byts.hex()
    out = [f"@{addr:x}"]
    out += [h[i:i + 64] for i in range(0, len(h), 64)]
    return out


def _section_counts(input_file):
    """(addr, n_elems) per @-section of an existing fp16 input file."""
    secs = []
    addr = None
    nbytes = 0
    for line in open(input_file):
        s = line.strip()
        if not s:
            continue
        if s.startswith("@"):
            if addr is not None:
                secs.append((addr, nbytes // 2))   # existing files are fp16 (2 B)
            addr = int(s[1:], 16)
            nbytes = 0
        else:
            nbytes += len(s) // 2
    if addr is not None:
        secs.append((addr, nbytes // 2))
    return secs


def _gen_values(op_name, n, seed):
    rng = np.random.default_rng(seed)
    if op_name in _POSITIVE:
        return rng.uniform(0.01, 4.0, n).astype(np.float32)
    return rng.uniform(-3.0, 3.0, n).astype(np.float32)


def main():
    sel = [a for a in sys.argv[1:] if not a.startswith("-")]
    ops = sel if sel else list(cao.OP_CONFIG.keys())
    print(f"[gen_fp32] {len(ops)} ops, base={os.path.join(REPO, 'test')}")
    ok = skip = 0
    for op in ops:
        cfg = cao.OP_CONFIG.get(op)
        if not cfg:
            print(f"  {op:<16} UNKNOWN"); continue
        ddir = os.path.join(REPO, "test", cfg["dir"], "data")
        kern = cfg["kernel"]
        in_f = os.path.join(ddir, f"{kern}_input.txt")
        if not os.path.exists(in_f):
            print(f"  {op:<16} NO-INPUT"); skip += 1; continue
        secs = _section_counts(in_f)

        # fresh fp32 input per section (range by op)
        dd = {}
        out_lines = []
        for si, (addr, n) in enumerate(secs):
            vals = _gen_values(op if si == 0 else "", n, seed=42 + si + hash(op) % 1000)
            if op in _NONZERO_B and si == 1:
                vals = np.where(vals == 0, np.float32(1.0), vals)
            b = _enc_fp32_elem(vals)
            out_lines += _hex_lines(b, addr)
            dd[f"@{addr:x}"] = _hex_lines(b, addr)[1:]   # hexlines (no @)
        with open(in_f, "w") as f:
            f.write("\n".join(out_lines) + "\n")

        # fp32 golden via vendor math (fp32 decode monkeypatched)
        _orig = cao.hex_to_fp16_array
        cao.hex_to_fp16_array = lambda hl: _dec_fp32_elem(hl)
        try:
            golden = cao.compute_scalar(op, cfg["type"], dd)
        except Exception as e:  # noqa: BLE001
            cao.hex_to_fp16_array = _orig
            print(f"  {op:<16} MATH-ERR {type(e).__name__}: {e}"); skip += 1; continue
        cao.hex_to_fp16_array = _orig
        if golden is None or len(golden) == 0:
            print(f"  {op:<16} NO-MATH"); skip += 1; continue

        gb = _enc_fp32_elem(np.asarray(golden, dtype=np.float32))
        with open(os.path.join(ddir, f"{kern}_fp32_golden.txt"), "w") as f:
            f.write("\n".join(_hex_lines(gb, 0)[1:]) + "\n")  # raw hex, no @
        ok += 1
        print(f"  {op:<16} {len(golden):>8} elems  OK")
    print(f"\n[gen_fp32] generated={ok} skipped={skip}")


if __name__ == "__main__":
    main()
