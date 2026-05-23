#!/usr/bin/env python3
"""gen_golden.py — regenerate per-op golden references with numpy (ground truth).

Reads each op's existing ``<kernel>_input.txt`` and computes the numpy reference
via the vendor scalar math (``vendor/gtx_cpp_reference/test/compare_all_ops.py``:
compute_scalar + OP_CONFIG), then writes ``<kernel>_numpy_golden.txt`` next to it.

Non-destructive: never overwrites ``_ref.txt``. Reports, per op, the ULP delta
between the numpy golden and the existing ref so wrong refs are surfaced.

Corpus / endianness (env):
  GTX_GOLDEN_BASE    base dir holding <OP>/n1s16/data trees (default: repo test/).
                     Set to the ggml_ops_c corpus for ISS-accurate, w=8/h=64 data.
  GTX_GOLDEN_ENDIAN  fp16 hex layout in the data files: 'be' (test/, default) or
                     'le' (ggml_ops_c — GTX_DDR_REVERSED: per-element little-endian
                     fp16, with the 16 elements of each 256-bit bus word in reversed
                     order, matching the SystemC ISS memory image — proven by the
                     ARANGE ref [15,14,..,0, 31,..,16]). compare_all_ops is BE-internal,
                     so non-BE corpora are translated to logical row-major at the edges.

Usage:
  python3 test/gen_golden.py                 # all ops in OP_CONFIG
  python3 test/gen_golden.py abs neg cos     # selected ops (compare_all_ops names)
"""
import os
import sys
from collections import OrderedDict

import numpy as np

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
# Corpus base holding the per-op <OP>/n1s16/data trees. Default = repo test/,
# but GTX_GOLDEN_BASE lets us target the ggml_ops_c corpus (ISS-accurate, w=8/h=64)
# non-destructively. cfg["dir"] (e.g. "ABS/n1s16") is identical across corpora.
TEST_DIR = os.environ.get("GTX_GOLDEN_BASE", os.path.join(REPO, "test"))
VENDOR_TEST = os.path.join(REPO, "vendor", "gtx_cpp_reference", "test")
ENDIAN = os.environ.get("GTX_GOLDEN_ENDIAN", "be").lower()
sys.path.insert(0, VENDOR_TEST)

from compare_all_ops import (  # noqa: E402  (vendor numpy reference math)
    OP_CONFIG, compute_scalar, load_data,
)

BYTES_PER_LINE = 32  # 256-bit bus word → 16 fp16 → 64 hex chars/line


_WORD = BYTES_PER_LINE // 2  # 16 fp16 elements per 256-bit bus word


def _bus_decode(raw):
    """Raw file bytes → logical row-major fp16 array, per corpus byte order.

    le: per-element little-endian fp16, with each 16-element bus word reversed back to
    logical order (GTX_DDR_REVERSED). be: big-endian in place (test/).
    """
    if ENDIAN != "le":
        return np.frombuffer(raw, dtype=">f2").astype(np.float16)
    v = np.frombuffer(raw, dtype="<f2").astype(np.float16)
    m = (len(v) // _WORD) * _WORD
    if m:
        v[:m] = v[:m].reshape(-1, _WORD)[:, ::-1].reshape(-1)
    return v


def _bus_encode(values):
    """Logical row-major fp16 → file bytes, per corpus byte order (inverse of _bus_decode)."""
    v = np.asarray(values, dtype="<f2").astype(np.float16)
    if ENDIAN != "le":
        return v.astype(">f2").tobytes()
    pad = (-len(v)) % _WORD
    if pad:
        v = np.concatenate([v, np.zeros(pad, dtype=np.float16)])
    return v.reshape(-1, _WORD)[:, ::-1].reshape(-1).astype("<f2").tobytes()


def _hexlines_to_fp16(hexlines):
    """Concatenate @-stripped hex lines and decode to logical fp16 (corpus byte order)."""
    raw = b"".join(bytes.fromhex(l.strip())
                   for l in hexlines
                   if l.strip() and not l.strip().startswith("@"))
    return _bus_decode(raw)


def load_input_be(input_file):
    """Load input sections, translating to BE hex so compute_scalar (BE-internal)
    sees the right values regardless of the corpus byte order."""
    secs = load_data(input_file)
    if ENDIAN == "be":
        return secs
    out = OrderedDict()
    for addr, hexlines in secs.items():
        be = _hexlines_to_fp16(hexlines).astype(">f2").tobytes().hex()
        out[addr] = [be[i:i + BYTES_PER_LINE * 2]
                     for i in range(0, len(be), BYTES_PER_LINE * 2)]
    return out


def fp16_hex_lines(values):
    """logical float array → corpus-layout FP16 hex, 32 bytes (16 values) per line."""
    h = _bus_encode(values).hex()
    return [h[i:i + BYTES_PER_LINE * 2]
            for i in range(0, len(h), BYTES_PER_LINE * 2)]


def decode_ref(ref_file):
    """Decode an existing ref/result file to logical fp16 in the corpus byte order."""
    secs = load_data(ref_file)
    flat = []
    for addr in secs:
        flat.extend(secs[addr])
    return _hexlines_to_fp16(flat)


def ulp_stats(golden, ref):
    """verify.py-style tolerance check of golden vs ref (both fp16): pass per element
    when sign-magnitude ULP<=1 or |abs diff|<=0.001; NaN/Inf on either side is a miss.
    Returns (verdict, one-line detail). verdict: PASS / NAN-EDGE / FAIL / LEN."""
    g = np.asarray(golden, dtype=np.float16)
    r = np.asarray(ref, dtype=np.float16)
    n = min(len(g), len(r))
    if n == 0:
        return "LEN", f"EMPTY (ref={len(r)} numpy={len(g)})"
    gu = g[:n].view(np.uint16).astype(np.int32)
    ru = r[:n].view(np.uint16).astype(np.int32)
    sm = lambda u: np.where(u & 0x8000, -(u & 0x7FFF), u & 0x7FFF)
    special = (lambda u: (u & 0x7C00) == 0x7C00)  # Inf or NaN
    bad = special(gu) | special(ru)
    ulp = np.where(bad, 0xFFFF, np.abs(sm(gu) - sm(ru)))
    absd = np.where(bad, np.inf,
                    np.abs(g[:n].astype(np.float32) - r[:n].astype(np.float32)))
    within = (ulp <= 1) | (absd <= 0.001)
    npass = int(within.sum())
    # failures restricted to finite positions (real math errors vs NaN/Inf edge)
    finite_fail = int(np.sum(~within & ~bad))
    same_len = len(g) == len(r)
    detail = (f"pass={npass}/{n} finite_fail={finite_fail} bad={int(bad.sum())}"
              + ("" if same_len else f" len(ref={len(r)} numpy={len(g)})"))
    if not same_len:
        return "LEN", detail
    if npass == n:
        return "PASS", detail
    if finite_fail == 0:
        return "NAN-EDGE", detail
    return "FAIL", detail


def main():
    sel = [a for a in sys.argv[1:] if not a.startswith("-")]
    ops = sel if sel else list(OP_CONFIG.keys())

    print(f"[gen_golden] base={TEST_DIR} endian={ENDIAN}")
    gen = skip = 0
    tally = {"PASS": 0, "NAN-EDGE": 0, "FAIL": 0, "LEN": 0}
    for op_name in ops:
        cfg = OP_CONFIG.get(op_name)
        if not cfg:
            print(f"  {op_name:<16} UNKNOWN-OP"); continue
        data_dir = os.path.join(TEST_DIR, cfg["dir"], "data")
        kernel = cfg["kernel"]
        input_file = os.path.join(data_dir, f"{kernel}_input.txt")
        ref_file = os.path.join(data_dir, f"{kernel}_ref.txt")
        out_file = os.path.join(data_dir, f"{kernel}_numpy_golden.txt")

        if not os.path.exists(input_file):
            print(f"  {op_name:<16} NO-INPUT"); skip += 1; continue
        try:
            golden = compute_scalar(op_name, cfg["type"], load_input_be(input_file))
        except Exception as e:  # noqa: BLE001
            print(f"  {op_name:<16} MATH-ERR {type(e).__name__}: {e}"); skip += 1; continue
        if golden is None or len(golden) == 0:
            print(f"  {op_name:<16} NO-NUMPY-MATH ({cfg['type']})"); skip += 1; continue

        with open(out_file, "w") as f:
            f.write("\n".join(fp16_hex_lines(golden)) + "\n")
        gen += 1

        verdict, detail = "NOREF", ""
        if os.path.exists(ref_file):
            verdict, detail = ulp_stats(golden, decode_ref(ref_file))
            tally[verdict] += 1
        print(f"  {op_name:<16} {len(golden):>7} elems  {verdict:<8} {detail}")

    print(f"\n[gen_golden] generated={gen} skipped={skip}  vs ISS ref: "
          f"PASS={tally['PASS']} NAN-EDGE={tally['NAN-EDGE']} "
          f"FAIL={tally['FAIL']} LEN={tally['LEN']}")


if __name__ == "__main__":
    main()
