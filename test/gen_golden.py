#!/usr/bin/env python3
"""gen_golden.py — regenerate per-op golden references with numpy (ground truth).

Reads each op's existing ``<kernel>_input.txt`` and computes the numpy reference
via the vendor scalar math (``vendor/gtx_cpp_reference/test/compare_all_ops.py``:
compute_scalar + OP_CONFIG), then writes ``<kernel>_numpy_golden.txt`` next to it.

Non-destructive: never overwrites ``_ref.txt``. Reports, per op, whether the new
numpy golden matches the existing ref (so wrong refs are surfaced).

Usage:
  python3 test/gen_golden.py                 # all ops in OP_CONFIG
  python3 test/gen_golden.py abs neg cos     # selected ops (compare_all_ops names)
"""
import os
import sys

import numpy as np

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
TEST_DIR = os.path.join(REPO, "test")
VENDOR_TEST = os.path.join(REPO, "vendor", "gtx_cpp_reference", "test")
sys.path.insert(0, VENDOR_TEST)

from compare_all_ops import (  # noqa: E402  (vendor numpy reference math)
    OP_CONFIG, compute_scalar, load_data, hex_to_fp16_array,
)

BYTES_PER_LINE = 32  # 256-bit bus word → 16 fp16 → 64 hex chars/line


def fp16_be_hex_lines(values):
    """float array → big-endian FP16 hex, 32 bytes (16 values) per line."""
    raw = np.asarray(values, dtype="<f2").astype(">f2").tobytes()
    hexstr = raw.hex()
    return [hexstr[i:i + BYTES_PER_LINE * 2]
            for i in range(0, len(hexstr), BYTES_PER_LINE * 2)]


def main():
    sel = [a for a in sys.argv[1:] if not a.startswith("-")]
    ops = sel if sel else list(OP_CONFIG.keys())

    gen = skip = match = differ = 0
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
            golden = compute_scalar(op_name, cfg["type"], load_data(input_file))
        except Exception as e:  # noqa: BLE001
            print(f"  {op_name:<16} MATH-ERR {type(e).__name__}: {e}"); skip += 1; continue
        if golden is None or len(golden) == 0:
            print(f"  {op_name:<16} NO-NUMPY-MATH ({cfg['type']})"); skip += 1; continue

        lines = fp16_be_hex_lines(golden)
        with open(out_file, "w") as f:
            f.write("\n".join(lines) + "\n")
        gen += 1

        # surface refs that disagree with the numpy ground truth
        status = "no-ref"
        if os.path.exists(ref_file):
            rd = load_data(ref_file); rl = []
            for a in rd:
                rl.extend(rd[a])
            ref = np.asarray(hex_to_fp16_array(rl), dtype=np.float16)  # hex_to_* returns f32
            g = np.asarray(golden, dtype=np.float16)
            n = min(len(ref), len(g))
            mm = int(np.sum(ref[:n].view(np.uint16) != g[:n].view(np.uint16)))
            if mm == 0 and len(ref) == len(g):
                status = "ref MATCHES"; match += 1
            else:
                status = f"ref DIFFERS mm={mm}/{n} (len ref={len(ref)} numpy={len(g)})"; differ += 1
        print(f"  {op_name:<16} {len(golden):>9} elems  {status}")

    print(f"\n[gen_golden] generated={gen} skipped={skip}  "
          f"vs existing ref: match={match} differ={differ}")


if __name__ == "__main__":
    main()
