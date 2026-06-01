#!/usr/bin/env python3
"""fp16_input_to_fp32.py — upcast a test/ fp16 @-addressed hex input to fp32.

SMM_ISA v2.0.0d unifies DDR/L2/L1/L0 to fp32. The committed test inputs are
fp16 (2 B/elem, GTX_DDR_REVERSE_MODE=elem → 16 lanes per 32-byte bus word). This
re-encodes them to fp32 (4 B/elem, 8 lanes per bus word), preserving logical
element order, so the unified-fp32 simulator reads the same values.

Usage: fp16_input_to_fp32.py <in_fp16.txt> <out_fp32.txt>
"""
import sys
import numpy as np

WORD16, WORD32 = 16, 8  # fp16 / fp32 elements per 32-byte bus word


def _decode_elem_fp16(hexstr: str) -> np.ndarray:
    raw = bytes.fromhex(hexstr)
    v = np.frombuffer(raw, dtype="<f2").astype(np.float16).copy()
    m = (len(v) // WORD16) * WORD16
    if m:                                  # undo the 16-lane bus-word reversal
        v[:m] = v[:m].reshape(-1, WORD16)[:, ::-1].reshape(-1)
    return v


def _encode_elem_fp32(values: np.ndarray) -> str:
    v = values.astype("<f4").astype(np.float32)
    pad = (-len(v)) % WORD32
    if pad:
        v = np.concatenate([v, np.zeros(pad, dtype=np.float32)])
    v = v.reshape(-1, WORD32)[:, ::-1].reshape(-1)   # 8-lane bus-word reversal
    return v.astype("<f4").tobytes().hex()


def main() -> None:
    src, dst = sys.argv[1], sys.argv[2]
    out_lines = []
    cur_hex = []

    def flush():
        if not cur_hex:
            return
        logical = _decode_elem_fp16("".join(cur_hex))
        h = _encode_elem_fp32(logical)
        for i in range(0, len(h), 64):       # 32 B (8 fp32) per line
            out_lines.append(h[i:i + 64])
        cur_hex.clear()

    for line in open(src):
        s = line.strip()
        if not s:
            continue
        if s.startswith("@"):
            flush()
            out_lines.append(s)
        else:
            cur_hex.append(s)
    flush()
    with open(dst, "w") as f:
        f.write("\n".join(out_lines) + "\n")
    print(f"[fp16->fp32] {src} -> {dst}  ({len(out_lines)} lines)")


if __name__ == "__main__":
    main()
