"""Minimal verify.py port for P4 strict-mode .elf regression (D-13 / D-14).

BE FP16 bit-pair compare per verify.py:235 (Pitfall 1).
P6 promotes this to riscv.gtx._verify with CLI; P4 keeps it test-only.
"""
import numpy as np
from typing import Tuple


def _parse_hex(path: str) -> bytes:
    out = bytearray()
    with open(path, 'r') as f:
        for raw in f:
            line = raw.strip()
            if not line or line.startswith('#') or line.startswith('@'):
                continue
            clean = ''.join(line.split())
            out.extend(bytes.fromhex(clean))
    return bytes(out)


def compare_hex(actual_path: str, golden_path: str, *,
                ulp: int = 1, atol: float = 0.001,
                strict: bool = True) -> Tuple[bool, dict]:
    """Compare two FP16 hex dumps. BE bit-pair per verify.py:235.

    Returns (passed: bool, stats: dict).
    Strict mode (D-14): passed iff exact_matches == total_fp16.
    Non-strict: passed iff failures == 0 (within_tolerance allowed).
    """
    a_bytes = _parse_hex(actual_path)
    g_bytes = _parse_hex(golden_path)
    n = min(len(a_bytes), len(g_bytes)) // 2
    exact = 0
    within = 0
    failures = 0
    first_failure = None

    for i in range(n):
        # BE bit-pair: high byte first (verify.py:235; Pitfall 1)
        r_raw = (a_bytes[i * 2] << 8) | a_bytes[i * 2 + 1]
        g_raw = (g_bytes[i * 2] << 8) | g_bytes[i * 2 + 1]
        if r_raw == g_raw:
            exact += 1
            continue
        # Decode for tolerance compare
        r_arr = np.frombuffer(np.uint16(r_raw).tobytes(), dtype=np.float16)
        g_arr = np.frombuffer(np.uint16(g_raw).tobytes(), dtype=np.float16)
        r_val = float(r_arr[0])
        g_val = float(g_arr[0])
        if np.isnan(r_val) or np.isnan(g_val):
            ulp_dist = 0xFFFF
            abs_diff = float('inf')
        else:
            # Signed-magnitude ULP per verify.py:150-158
            r_sm = r_raw if (r_raw & 0x8000) == 0 else -(r_raw & 0x7FFF)
            g_sm = g_raw if (g_raw & 0x8000) == 0 else -(g_raw & 0x7FFF)
            ulp_dist = abs(r_sm - g_sm)
            abs_diff = abs(r_val - g_val)
        if ulp_dist <= ulp or abs_diff <= atol:
            within += 1
        else:
            failures += 1
            if first_failure is None:
                first_failure = (i, r_raw, g_raw)

    stats = dict(exact_matches=exact, within_tolerance=within,
                 failures=failures, total_fp16=n,
                 first_failure=first_failure)
    if strict:
        return (exact == n, stats)
    return (failures == 0, stats)
