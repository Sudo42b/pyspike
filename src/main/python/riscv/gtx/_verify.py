from __future__ import annotations

import argparse
import importlib.resources as r
import sys
from pathlib import Path
from typing import Optional, Tuple

import torch

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
        # Decode for tolerance compare. `r_raw`/`g_raw` are uint16 bit patterns;
        # use int.to_bytes (LE host) → torch.frombuffer to get the matching FP16.
        # NOTE: `torch.uint16` is a dtype, not a constructor — do not call it.
        r_arr = torch.frombuffer(r_raw.to_bytes(2, 'little'), dtype=torch.float16)
        g_arr = torch.frombuffer(g_raw.to_bytes(2, 'little'), dtype=torch.float16)
        r_val = r_arr[0].item()
        g_val = g_arr[0].item()
        if r_val != r_val or g_val != g_val:  # NaN check (FP16 already .item()-ed)
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

    stats = dict(
        exact_matches=exact, within_tolerance=within,
        failures=failures, total_fp16=n,
        first_failure=first_failure,
        mismatches=failures,
        first_mismatch=(first_failure[0] * 2) if first_failure else None,
        size_result=len(a_bytes), size_golden=len(g_bytes),
        total_bytes=min(len(a_bytes), len(g_bytes)),
        trailing_bytes=min(len(a_bytes), len(g_bytes)) % 2,
    )
    return (failures == 0, stats)


# ---------------------------------------------------------------------------
# Section 2 — Helpers (D-14)
# ---------------------------------------------------------------------------

def bundled_elfs() -> list:
    try:
        fw_dir = r.files('riscv.gtx').joinpath('data', 'firmware')
        return sorted(Path(str(p)) for p in fw_dir.iterdir() if str(p).endswith('.elf'))
    except (FileNotFoundError, ModuleNotFoundError, AttributeError, NotADirectoryError):
        return []

def load_golden(name: str) -> bytes:
    """Load golden hex file by op name (without .hex suffix). D-14."""
    target = r.files('riscv.gtx').joinpath('data', 'golden', f"{name}.hex")
    return target.read_bytes()

def _print_report_fp16(stats: dict, result_file: str, golden_file: str,
                       ulp: int, atol: float, strict: bool) -> bool:
    """Print FP16 comparison report (vendor verify.py:312-343 + --strict)."""
    print("=" * 60)
    print("DDR Hex Verification Report (FP16)")
    print("=" * 60)
    print(f"  Result file : {result_file} ({stats['size_result']} bytes)")
    print(f"  Golden file : {golden_file} ({stats['size_golden']} bytes)")
    print(f"  ULP tolerance  : {ulp}")
    print(f"  Abs tolerance  : {atol}")
    print(f"  Strict mode    : {strict}")
    print(f"  FP16 elements  : {stats['total_fp16']}")
    print(f"  Exact matches  : {stats['exact_matches']}")
    print(f"  Within tolerance: {stats['within_tolerance']}")
    print(f"  Mismatches     : {stats['failures']}")
    if stats['size_result'] != stats['size_golden']:
        print(f"  WARNING: size mismatch (result={stats['size_result']}, "
              f"golden={stats['size_golden']})")
    if stats['first_failure'] is not None:
        idx, r_raw, g_raw = stats['first_failure']
        print(f"  First mismatch at FP16 idx {idx}: "
              f"result=0x{r_raw:04x} golden=0x{g_raw:04x}")
    print("-" * 60)
    # P8 B1 (2026-05-11): strict and non-strict both gate on failures==0
    # (ULP≤1 + atol≤0.001 allowed per CLAUDE.md). See compare_hex docstring.
    passed = stats['failures'] == 0
    print(f"  Result: {'PASS' if passed else 'FAIL'}")
    print("=" * 60)
    return passed

def main(argv: Optional[list] = None) -> int:
    """Vendor verify.py:350-388 1:1 + --strict (D-03)."""
    parser = argparse.ArgumentParser(
        description='DDR hex diff tool with FP16 rounding tolerance',
        epilog='Example: pyspike-verify result.hex golden.hex --fp16 --strict --ulp 1 --atol 0.001'
    )
    parser.add_argument('result', help='Result hex dump file')
    parser.add_argument('golden', help='Golden reference hex dump file')
    parser.add_argument('--ulp', type=int, default=1,
                        help='FP16 ULP tolerance (default: 1)')
    parser.add_argument('--atol', type=float, default=0.001,
                        help='Absolute tolerance (default: 0.001)')
    parser.add_argument('--fp16', action='store_true',
                        help='Interpret data as FP16 pairs and compare with tolerance')
    parser.add_argument('--strict', action='store_true',
                        help='Strict mode (P8 B1): PASS iff failures == 0 '
                             '(within ULP=1 + atol=0.001; same as default gate, '
                             'preserved for backward-compat)')
    args = parser.parse_args(argv)

    passed, stats = compare_hex(args.result, args.golden,
                                ulp=args.ulp, atol=args.atol,
                                strict=args.strict)
    if args.fp16 or args.strict:
        _print_report_fp16(stats, args.result, args.golden,
                           args.ulp, args.atol, args.strict)
    else:
        print(f"  bytes={stats['total_bytes']} exact={stats['exact_matches']} "
              f"failures={stats['failures']} -> "
              f"{'PASS' if passed else 'FAIL'}")
    return 0 if passed else 1


if __name__ == '__main__':
    sys.exit(main())
