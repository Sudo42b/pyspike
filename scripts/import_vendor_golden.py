#!/usr/bin/env python3
"""One-shot vendor _ref.txt -> golden/<op>.hex transform (P6 D-10/D-11/D-12).

Source: vendor/gtx_cpp_reference/test/<OP>/n1s16/data/<kernel>_ref.txt
Dest:   tests/gtx/data/golden/<op>.hex

RESEARCH finding #1: vendor _ref.txt format is byte-identical to existing .hex
(both use ``@<addr>`` directive + 32-byte/line data). Conversion is single-row
truncate (matches mm_basic_n1s16.hex / activation_relu_gelu.hex precedent).

Plan 03 owns this file (D-18 zero-overlap -- Plan 01 does not create a stub).

Usage:
    python3 scripts/import_vendor_golden.py
    python3 scripts/import_vendor_golden.py --verify   # dry-run, list mappings
"""
import argparse
import pathlib
import sys

REPO_ROOT = pathlib.Path(__file__).resolve().parent.parent
VENDOR_TEST = REPO_ROOT / "vendor" / "gtx_cpp_reference" / "test"
PYSPIKE_GOLDEN = REPO_ROOT / "tests" / "gtx" / "data" / "golden"

# Map: vendor_op_dir -> (vendor_kernel_filename_prefix, pyspike_op_name, n_data_lines)
# Plan 03 GREEN-filled per RESEARCH §Vendor Core Op Set Survey (lines 1064-1091).
#
# NAMING NOTES (verified at execution time):
# - SOFT_MAX vendor uses "n1s16_softmax_ref.txt" (NOT "n1s16_soft_max_ref.txt").
# - ADD vendor uses "n1s16_add_vv_ref.txt" (NOT "n1s16_add_ref.txt").
VENDOR_TO_PYSPIKE_OPS: dict = {
    "RELU":       ("n1s16_relu",       "relu",       1),
    "SIGMOID":    ("n1s16_sigmoid",    "sigmoid",    1),
    "TANH":       ("n1s16_tanh",       "tanh",       1),
    "SOFT_MAX":   ("n1s16_softmax",    "softmax",    1),
    "ADD":        ("n1s16_add_vv",     "add_vv",     1),
    "MUL":        ("n1s16_mul",        "mul_vv",     1),
    "SUM":        ("n1s16_sum",        "sum",        1),
    "ABS":        ("n1s16_abs",        "abs",        1),
    "LEAKY_RELU": ("n1s16_leaky_relu", "leaky_relu", 1),
}

# P7 NJIT-04: full 84-op vendor directory list (W2 fix: complete list inlined,
# no ellipsis). Auto-discovery via `_discover_vendor_ops()` cross-validates.
VENDOR_OPS_84: list = [
    "ABS", "ACC", "ADD", "ADD1",
    "ADD_ID", "ADD_REL_POS", "ARANGE", "CLAMP",
    "CONCAT", "CONV_2D", "CONV_TRANSPOSE_1D", "CONV_TRANSPOSE_2D",
    "COS", "CPY", "CUMSUM", "DIAG",
    "DIAG_MASK_INF", "DIAG_MASK_ZERO", "DIV", "DUP",
    "ELU", "EXP", "EXPM1", "FILL",
    "FLOOR", "GATED_LINEAR_ATTN", "GEGLU", "GEGLU_ERF",
    "GEGLU_QUICK", "GELU", "GELU_ERF", "GELU_QUICK",
    "GET_REL_POS", "GET_ROWS", "GROUP_NORM", "HARDSIGMOID",
    "HARDSWISH", "IM2COL", "IM2COL_3D", "L2_NORM",
    "LEAKY_RELU", "LOG", "MEAN", "MUL",
    "MUL_MAT", "MUL_MAT_ID", "NEG", "NORM",
    "OUT_PROD", "PAD", "PAD_REFLECT_1D", "POOL_1D",
    "POOL_2D", "REGLU", "RELU", "REPEAT",
    "RMS_NORM", "ROLL", "ROPE", "ROUND",
    "RWKV_WKV6", "RWKV_WKV7", "SCALE", "SET",
    "SET_ROWS", "SGN", "SIGMOID", "SILU",
    "SIN", "SOFTPLUS", "SOFT_MAX", "SOLVE_TRI",
    "SQR", "STEP", "SUB", "SUM",
    "SWIGLU_OAI", "TANH", "TIMESTEP_EMBEDDING", "TRI",
    "TRUNC", "WIN_PART", "WIN_UNPART", "XIELU",
]
assert len(VENDOR_OPS_84) == 84, (
    "P7 NJIT-04: VENDOR_OPS_84 must have 84 entries, got " + str(len(VENDOR_OPS_84))
)


def _discover_vendor_ops() -> list:
    """Auto-discover all 84 op directories at runtime (verify VENDOR_OPS_84)."""
    return sorted(
        p.name for p in VENDOR_TEST.iterdir()
        if p.is_dir() and p.name != "__pycache__" and p.name[:1].isupper()
    )


def _discover_kernel_filename(op_dir: str):
    """Auto-discover vendor kernel filename for an op directory.

    Returns (kernel_prefix, op_name_for_pyspike) or None if no _ref.txt found.

    kernel_prefix: vendor's filename minus '_ref.txt' (e.g. 'n1s16_relu' for
                   'n1s16_relu_ref.txt')
    op_name_for_pyspike: lowercase vendor dir name (e.g. 'relu') -- matches
                        .elf naming convention used in tests/gtx/data/firmware/
    """
    data_dir = VENDOR_TEST / op_dir / "n1s16" / "data"
    if not data_dir.is_dir():
        return None
    ref_files = sorted(data_dir.glob("n1s16_*_ref.txt"))
    if not ref_files:
        # Fallback: try _result.hex pattern (some vendor ops)
        ref_files = sorted(data_dir.glob("n1s16_*_result.hex"))
    if not ref_files:
        return None
    # Default: pick the first matching file (most ops have a single _ref.txt)
    ref = ref_files[0]
    stem = ref.stem
    if stem.endswith("_ref"):
        kernel_prefix = stem[:-4]
    elif stem.endswith("_result"):
        kernel_prefix = stem[:-7]
    else:
        kernel_prefix = stem
    op_name = op_dir.lower()
    return (kernel_prefix, op_name)


def convert_one(vendor_dir: str, kernel_prefix: str, op_name: str,
                n_lines: int, dry_run: bool = False):
    """Returns (success, message) tuple."""
    # Try exact filename first
    src_ref = VENDOR_TEST / vendor_dir / "n1s16" / "data" / (kernel_prefix + "_ref.txt")
    if not src_ref.exists():
        # Fallback: try alternate naming (some vendor ops use _result.hex)
        alt_ref = VENDOR_TEST / vendor_dir / "n1s16" / "data" / (kernel_prefix + "_result.hex")
        if alt_ref.exists():
            src_ref = alt_ref
        else:
            return (False, "missing: " + str(src_ref))

    dst_hex = PYSPIKE_GOLDEN / (op_name + ".hex")

    addr_line = None
    data_lines = []
    with open(src_ref) as f:
        for line in f:
            line = line.rstrip()
            if not line:
                continue
            if line.startswith('@'):
                if addr_line is None:
                    addr_line = line
                continue
            if line.startswith('#'):
                continue
            data_lines.append(line)
            if len(data_lines) >= n_lines:
                break

    if not data_lines:
        return (False, "no data lines in " + str(src_ref))

    if dry_run:
        return (True, "DRY: would write " + str(dst_hex) + " from " + str(src_ref))

    dst_hex.parent.mkdir(parents=True, exist_ok=True)
    with open(dst_hex, 'w') as f:
        f.write("# Source: vendor/gtx_cpp_reference/test/" + vendor_dir
                + "/n1s16/data/" + src_ref.name + "\n")
        f.write("# Vendor C++ libgtx_npu.so output, ISS-captured. P6 VRF-03.\n")
        f.write("# Single-row truncation (32 bytes / 16 FP16) per P4/P5 precedent.\n")
        if addr_line:
            f.write(addr_line + "\n")
        for dl in data_lines:
            f.write(dl + "\n")
    return (True, "WROTE: " + str(dst_hex) + " (" + str(n_lines) + " line)")


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--verify', action='store_true',
                        help='Dry-run: list mappings without writing files')
    parser.add_argument('--all', action='store_true',
                        help='P7 NJIT-04: import all 84 vendor ops (Plan 05 GREEN-fills)')
    args = parser.parse_args(argv)

    if args.all:
        # P7 NJIT-04 Plan 05 GREEN: walk all 84 vendor op directories and
        # convert each via _discover_kernel_filename + convert_one. Some ops
        # may legitimately lack _ref.txt assets -- skip with explicit count.
        ok_count = 0
        skip_count = 0
        for op_dir in VENDOR_OPS_84:
            result = _discover_kernel_filename(op_dir)
            if result is None:
                print("SKIP: " + op_dir + " (no _ref.txt found)")
                skip_count += 1
                continue
            kernel_prefix, op_name = result
            ok, msg = convert_one(op_dir, kernel_prefix, op_name,
                                  n_lines=1, dry_run=args.verify)
            print(msg)
            if ok:
                ok_count += 1
            else:
                skip_count += 1
        print()
        print("--all summary: " + str(ok_count) + " converted, "
              + str(skip_count) + " skipped/missing.")
        return 0

    ok_count = 0
    skip_count = 0
    for vendor_dir, (kernel_prefix, op_name, n_lines) in VENDOR_TO_PYSPIKE_OPS.items():
        ok, msg = convert_one(vendor_dir, kernel_prefix, op_name, n_lines,
                              dry_run=args.verify)
        print(msg)
        if ok:
            ok_count += 1
        else:
            skip_count += 1
    print()
    print("Summary: " + str(ok_count) + " converted, " + str(skip_count) + " skipped/missing.")
    return 0 if (ok_count >= 8 or args.verify) else 1


if __name__ == '__main__':
    sys.exit(main())
