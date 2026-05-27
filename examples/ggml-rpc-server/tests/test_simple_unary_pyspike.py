"""Direct-runner verification for the 5 SIMPLE_UNARY pyspike kernels.

Bypasses the RPC server — calls `pyspike_runner.run_*_fp16` directly and
compares against a NumPy reference. Builds & runs each .elf once, then prints
a PASS/FAIL summary.

Usage:
    uv run --no-sync python examples/ggml-rpc-server/tests/test_simple_unary_pyspike.py
    uv run --no-sync python examples/ggml-rpc-server/tests/test_simple_unary_pyspike.py --only sqr
"""
from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
SERVER_DIR = HERE.parent
sys.path.insert(0, str(SERVER_DIR))

import pyspike_runner as psr  # noqa: E402


def _to_fp16_bytes(arr: np.ndarray) -> bytes:
    return arr.astype(np.float16).tobytes()


def _from_fp16(bt: bytes, shape: tuple) -> np.ndarray:
    return np.frombuffer(bt, dtype=np.float16).reshape(shape).copy()


def _max_diff(expected: np.ndarray, actual: np.ndarray) -> float:
    return float(np.max(np.abs(expected.astype(np.float32) -
                                actual.astype(np.float32))))


def case_sqr(rng) -> tuple[str, float, float]:
    x = rng.uniform(-3.0, 3.0, (16, 16)).astype(np.float16)
    out = psr.run_sqr_fp16(_to_fp16_bytes(x), width=16)
    got = _from_fp16(out, x.shape)
    ref = (x.astype(np.float32) * x.astype(np.float32)).astype(np.float16)
    return ("SQR", _max_diff(ref, got), 1e-1)


def case_sum_rows(rng) -> tuple[str, float, float]:
    x = rng.uniform(-1.0, 1.0, (16, 32)).astype(np.float16)
    out = psr.run_sum_rows_fp16(_to_fp16_bytes(x), width=32)
    got = _from_fp16(out, (16,))
    ref = x.astype(np.float32).sum(axis=-1).astype(np.float16)
    return ("SUM_ROWS", _max_diff(ref, got), 0.5)


def case_norm(rng) -> tuple[str, float, float]:
    x = rng.uniform(-2.0, 2.0, (16, 32)).astype(np.float16)
    eps = 1e-5
    out = psr.run_norm_fp16(_to_fp16_bytes(x), width=32, eps=eps)
    got = _from_fp16(out, x.shape)
    xf = x.astype(np.float32)
    mean = xf.mean(axis=-1, keepdims=True)
    var = ((xf - mean) ** 2).mean(axis=-1, keepdims=True)
    ref = ((xf - mean) / np.sqrt(var + eps)).astype(np.float16)
    return ("NORM", _max_diff(ref, got), 0.1)


def case_group_norm(rng) -> tuple[str, float, float]:
    x = rng.uniform(-2.0, 2.0, (16, 32)).astype(np.float16)
    eps = 1e-5
    out = psr.run_group_norm_fp16(_to_fp16_bytes(x), width=32, eps=eps)
    got = _from_fp16(out, x.shape)
    xf = x.astype(np.float32)
    mean = xf.mean()
    var = ((xf - mean) ** 2).mean()
    ref = ((xf - mean) / np.sqrt(var + eps)).astype(np.float16)
    return ("GROUP_NORM", _max_diff(ref, got), 0.1)


def case_scale(rng) -> tuple[str, float, float]:
    x = rng.uniform(-3.0, 3.0, (16, 32)).astype(np.float16)
    scale = 2.5
    out = psr.run_scale_fp16(_to_fp16_bytes(x), width=32, scale=scale)
    got = _from_fp16(out, x.shape)
    ref = (x.astype(np.float32) * scale).astype(np.float16)
    return ("SCALE", _max_diff(ref, got), 1e-1)


def case_ceil(rng) -> tuple[str, float, float]:
    x = rng.uniform(-3.0, 3.0, (16, 32)).astype(np.float16)
    out = psr.run_ceil_fp16(_to_fp16_bytes(x), width=32)
    got = _from_fp16(out, x.shape)
    ref = np.ceil(x.astype(np.float32)).astype(np.float16)
    return ("CEIL", _max_diff(ref, got), 1e-1)


def case_expm1(rng) -> tuple[str, float, float]:
    # x in [-1, 1] keeps exp(x) within fp16's safe range with room for the -1.
    x = rng.uniform(-1.0, 1.0, (16, 32)).astype(np.float16)
    out = psr.run_expm1_fp16(_to_fp16_bytes(x), width=32)
    got = _from_fp16(out, x.shape)
    ref = (np.exp(x.astype(np.float32)) - 1.0).astype(np.float16)
    return ("EXPM1", _max_diff(ref, got), 0.05)


def case_clamp(rng) -> tuple[str, float, float]:
    x = rng.uniform(-3.0, 3.0, (16, 32)).astype(np.float16)
    min_v, max_v = -1.0, 1.0
    out = psr.run_clamp_fp16(_to_fp16_bytes(x), width=32,
                              min_val=min_v, max_val=max_v)
    got = _from_fp16(out, x.shape)
    ref = np.clip(x.astype(np.float32), min_v, max_v).astype(np.float16)
    return ("CLAMP", _max_diff(ref, got), 1e-1)


def case_mean(rng) -> tuple[str, float, float]:
    # HEIGHT % 16 == 0 required → (16, 32) gives HEIGHT=16.
    x = rng.uniform(-1.0, 1.0, (16, 32)).astype(np.float16)
    out = psr.run_mean_fp16(_to_fp16_bytes(x), width=32)
    got = _from_fp16(out, (16,))
    ref = x.astype(np.float32).mean(axis=-1).astype(np.float16)
    return ("MEAN", _max_diff(ref, got), 0.05)


def case_sum(rng) -> tuple[str, float, float]:
    # Total reduction → 1 fp16 scalar. Keep magnitudes small so the sum stays
    # well within fp16's normal range.
    x = rng.uniform(-0.05, 0.05, (16, 32)).astype(np.float16)
    out = psr.run_sum_fp16(_to_fp16_bytes(x), width=32)
    got = _from_fp16(out, (1,))
    ref = np.array([x.astype(np.float32).sum()], dtype=np.float16)
    return ("SUM", _max_diff(ref, got), 0.5)


def case_arange(rng) -> tuple[str, float, float]:
    n = 64  # multiple of 8 (COLS=8 in firmware)
    out = psr.run_arange_fp16(n)
    got = _from_fp16(out, (n,))
    ref = np.arange(0, n, dtype=np.float16)
    return ("ARANGE", _max_diff(ref, got), 0.0)


def case_tri(rng) -> tuple[str, float, float]:
    # Square 8x8 with tri_type=0 (UPPER_DIAG): keep upper incl. diagonal,
    # zero strictly-lower entries.
    x = rng.uniform(-2.0, 2.0, (8, 8)).astype(np.float16)
    out = psr.run_tri_fp16(_to_fp16_bytes(x), width=8, tri_type=0)
    got = _from_fp16(out, x.shape)
    ref = np.triu(x.astype(np.float32), k=0).astype(np.float16)
    return ("TRI", _max_diff(ref, got), 0.0)


def case_repeat(rng) -> tuple[str, float, float]:
    # 2D tile: (4, 4) → (8, 8), each dim x2. Use distinct integer values so
    # any addressing bug shows up as a clear element mismatch.
    x = np.arange(16, dtype=np.float16).reshape(4, 4)
    src_ne = (4, 4, 1, 1)
    dst_ne = (8, 8, 1, 1)
    out = psr.run_repeat_fp16(x.tobytes(), src_ne, dst_ne)
    got = _from_fp16(out, (8, 8))
    ref = np.tile(x, (2, 2))
    return ("REPEAT", _max_diff(ref, got), 0.0)


def case_pad(rng) -> tuple[str, float, float]:
    # 4x8 src, pad +1 col / +1 row → 5x9 dst with zeros on right + bottom.
    x = rng.uniform(-2.0, 2.0, (4, 8)).astype(np.float16)
    pad_right, pad_bottom = 1, 1
    out = psr.run_pad_fp16(x.tobytes(), src_rows=4, src_cols=8,
                            pad_right=pad_right, pad_bottom=pad_bottom)
    got = _from_fp16(out, (4 + pad_bottom, 8 + pad_right))
    ref = np.pad(x, ((0, pad_bottom), (0, pad_right)),
                 mode="constant", constant_values=0)
    return ("PAD", _max_diff(ref, got), 0.0)


def case_concat(rng) -> tuple[str, float, float]:
    # Two (4, 8) FP16 tensors → (4, 16) along axis=0 (innermost dim).
    a = rng.uniform(-1.0, 1.0, (4, 8)).astype(np.float16)
    b = rng.uniform(-1.0, 1.0, (4, 8)).astype(np.float16)
    out = psr.run_concat_fp16(a.tobytes(), b.tobytes(),
                               src0_cols=8, src1_cols=8, rows=4)
    got = _from_fp16(out, (4, 16))
    ref = np.concatenate([a, b], axis=-1)
    return ("CONCAT", _max_diff(ref, got), 0.0)


def case_pool_2d_avg(rng) -> tuple[str, float, float]:
    # AvgPool 2x2 stride 2: input (8, 8) → output (4, 4).
    x = rng.uniform(-1.0, 1.0, (8, 8)).astype(np.float16)
    out = psr.run_pool_2d_avg_fp16(x.tobytes(), in_h=8, in_w=8,
                                    k_h=2, k_w=2, s_h=2, s_w=2)
    got = _from_fp16(out, (4, 4))
    # Reference: numpy mean over 2x2 windows.
    xf = x.astype(np.float32).reshape(4, 2, 4, 2).mean(axis=(1, 3))
    ref = xf.astype(np.float16)
    return ("POOL_2D_AVG", _max_diff(ref, got), 0.05)


def case_conv_2d(rng) -> tuple[str, float, float]:
    # IC=1 OC=1 conv: input (5, 5), kernel (3, 3), stride 1, pad 0 → output (3, 3).
    input_x = rng.uniform(-1.0, 1.0, (5, 5)).astype(np.float16)
    kernel = rng.uniform(-1.0, 1.0, (3, 3)).astype(np.float16)
    out = psr.run_conv_2d_fp16(kernel.tobytes(), input_x.tobytes(),
                                in_h=5, in_w=5, k_h=3, k_w=3)
    got = _from_fp16(out, (3, 3))
    # Reference: direct 2D convolution (correlation), single channel.
    xf = input_x.astype(np.float32)
    kf = kernel.astype(np.float32)
    ref = np.zeros((3, 3), dtype=np.float32)
    for oh in range(3):
        for ow in range(3):
            ref[oh, ow] = (xf[oh:oh + 3, ow:ow + 3] * kf).sum()
    ref = ref.astype(np.float16)
    return ("CONV_2D", _max_diff(ref, got), 0.05)


def case_im2col(rng) -> tuple[str, float, float]:
    # Single-channel im2col: input (5, 5) with 3x3 kernel, stride 1.
    # Output: (OH*OW, KH*KW) = (3*3, 9) = (9, 9).
    x = rng.uniform(-1.0, 1.0, (5, 5)).astype(np.float16)
    out = psr.run_im2col_fp16(x.tobytes(), in_h=5, in_w=5,
                               k_h=3, k_w=3, stride=1)
    got = _from_fp16(out, (9, 9))
    # Reference: collect 3x3 patches with stride 1.
    ref = np.zeros((9, 9), dtype=np.float16)
    for oh in range(3):
        for ow in range(3):
            patch = x[oh:oh + 3, ow:ow + 3]
            ref[oh * 3 + ow] = patch.flatten()
    return ("IM2COL", _max_diff(ref, got), 0.0)


CASES = {
    "sqr":        case_sqr,
    "sum_rows":   case_sum_rows,
    "norm":       case_norm,
    "group_norm": case_group_norm,
    "scale":      case_scale,
    "ceil":       case_ceil,
    "expm1":      case_expm1,
    "clamp":      case_clamp,
    "mean":       case_mean,
    "sum":        case_sum,
    "arange":     case_arange,
    "tri":        case_tri,
    "repeat":     case_repeat,
    "pad":        case_pad,
    "concat":     case_concat,
    "pool_2d_avg": case_pool_2d_avg,
    "im2col":     case_im2col,
    "conv_2d":    case_conv_2d,
}


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--only", choices=list(CASES),
                   help="run a single case (default: all)")
    args = p.parse_args()

    rng = np.random.default_rng(0)
    selected = [args.only] if args.only else list(CASES)

    results = []
    for key in selected:
        print(f"--- running {key} ---", flush=True)
        try:
            name, max_diff, tol = CASES[key](rng)
            ok = max_diff <= tol
            results.append((name, ok, max_diff, tol))
            print(f"{'PASS' if ok else 'FAIL'} {name:12s} "
                  f"max_diff={max_diff:.4g} tol={tol}", flush=True)
        except Exception as e:
            print(f"ERROR {key}: {e}", flush=True)
            results.append((key.upper(), False, float("inf"), 0.0))

    print()
    print("=" * 50)
    n_pass = sum(1 for _, ok, _, _ in results if ok)
    for name, ok, md, tol in results:
        marker = "PASS" if ok else "FAIL"
        print(f"  {marker} {name:12s} max_diff={md:.4g} (tol {tol})")
    print(f"{n_pass}/{len(results)} cases PASS")
    return 0 if n_pass == len(results) else 1


if __name__ == "__main__":
    sys.exit(main())
