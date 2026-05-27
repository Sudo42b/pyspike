"""NumPy fallback verification for YOLOv8-class ops in ggml_compute.py.

Validates the handlers ggml-rpc relies on when no pyspike kernel exists:
  - GGML_OP_CONT          (contiguous copy after permute/view)
  - GGML_OP_CONV_2D       (standalone direct conv; YOLO graphs may emit)
  - GGML_OP_REPEAT        (generalised np.tile, was broadcast_to only)

Run:
    uv run --no-sync python examples/ggml-rpc-server/tests/test_yolo_fallback.py
"""
from __future__ import annotations

import os
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
SERVER_DIR = HERE.parent
sys.path.insert(0, str(SERVER_DIR))

import ggml_compute as gc  # noqa: E402
import op_registry as opr  # noqa: E402
import wire_protocol as wp  # noqa: E402


def _tensor(op: int, ne: tuple, op_params: tuple = (), srcs: tuple = ()) -> wp.RpcTensor:
    """Build a minimal RpcTensor shell for compute() — only the fields gc.compute reads."""
    ne_full = tuple(ne) + (1,) * (4 - len(ne))
    return wp.RpcTensor(
        id=0, type=0,
        ne=ne_full, nb=(0, 0, 0, 0),
        op=op,
        op_params=tuple(op_params) + (0,) * (16 - len(op_params)),
        src=tuple(srcs) + (0,) * (10 - len(srcs)),
        data=0,
    )


def case_cont() -> tuple[str, bool, str]:
    # Permute-then-cont pattern: src has a non-contiguous numpy view.
    base = np.arange(24, dtype=np.float32).reshape(2, 3, 4)
    permuted = base.transpose(2, 0, 1)              # (4, 2, 3) non-contiguous
    node = _tensor(opr.GGML_OP_CONT, ne=permuted.shape[::-1])
    got = gc.compute(node, [permuted])
    ok = np.array_equal(got, permuted)
    return ("CONT", ok, f"shape={got.shape}, contiguous={got.flags['C_CONTIGUOUS']}")


def case_conv2d() -> tuple[str, bool, str]:
    # Small standalone CONV_2D: input (B=1, IC=3, IH=4, IW=4), kernel (OC=2, IC=3, KH=3, KW=3)
    # stride 1, pad 1 → output (1, 2, 4, 4).
    rng = np.random.default_rng(0)
    kernel = rng.standard_normal((2, 3, 3, 3)).astype(np.float32)
    inp = rng.standard_normal((1, 3, 4, 4)).astype(np.float32)
    node = _tensor(
        opr.GGML_OP_CONV_2D,
        ne=(4, 4, 2, 1),                            # (W, H, OC, B) in ggml ne order
        op_params=(1, 1, 1, 1, 1, 1),               # s0,s1,p0,p1,d0,d1
        srcs=(1, 2),
    )
    got = gc.compute(node, [kernel, inp])

    # Reference: explicit double loop.
    OC, IC, KH, KW = kernel.shape
    B, _, IH, IW = inp.shape
    s0 = s1 = 1
    p0 = p1 = 1
    OH = (IH + 2 * p1 - (KH - 1) - 1) // s1 + 1
    OW = (IW + 2 * p0 - (KW - 1) - 1) // s0 + 1
    bp = np.pad(inp, ((0, 0), (0, 0), (p1, p1), (p0, p0)))
    ref = np.zeros((B, OC, OH, OW), dtype=np.float32)
    for oc in range(OC):
        for oh in range(OH):
            for ow in range(OW):
                patch = bp[:, :, oh:oh + KH, ow:ow + KW]
                ref[:, oc, oh, ow] = (patch * kernel[oc][None]).sum(axis=(1, 2, 3))

    max_diff = float(np.abs(got - ref).max())
    ok = max_diff < 1e-4
    return ("CONV_2D", ok, f"max_diff={max_diff:.2e}, shape={got.shape}")


def case_repeat_tile() -> tuple[str, bool, str]:
    # The case the old broadcast_to handler failed on: src (4, 4) → dst (8, 8).
    src = np.arange(16, dtype=np.float32).reshape(4, 4)
    # ggml ne order: (W, H, C, B) — innermost = ne[0]; numpy view is reversed
    # (H, W). So dst.ne = (8, 8, 1, 1) maps back to numpy (8, 8).
    node = _tensor(opr.GGML_OP_REPEAT, ne=(8, 8, 1, 1), srcs=(1,))
    got = gc.compute(node, [src])
    ref = np.tile(src, (2, 2))
    ok = np.array_equal(got, ref)
    return ("REPEAT_tile", ok, f"shape={got.shape}")


def case_repeat_broadcast() -> tuple[str, bool, str]:
    # broadcast-style: src (1, 4) → dst (3, 4). Old handler also covered this.
    src = np.arange(4, dtype=np.float32).reshape(1, 4)
    node = _tensor(opr.GGML_OP_REPEAT, ne=(4, 3, 1, 1), srcs=(1,))
    got = gc.compute(node, [src])
    ref = np.broadcast_to(src, (3, 4)).copy()
    ok = np.array_equal(got, ref)
    return ("REPEAT_bcast", ok, f"shape={got.shape}")


CASES = [case_cont, case_conv2d, case_repeat_tile, case_repeat_broadcast]


def main() -> int:
    results = []
    for fn in CASES:
        try:
            name, ok, info = fn()
        except Exception as e:
            name, ok, info = (fn.__name__, False, f"EXC: {e}")
        marker = "PASS" if ok else "FAIL"
        print(f"{marker} {name:14s} {info}")
        results.append((name, ok))
    n_pass = sum(1 for _, ok in results if ok)
    print("=" * 50)
    print(f"{n_pass}/{len(results)} cases PASS")
    return 0 if n_pass == len(results) else 1


if __name__ == "__main__":
    sys.exit(main())
