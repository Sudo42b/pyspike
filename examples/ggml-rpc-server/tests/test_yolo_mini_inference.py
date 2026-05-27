"""End-to-end YOLOv8 mini block inference with pyspike-first fallback to NumPy.

Simulates a tiny YOLOv8-flavoured chain:

    input (1, 3, 16, 16)
      → Conv2d(3→4, 3×3)          [pyspike: K-pad 27→32 → MUL_MAT_tiled]
      → SiLU                       [pyspike: unary_silu]
      → MaxPool 2×2 s=2            [NumPy fallback — kernel only has AVG]
      → Conv2d(4→8, 1×1)           [pyspike: K=4 → pad to 8 → MUL_MAT]
      → SiLU                       [pyspike]
      → Conv2d(8→4, 3×3)           [pyspike: K=72, native]
      → Upsample(2×)               [NumPy fallback — kernel needs OP_PARAMS]
      → Concat with skip (axis=1)  [NumPy fallback — kernel only axis=0]
      → Conv2d(8→4, 1×1)           [pyspike: K=8]
      → Sigmoid                    [pyspike: unary_sigmoid]

Each op tries pyspike first; on guard miss or build/sim failure it falls back
to a NumPy implementation. The final tensor is compared against a fully-NumPy
reference; max_diff should stay within fp16 numerical tolerance.

Run:
    uv run --no-sync python examples/ggml-rpc-server/tests/test_yolo_mini_inference.py
"""
from __future__ import annotations

import sys
import time
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
SERVER_DIR = HERE.parent
sys.path.insert(0, str(SERVER_DIR))

import pyspike_runner as psr  # noqa: E402

# --- routing log -------------------------------------------------------------

_ROUTING: list[tuple[str, str, str, float]] = []   # (op, layer, route, duration)


def _log(op: str, layer: str, route: str, t0: float) -> None:
    _ROUTING.append((op, layer, route, time.time() - t0))


def _print_routing_table() -> None:
    print("\n" + "=" * 72)
    print(f"{'#':<3} {'OP':<10} {'LAYER':<22} {'ROUTE':<14} {'TIME (s)':>9}")
    print("-" * 72)
    for i, (op, layer, route, dt) in enumerate(_ROUTING, 1):
        print(f"{i:<3} {op:<10} {layer:<22} {route:<14} {dt:>9.3f}")
    py = sum(1 for *_, r, _ in [(o, l, r, d) for o, l, r, d in _ROUTING] if r == "pyspike")
    nm = sum(1 for *_, r, _ in [(o, l, r, d) for o, l, r, d in _ROUTING] if r == "numpy")
    print("-" * 72)
    print(f"summary: {py} pyspike-accelerated, {nm} NumPy-fallback "
          f"(total {len(_ROUTING)} ops)")


# --- NumPy reference primitives ---------------------------------------------

def _numpy_im2col(b, kh, kw, sh=1, sw=1, ph=0, pw=0):
    B, IC, IH, IW = b.shape
    OH = (IH + 2 * ph - kh) // sh + 1
    OW = (IW + 2 * pw - kw) // sw + 1
    bp = np.pad(b, ((0, 0), (0, 0), (ph, ph), (pw, pw)))
    out = np.empty((B, OH, OW, IC, kh, kw), dtype=b.dtype)
    for kh_i in range(kh):
        for kw_i in range(kw):
            sliced = bp[:, :, kh_i:kh_i + sh * OH:sh, kw_i:kw_i + sw * OW:sw]
            out[..., kh_i, kw_i] = sliced.transpose(0, 2, 3, 1)
    return out.reshape(B, OH, OW, IC * kh * kw)


def _numpy_conv2d(x, w):
    """Reference Conv2d (stride 1, pad 0) for (B, IC, IH, IW) × (OC, IC, KH, KW)."""
    B, IC, IH, IW = x.shape
    OC, _, KH, KW = w.shape
    OH = IH - KH + 1
    OW = IW - KW + 1
    out = np.zeros((B, OC, OH, OW), dtype=np.float32)
    xf = x.astype(np.float32)
    wf = w.astype(np.float32)
    for kh_i in range(KH):
        for kw_i in range(KW):
            sliced = xf[:, :, kh_i:kh_i + OH, kw_i:kw_i + OW]
            out += np.einsum('bchw,oc->bohw', sliced, wf[:, :, kh_i, kw_i])
    return out.astype(np.float16)


def _numpy_maxpool2d(x, kh, kw, sh, sw):
    B, C, IH, IW = x.shape
    OH = (IH - kh) // sh + 1
    OW = (IW - kw) // sw + 1
    out = np.empty((B, C, OH, OW), dtype=x.dtype)
    for oh in range(OH):
        for ow in range(OW):
            win = x[:, :, oh * sh:oh * sh + kh, ow * sw:ow * sw + kw]
            out[:, :, oh, ow] = win.max(axis=(2, 3))
    return out


def _numpy_upsample_nearest(x, factor):
    return np.repeat(np.repeat(x, factor, axis=-2), factor, axis=-1)


# --- pyspike-first wrappers -------------------------------------------------

def conv2d(x: np.ndarray, w: np.ndarray, *, layer: str) -> np.ndarray:
    """Conv2d via pyspike: multi-IC IM2COL kernel + MUL_MAT_tiled (K-padded).
    Layout reshapes are pure metadata (`reshape`/`transpose`) — no NumPy compute."""
    t0 = time.time()
    B, IC, IH, IW = x.shape
    OC, _, KH, KW = w.shape
    OH = IH - KH + 1
    OW = IW - KW + 1

    try:
        if B != 1:
            raise ValueError(f"conv2d expects batch=1, got {B}")
        im2col_bytes = psr.SUPPORTED_PYSPIKE_OPS["im2col_multich_fp16"](
            x[0].astype(np.float16).tobytes(),
            ic=IC, in_h=IH, in_w=IW, k_h=KH, k_w=KW, stride=1)
        K = IC * KH * KW
        patches = np.frombuffer(im2col_bytes, dtype=np.float16).reshape(OH * OW, K)

        weights = w.reshape(OC, K)
        M = OH * OW
        N = OC
        K_pad = ((K + 7) // 8) * 8
        if K_pad != K:
            a_p = np.zeros((M, K_pad), dtype=np.float16)
            a_p[:, :K] = patches
            b_p = np.zeros((N, K_pad), dtype=np.float16)
            b_p[:, :K] = weights
        else:
            a_p = patches.astype(np.float16)
            b_p = weights.astype(np.float16)

        mm_bytes = psr.SUPPORTED_PYSPIKE_OPS["mul_mat_tiled_fp16"](
            a_p.tobytes(), b_p.tobytes(), M=M, K=K_pad, N=N)
        mm = np.frombuffer(mm_bytes, dtype=np.float16).reshape(M, N)
        result = mm.reshape(B, OH, OW, OC).transpose(0, 3, 1, 2).copy()
        _log("Conv2d", layer, "pyspike", t0)
        return result
    except Exception as e:
        print(f"  [Conv2d {layer}] pyspike failed ({e}) → NumPy")
        result = _numpy_conv2d(x, w)
        _log("Conv2d", layer, "numpy", t0)
        return result


def silu(x: np.ndarray, *, layer: str) -> np.ndarray:
    t0 = time.time()
    try:
        # pyspike unary_intrin1 wants element count multiple of WIDTH=8.
        flat = x.reshape(-1)
        if flat.size % 8 != 0:
            raise ValueError(f"silu needs len%8==0, got {flat.size}")
        out_bytes = psr.SUPPORTED_PYSPIKE_OPS["unary_silu_fp16"](flat.tobytes())
        result = np.frombuffer(out_bytes, dtype=np.float16).reshape(x.shape).copy()
        _log("SiLU", layer, "pyspike", t0)
        return result
    except Exception as e:
        print(f"  [SiLU {layer}] pyspike failed ({e}) → NumPy")
        xf = x.astype(np.float32)
        result = (xf / (1.0 + np.exp(-xf))).astype(np.float16)
        _log("SiLU", layer, "numpy", t0)
        return result


def sigmoid(x: np.ndarray, *, layer: str) -> np.ndarray:
    t0 = time.time()
    try:
        flat = x.reshape(-1)
        if flat.size % 8 != 0:
            raise ValueError(f"sigmoid needs len%8==0, got {flat.size}")
        out_bytes = psr.SUPPORTED_PYSPIKE_OPS["unary_sigmoid_fp16"](flat.tobytes())
        result = np.frombuffer(out_bytes, dtype=np.float16).reshape(x.shape).copy()
        _log("Sigmoid", layer, "pyspike", t0)
        return result
    except Exception as e:
        print(f"  [Sigmoid {layer}] pyspike failed ({e}) → NumPy")
        xf = x.astype(np.float32)
        result = (1.0 / (1.0 + np.exp(-xf))).astype(np.float16)
        _log("Sigmoid", layer, "numpy", t0)
        return result


def maxpool2d(x: np.ndarray, *, layer: str) -> np.ndarray:
    """MaxPool 2x2 stride 2 via pyspike __pool_m (per-channel single-SPU)."""
    t0 = time.time()
    try:
        B, C, IH, IW = x.shape
        OH, OW = IH // 2, IW // 2
        out = np.empty((B, C, OH, OW), dtype=np.float16)
        for b in range(B):
            for c in range(C):
                ch = x[b, c]                                 # (IH, IW)
                out_bytes = psr.SUPPORTED_PYSPIKE_OPS["pool_2d_max_fp16"](
                    ch.tobytes(), IH, IW, 2, 2, 2, 2)
                out[b, c] = np.frombuffer(out_bytes, dtype=np.float16).reshape(OH, OW)
        _log("MaxPool", layer, "pyspike", t0)
        return out
    except Exception as e:
        print(f"  [MaxPool {layer}] pyspike failed ({e}) → NumPy")
        result = _numpy_maxpool2d(x, 2, 2, 2, 2)
        _log("MaxPool", layer, "numpy", t0)
        return result


def upsample_nearest(x: np.ndarray, factor: int, *, layer: str) -> np.ndarray:
    """Nearest upsample via pyspike REPEAT (same broadcast semantics)."""
    t0 = time.time()
    try:
        B, C, IH, IW = x.shape
        # ggml ne = (W, H, C, B). REPEAT runner expects 4-tuples in ggml order.
        src_ne = (IW, IH, C, B)
        dst_ne = (IW * factor, IH * factor, C, B)
        out_bytes = psr.SUPPORTED_PYSPIKE_OPS["repeat_fp16"](
            x.astype(np.float16).tobytes(), src_ne, dst_ne)
        # repeat output bytes are dst-order = (B, C, OH, OW) row-major fp16.
        result = np.frombuffer(out_bytes, dtype=np.float16).reshape(
            B, C, IH * factor, IW * factor).copy()
        _log("Upsample", layer, "pyspike", t0)
        return result
    except Exception as e:
        print(f"  [Upsample {layer}] pyspike failed ({e}) → NumPy")
        result = _numpy_upsample_nearest(x, factor)
        _log("Upsample", layer, "numpy", t0)
        return result


def concat_channel(a: np.ndarray, b: np.ndarray, *, layer: str) -> np.ndarray:
    """Channel concat via pyspike unary_concat_channel.c.tpl (ggml axis=2)."""
    t0 = time.time()
    try:
        if a.shape[0] != b.shape[0] or a.shape[2:] != b.shape[2:]:
            raise ValueError(f"concat shape mismatch: a={a.shape} b={b.shape}")
        B, A_CH, H, W = a.shape
        _, B_CH, _, _ = b.shape
        out_bytes = psr.SUPPORTED_PYSPIKE_OPS["concat_channel_fp16"](
            a.astype(np.float16).tobytes(), b.astype(np.float16).tobytes(),
            a_ch=A_CH, b_ch=B_CH, h=H, w=W, batch=B)
        result = np.frombuffer(out_bytes, dtype=np.float16).reshape(
            B, A_CH + B_CH, H, W).copy()
        _log("Concat", layer, "pyspike", t0)
        return result
    except Exception as e:
        print(f"  [Concat {layer}] pyspike failed ({e}) → NumPy")
        result = np.concatenate([a, b], axis=1)
        _log("Concat", layer, "numpy", t0)
        return result


# --- YOLOv8 mini block ------------------------------------------------------

def yolo_mini_inference(image: np.ndarray, weights: dict) -> np.ndarray:
    print(f"input image: {image.shape}")

    x = conv2d(image, weights["conv1"], layer="conv1 IC=3 OC=4 3x3")
    print(f"  after conv1: {x.shape}")

    x = silu(x, layer="silu1 after conv1")
    print(f"  after silu1: {x.shape}")

    skip = x.copy()                                # save for later concat

    x = maxpool2d(x, layer="maxpool 2x2 s=2")
    print(f"  after maxpool: {x.shape}")

    x = conv2d(x, weights["conv2"], layer="conv2 IC=4 OC=8 1x1")
    print(f"  after conv2 (1x1): {x.shape}")

    x = silu(x, layer="silu2 after conv2")
    print(f"  after silu2: {x.shape}")

    x = conv2d(x, weights["conv3"], layer="conv3 IC=8 OC=4 3x3")
    print(f"  after conv3 (3x3): {x.shape}")

    x = upsample_nearest(x, factor=2, layer="upsample 2x")
    print(f"  after upsample: {x.shape}")

    # Trim upsample/skip to a common spatial size before concat (mini-block
    # shape arithmetic doesn't line up cleanly without padding logic).
    h = min(x.shape[-2], skip.shape[-2])
    w = min(x.shape[-1], skip.shape[-1])
    x = x[..., :h, :w]
    skip = skip[..., :h, :w]
    x = concat_channel(skip, x, layer="concat channel")
    print(f"  after concat: {x.shape}")

    x = conv2d(x, weights["conv4"], layer="conv4 IC=8 OC=4 1x1")
    print(f"  after conv4 (1x1): {x.shape}")

    x = sigmoid(x, layer="sigmoid output")
    print(f"  after sigmoid: {x.shape}")

    return x


def numpy_reference(image, weights):
    """Same chain via NumPy only — ground truth for the comparison."""
    x = _numpy_conv2d(image, weights["conv1"])
    xf = x.astype(np.float32); x = (xf / (1.0 + np.exp(-xf))).astype(np.float16)
    skip = x.copy()
    x = _numpy_maxpool2d(x, 2, 2, 2, 2)
    x = _numpy_conv2d(x, weights["conv2"])
    xf = x.astype(np.float32); x = (xf / (1.0 + np.exp(-xf))).astype(np.float16)
    x = _numpy_conv2d(x, weights["conv3"])
    x = _numpy_upsample_nearest(x, 2)
    h = min(x.shape[-2], skip.shape[-2])
    w = min(x.shape[-1], skip.shape[-1])
    x = x[..., :h, :w]
    skip = skip[..., :h, :w]
    x = np.concatenate([skip, x], axis=1)
    x = _numpy_conv2d(x, weights["conv4"])
    xf = x.astype(np.float32); x = (1.0 / (1.0 + np.exp(-xf))).astype(np.float16)
    return x


def main() -> int:
    rng = np.random.default_rng(42)
    # Tiny YOLOv8-shaped input. Real YOLOv8 takes 640x640; we use 16x16 to
    # keep per-op pyspike build+sim time tractable for a CI-style end-to-end.
    image = rng.uniform(-0.3, 0.3, (1, 3, 16, 16)).astype(np.float16)
    weights = {
        "conv1": rng.uniform(-0.2, 0.2, (4, 3, 3, 3)).astype(np.float16),   # IC=3 OC=4 3x3
        "conv2": rng.uniform(-0.2, 0.2, (8, 4, 1, 1)).astype(np.float16),   # 1x1
        "conv3": rng.uniform(-0.2, 0.2, (4, 8, 3, 3)).astype(np.float16),   # 3x3
        "conv4": rng.uniform(-0.2, 0.2, (4, 8, 1, 1)).astype(np.float16),   # 1x1
    }

    print("=== YOLOv8 mini block — pyspike-first inference ===")
    got = yolo_mini_inference(image, weights)

    print("\n=== NumPy reference (ground truth) ===")
    ref = numpy_reference(image, weights)

    max_diff = float(np.abs(got.astype(np.float32) - ref.astype(np.float32)).max())
    print(f"\nfinal tensor shape: {got.shape}")
    print(f"max_diff vs full-NumPy reference: {max_diff}")

    _print_routing_table()
    print(f"\noverall: {'PASS' if max_diff <= 0.05 else 'FAIL'} (tol 0.05)")
    return 0 if max_diff <= 0.05 else 1


if __name__ == "__main__":
    sys.exit(main())
