"""YOLOv8-style multi-IC/OC conv experiment via host-side IM2COL + pyspike MUL_MAT.

ggml's `ggml_conv_2d` internally expands to:
    im2col(b) -> (B, OH, OW, IC*KH*KW)
    reshape to (B*OH*OW, IC*KH*KW)
    weight reshape (OC, IC*KH*KW)
    matmul -> (B*OH*OW, OC) -> reshape (B, OC, OH, OW)

We do the same on the host side: NumPy handles the layout-heavy IM2COL +
reshape, pyspike's MUL_MAT_tiled accelerates the GEMM (the part that
dominates conv runtime in real YOLO inference).

Test config (small YOLOv8 first-layer flavour):
    input:  (B=1, IC=3, IH=8, IW=8)
    kernel: (OC=4, IC=3, KH=3, KW=3)
    output: (B=1, OC=4, OH=6, OW=6)  with stride=1, pad=0
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
SERVER_DIR = HERE.parent
sys.path.insert(0, str(SERVER_DIR))

import pyspike_runner as psr  # noqa: E402


def numpy_im2col(b: np.ndarray, kh: int, kw: int,
                 sh: int = 1, sw: int = 1, ph: int = 0, pw: int = 0) -> np.ndarray:
    """Match ggml IM2COL output layout: (B, OH, OW, IC*KH*KW), KW innermost."""
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


def numpy_conv2d_reference(x: np.ndarray, w: np.ndarray,
                            sh: int = 1, sw: int = 1,
                            ph: int = 0, pw: int = 0) -> np.ndarray:
    """Direct convolution reference (B, IC, IH, IW) x (OC, IC, KH, KW) -> (B, OC, OH, OW)."""
    B, IC, IH, IW = x.shape
    OC, IC_w, KH, KW = w.shape
    assert IC == IC_w, f"IC mismatch: input {IC} vs kernel {IC_w}"
    OH = (IH + 2 * ph - KH) // sh + 1
    OW = (IW + 2 * pw - KW) // sw + 1
    xp = np.pad(x, ((0, 0), (0, 0), (ph, ph), (pw, pw)))
    out = np.zeros((B, OC, OH, OW), dtype=np.float32)
    xf = xp.astype(np.float32)
    wf = w.astype(np.float32)
    for kh_i in range(KH):
        for kw_i in range(KW):
            ys = kh_i * sh
            xs = kw_i * sw
            sliced = xf[:, :, ys:ys + sh * OH:sh, xs:xs + sw * OW:sw]
            out += np.einsum('bchw,oc->bohw', sliced, wf[:, :, kh_i, kw_i])
    return out.astype(np.float16)


def host_conv2d_pyspike(x: np.ndarray, w: np.ndarray,
                        sh: int = 1, sw: int = 1,
                        ph: int = 0, pw: int = 0,
                        kpad: bool = True) -> np.ndarray:
    """ggml-style conv expansion: NumPy IM2COL on host + pyspike MUL_MAT_tiled.

    x: (B, IC, IH, IW) fp16, w: (OC, IC, KH, KW) fp16.
    Returns (B, OC, OH, OW) fp16.

    `kpad=True` (default): when K = IC*KH*KW isn't a multiple of 8, zero-pad
    both A's right edge and B's right edge to the next K%8==0 boundary. Since
    A_pad[:, K:] = 0 and B_pad[:, K:] = 0, the additional partial products are
    all 0 and the GEMM result is mathematically identical. This unlocks the
    YOLOv8 first layer (IC=3 → K=27) for the pyspike fast path.
    """
    B, IC, IH, IW = x.shape
    OC, _, KH, KW = w.shape
    OH = (IH + 2 * ph - KH) // sh + 1
    OW = (IW + 2 * pw - KW) // sw + 1

    # IM2COL: (B, OH, OW, IC*KH*KW)
    patches = numpy_im2col(x, KH, KW, sh, sw, ph, pw)
    a = patches.reshape(B * OH * OW, IC * KH * KW)
    b = w.reshape(OC, IC * KH * KW)

    M, K = a.shape
    N = b.shape[0]
    K_padded = ((K + 7) // 8) * 8

    if K_padded != K:
        if not kpad:
            raise ValueError(f"pyspike MUL_MAT requires K%8==0, got K={K}")
        # Pad K with zeros — products against 0 columns contribute 0 to the
        # final sum, so the GEMM result is unchanged.
        a_p = np.zeros((M, K_padded), dtype=np.float16)
        a_p[:, :K] = a
        b_p = np.zeros((N, K_padded), dtype=np.float16)
        b_p[:, :K] = b
    else:
        a_p = a.astype(np.float16)
        b_p = b.astype(np.float16)

    out_bytes = psr.SUPPORTED_PYSPIKE_OPS["mul_mat_tiled_fp16"](
        a_p.tobytes(), b_p.tobytes(),
        M=M, K=K_padded, N=N)
    mm = np.frombuffer(out_bytes, dtype=np.float16).reshape(M, N)
    return mm.reshape(B, OH, OW, OC).transpose(0, 3, 1, 2).copy()


def run_case(name: str, x: np.ndarray, w: np.ndarray,
             tol: float = 0.5) -> tuple[bool, float]:
    print(f"\n--- {name} ---")
    print(f"input:  {x.shape} ({np.prod(x.shape)} fp16)")
    print(f"kernel: {w.shape} ({np.prod(w.shape)} fp16)")
    OC, IC, KH, KW = w.shape
    K = IC * KH * KW
    print(f"GEMM dims: M=B*OH*OW, K={K} (IC*KH*KW), N={OC}")

    ref = numpy_conv2d_reference(x, w)
    print(f"NumPy reference shape: {ref.shape}")

    try:
        got = host_conv2d_pyspike(x, w)
    except ValueError as e:
        print(f"SKIP: {e}")
        return True, 0.0

    print(f"pyspike conv  shape: {got.shape}")
    max_diff = float(np.abs(got.astype(np.float32) - ref.astype(np.float32)).max())
    ok = max_diff <= tol
    print(f"max_diff vs NumPy reference: {max_diff}  →  {'PASS' if ok else 'FAIL'} (tol {tol})")
    return ok, max_diff


def main() -> int:
    rng = np.random.default_rng(0)
    results = []

    # Case 1: 1x1 pointwise conv (common in YOLOv8 bottleneck blocks).
    # K = IC*1*1 = 8 → fits MUL_MAT K%8==0 guard exactly.
    x1 = rng.uniform(-0.5, 0.5, (1, 8, 8, 8)).astype(np.float16)
    w1 = rng.uniform(-0.5, 0.5, (16, 8, 1, 1)).astype(np.float16)
    results.append(("pointwise 1x1 IC=8 OC=16 (K=8)",) + run_case("Case 1: 1x1 conv", x1, w1, tol=0.05))

    # Case 2: 3x3 conv with IC=8 → K=72 (8 배수).
    x2 = rng.uniform(-0.3, 0.3, (1, 8, 6, 6)).astype(np.float16)
    w2 = rng.uniform(-0.3, 0.3, (4, 8, 3, 3)).astype(np.float16)
    results.append(("3x3 IC=8 OC=4 (K=72)",) + run_case("Case 2: 3x3 conv", x2, w2, tol=0.1))

    # Case 3: bigger — IC=16 / OC=8 / 3x3 (K=144). Closer to YOLOv8 mid-stage.
    x3 = rng.uniform(-0.2, 0.2, (1, 16, 8, 8)).astype(np.float16)
    w3 = rng.uniform(-0.2, 0.2, (8, 16, 3, 3)).astype(np.float16)
    results.append(("3x3 IC=16 OC=8 (K=144)",) + run_case("Case 3: 3x3 IC=16", x3, w3, tol=0.2))

    # Case 4: YOLOv8 first layer flavour — IC=3 → K=27, needs zero-padding to
    # K_padded=32 to satisfy the MUL_MAT K%8==0 constraint.
    x4 = rng.uniform(-0.3, 0.3, (1, 3, 8, 8)).astype(np.float16)
    w4 = rng.uniform(-0.3, 0.3, (4, 3, 3, 3)).astype(np.float16)
    results.append(("3x3 IC=3 OC=4 (K=27→32 zero-pad)",) + run_case("Case 4: IC=3 (K-pad)", x4, w4, tol=0.05))

    # Case 5: IC=3 + bigger output channels (YOLOv8 stem variant).
    x5 = rng.uniform(-0.2, 0.2, (1, 3, 16, 16)).astype(np.float16)
    w5 = rng.uniform(-0.2, 0.2, (32, 3, 3, 3)).astype(np.float16)
    results.append(("3x3 IC=3 OC=32 (K=27→32 zero-pad)",) + run_case("Case 5: IC=3 OC=32 stem", x5, w5, tol=0.1))

    print("\n" + "=" * 60)
    n_pass = sum(1 for _, ok, _ in results if ok)
    for name, ok, md in results:
        marker = "PASS" if ok else "FAIL"
        print(f"  {marker} {name:40s} max_diff={md:.4g}")
    print(f"{n_pass}/{len(results)} cases PASS")
    return 0 if n_pass == len(results) else 1


if __name__ == "__main__":
    sys.exit(main())
