"""MM / MMC matrix-multiply handlers — context-resolved, broadcast-vectorized.

Encoding (RoCC custom0 0x0b). funct7 = family, funct3 = variant:

  family   funct7   variant  funct3  result
  ──────   ──────   ───────  ──────  ────────────────────────────────────
  MM       0x00     mm.s     0       A@B  [+bias] → ADDRC, FP32
                    mm.o     1       sum(A)       → L0, FP32 + mxe_accum
                    mm       2       A@B  [+bias] → ADDRR, FP16   ** DEPRECATED (soon) **
                    mm.v     3       dot(A,B)     → L0, FP32 + mxe_accum
                    mm.t     7       (A@B)^T      → ADDRR, FP32 (N×M)

  ``.o`` / ``.v`` / ``.t`` results use the MX I/O width (FP32 by default) — a
  deliberate divergence from vendor (which used FP16/INT32) to remove the
  FP16-cast precision loss. Gated by config_params.MX_IO_DTYPE (flip to FP16
  for vendor parity). ``.s`` is FP32→ADDRC in vendor too, so it is not gated.
  MMC      0x01     mmc.{s,o,..}     same, but accumulate (+ ADDRC bias / mxe prior)
                    mmc      2       (the accumulate twin of ``mm``) ** DEPRECATED (soon) **

NOTE: the plain ``mm`` / ``mmc`` GEMM variants (funct3=2) are slated for
removal soon — prefer ``mm.s`` (FP32→ADDRC) / ``mm.t`` for new firmware.

rs1 packs the dims: ``colB[63:48] | colA[31:16] | rowA[15:0]`` (0 ⇒ 0x10000).

Context model (the simplification): each handler asks the memory layer for a
CONTEXT-SCOPED view via ``npu.mem.view(cxt, level, ws, dtype)``. CONTEXT decides
the (NEST, SPU) scope and the leading tensor dims ARE the broadcast batch, so
ONE batched op covers every mode with no per-context branching and no Python
loop over NEST/SPU:

  C3 (Mode 4) → view is ``(…)``          → single SPU (hot path)
  C4 (Mode 2) → view is ``(SPU, …)``     → batched over the NEST's SPUs
  C1 (Mode 1) → view is ``(NEST, SPU, …)`` → batched over everything

Assumptions (kept for simplicity — revisit if a regression needs them):
  - the matrix region does not wrap the L1 ring (tiles ≪ 384 KB);
  - SPM_ADDR* are uniform across a broadcast scope (P-loop WRSPR broadcast
    contract), so addresses are read from the representative (cur_nest, cur_spu).

Depends on ``npu.warp`` exposing ``current_nest`` / ``current_spu`` and
``npu.mem.view`` (added in memory.py). RoCC return is 0 — results land in L1/L0.
"""
from __future__ import annotations

from functools import lru_cache
from typing import TYPE_CHECKING

import numpy as np

from ...disasm import Custom0_Insn, inst_register
from ...exec_st import CXT
from ... import operand3
from ....config_params import L0_SIZE_BYTES, MX_IO_DTYPE, MX_IO_BYTES

if TYPE_CHECKING:
    from ....npu import GtxNpu   # noqa: F401


# =============================================================================
# rs1 dim decode (hot path — memoised; rs1 value space is small per graph)
# =============================================================================
@lru_cache(maxsize=256)
def decode_mm_args(rs1: int) -> tuple[int, int, int]:
    """Decode packed ``rs1`` → ``(row_A, col_A, col_B)`` (0 ⇒ 0x10000).

    Returns a tuple instead of a dict so the result is hashable / immutable and
    the lru_cache entry doesn't outlive cheap dict allocation. Callers unpack
    via ``M, K, N = decode_mm_args(rs1)``.
    """
    def dim16(v: int) -> int:
        d = v & 0xFFFF
        return d if d != 0 else 0x10000
    return dim16(rs1), dim16(rs1 >> 16), dim16(rs1 >> 48)


def _state_scope(cxt: CXT, ws) -> tuple:
    """Index into a ``(NEST, SPU)``-shaped state tensor (e.g. ``mxe_accum``)
    matching the data scope of *cxt* — so the reduction result shape lines up.
    """
    if cxt is CXT.C3:
        return (ws.current_nest, ws.current_spu)
    if cxt is CXT.C2 or cxt is CXT.C4:
        return (ws.current_nest,)
    return ()


# =============================================================================
# Core GEMM (mm / mm.s / mm.t + accumulate variants)
# =============================================================================
def _mm_gemm(npu, proc, inst: Custom0_Insn, cxt: CXT, *,
             accumulate: bool, transposed: bool, fp32_out: bool) -> int:
    ws = npu.warp
    M, K, N = decode_mm_args(int(proc.state.XPR[inst.rs1]))

    lspr = npu.lspr[ws.current_nest][ws.current_spu]
    a_hw = lspr.get('SPM_ADDRA', 0) // 2     # FP16 halfword offsets
    b_hw = lspr.get('SPM_ADDRB', 0) // 2

    l1h = npu.mem.view(cxt, 'l1', ws, np.float16)   # (*batch, HW)
    batch = l1h.shape[:-1]
    A = l1h[..., a_hw:a_hw + M * K].reshape(*batch, M, K).astype(np.float32)
    # B is stored transposed in L1 as (N, K) row-major (vendor gemm_core /
    # SystemC SPU::MM B-load): B_l1[j, k] at offset j*K + k. The GEMM is
    # C = A[M×K] @ B_l1[N×K]^T → (M, N).
    B = l1h[..., b_hw:b_hw + N * K].reshape(*batch, N, K).astype(np.float32)
    C = A @ np.swapaxes(B, -1, -2)                       # FP32 accumulate

    if accumulate:
        c_w = lspr.get('SPM_ADDRC', 0) // 4              # FP32 word offset
        l1f = npu.mem.view(cxt, 'l1', ws, np.float32)
        C = C + l1f[..., c_w:c_w + M * N].reshape(*batch, M, N)

    if fp32_out:                                         # mm.s / mmc.s → ADDRC, FP32
        c_w = lspr.get('SPM_ADDRC', 0) // 4
        l1f = npu.mem.view(cxt, 'l1', ws, np.float32)
        l1f[..., c_w:c_w + M * N] = C.reshape(*batch, M * N)
        return 0

    out = C.transpose(-2, -1) if transposed else C       # mm.t → (N, M)
    if transposed:                                       # mm.t / mmc.t → ADDRR
        # Divergence from vendor (FP16): keep the transposed GEMM result in the
        # MX I/O width (FP32 by default) to avoid the FP16-cast precision loss.
        # Gated by config_params.MX_IO_DTYPE — flip to FP16 for vendor parity.
        r_w = lspr.get('SPM_ADDRR', 0) // MX_IO_BYTES
        l1io = npu.mem.view(cxt, 'l1', ws, MX_IO_DTYPE)
        l1io[..., r_w:r_w + M * N] = out.reshape(*batch, M * N).astype(MX_IO_DTYPE)
        return 0

    r_hw = lspr.get('SPM_ADDRR', 0) // 2                  # mm / mmc (deprecated) → FP16
    l1h[..., r_hw:r_hw + M * N] = out.reshape(*batch, M * N).astype(np.float16)
    return 0


# =============================================================================
# Reduction (mm.o sum / mm.v dot + accumulate variants) → L0 scalar + mxe_accum
# =============================================================================
def _mm_reduce(npu, proc, inst: Custom0_Insn, cxt: CXT, *,
               accumulate: bool, is_dot: bool) -> int:
    ws = npu.warp
    _row, vec, _col_b = decode_mm_args(int(proc.state.XPR[inst.rs1]))

    lspr = npu.lspr[ws.current_nest][ws.current_spu]
    a_hw = lspr.get('SPM_ADDRA', 0) // 2

    l1h = npu.mem.view(cxt, 'l1', ws, np.float16)     # (*batch, HW)
    A = l1h[..., a_hw:a_hw + vec].astype(np.float32)
    if is_dot:
        b_hw = lspr.get('SPM_ADDRB', 0) // 2
        B = l1h[..., b_hw:b_hw + vec].astype(np.float32)
        red = (A * B).sum(axis=-1)                       # dot, FP32
    else:
        red = A.sum(axis=-1)                             # sum, FP32

    idx = _state_scope(cxt, ws)
    if accumulate:
        red = red + npu._mxe_accum[idx].astype(np.float32)
    npu._mxe_accum[idx] = red.astype(npu._mxe_accum.dtype)   # always update (Pitfall B)

    import os as _os, sys as _sys
    if _os.environ.get("GTX_DEBUG_NORM"):
        _slot = operand3(npu) & 0x1F
        print(f"[MMRED] {'dot' if is_dot else 'sum'} cxt={cxt} vec={vec} "
              f"a_hw={a_hw} A0={float(A.reshape(-1)[0]):.4f} red={float(np.asarray(red).reshape(-1)[0]):.4f} "
              f"-> SVR{_slot} (op3={operand3(npu):#x})", file=_sys.stderr, flush=True)

    # Result → L0 slot in the MX I/O width (FP32 LE by default), rest of the
    # 32 B reg zeroed. Divergence from vendor (which stored an FP16/INT32
    # scalar): the FP16 output cast was the dominant precision-loss source.
    # Gated by config_params.MX_IO_DTYPE — flip to FP16 for vendor parity.
    slot = operand3(npu) & 0x1F                           # rs3 result SVR addr[4:0]
    off = (slot * 32) % L0_SIZE_BYTES
    l0b = npu.mem.view(cxt, 'l0', ws, np.uint8)       # (*batch, L0_BYTES)
    l0b[..., off:off + 32] = 0                            # zero the 32 B reg
    # Write the scalar straight through the MX_IO_DTYPE view — it aliases the
    # same bytes as the uint8 view, so one batched element assignment replaces
    # the per-byte Python loop (and stays correct across every CONTEXT scope).
    l0io = npu.mem.view(cxt, 'l0', ws, MX_IO_DTYPE)
    l0io[..., off // MX_IO_BYTES] = red.astype(MX_IO_DTYPE)
    return 0


# =============================================================================
# Handler entries — Pitfall F: rs1 index 0 ⇒ gem5 WRSPR/RDSPR, not MM ⇒ NOP.
# =============================================================================
@inst_register.custom0(name='mm.s', funct7=0b0000000, funct3=0)
def _mm_s(npu, proc, inst: Custom0_Insn, cxt: CXT) -> int:
    if inst.rs1 == 0:
        return 0
    return _mm_gemm(npu, proc, inst, cxt, accumulate=False, transposed=False, fp32_out=True)


@inst_register.custom0(name='mm.o', funct7=0b0000000, funct3=1)
def _mm_o(npu, proc, inst: Custom0_Insn, cxt: CXT) -> int:
    if inst.rs1 == 0:
        return 0
    return _mm_reduce(npu, proc, inst, cxt, accumulate=False, is_dot=False)


@inst_register.custom0(name='mm', funct7=0b0000000, funct3=2)
def _mm(npu, proc, inst: Custom0_Insn, cxt: CXT) -> int:
    """DEPRECATED (soon): plain FP16 GEMM → ADDRR. Prefer mm.s / mm.t."""
    if inst.rs1 == 0:
        return 0
    return _mm_gemm(npu, proc, inst, cxt, accumulate=False, transposed=False, fp32_out=False)


@inst_register.custom0(name='mm.v', funct7=0b0000000, funct3=3)
def _mm_v(npu, proc, inst: Custom0_Insn, cxt: CXT) -> int:
    if inst.rs1 == 0:
        return 0
    return _mm_reduce(npu, proc, inst, cxt, accumulate=False, is_dot=True)


@inst_register.custom0(name='mm.t', funct7=0b0000000, funct3=7)
def _mm_t(npu, proc, inst: Custom0_Insn, cxt: CXT) -> int:
    if inst.rs1 == 0:
        return 0
    return _mm_gemm(npu, proc, inst, cxt, accumulate=False, transposed=True, fp32_out=False)


@inst_register.custom0(name='mmc.s', funct7=0b0000001, funct3=0)
def _mmc_s(npu, proc, inst: Custom0_Insn, cxt: CXT) -> int:
    if inst.rs1 == 0:
        return 0
    return _mm_gemm(npu, proc, inst, cxt, accumulate=True, transposed=False, fp32_out=True)


@inst_register.custom0(name='mmc.o', funct7=0b0000001, funct3=1)
def _mmc_o(npu, proc, inst: Custom0_Insn, cxt: CXT) -> int:
    if inst.rs1 == 0:
        return 0
    return _mm_reduce(npu, proc, inst, cxt, accumulate=True, is_dot=False)


@inst_register.custom0(name='mmc', funct7=0b0000001, funct3=2)
def _mmc(npu, proc, inst: Custom0_Insn, cxt: CXT) -> int:
    """DEPRECATED (soon): accumulate twin of ``mm``. Prefer mmc.s / mmc.t."""
    if inst.rs1 == 0:
        return 0
    return _mm_gemm(npu, proc, inst, cxt, accumulate=True, transposed=False, fp32_out=False)


@inst_register.custom0(name='mmc.v', funct7=0b0000001, funct3=3)
def _mmc_v(npu, proc, inst: Custom0_Insn, cxt: CXT) -> int:
    if inst.rs1 == 0:
        return 0
    return _mm_reduce(npu, proc, inst, cxt, accumulate=True, is_dot=True)


@inst_register.custom0(name='mmc.t', funct7=0b0000001, funct3=7)
def _mmc_t(npu, proc, inst: Custom0_Insn, cxt: CXT) -> int:
    if inst.rs1 == 0:
        return 0
    return _mm_gemm(npu, proc, inst, cxt, accumulate=True, transposed=True, fp32_out=False)
