"""MM / MMC matrix-multiply handlers — context-resolved, broadcast-vectorized.

Encoding (RoCC custom0 0x0b). funct7 = family, funct3 = variant:

  family   funct7   variant  funct3  result
  ──────   ──────   ───────  ──────  ────────────────────────────────────
  MM       0x00     mm       0       A, B → C : matrix multiplication (A*B), result fp32 / int32
                    mmt      1       A, B → C : matrix multiplication (A*B), transposed write fp32 / int32
                    mmc      0       A, B, C → C : matrix multiplication and accumulation(A*B+C), result fp32 / int32
                    mmct     7       A, B, C → R : matrix multiplication and accumulation (A*B+C), transposed write fp32 / int32

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

from typing import TYPE_CHECKING

import numpy as np

from ...disasm import Custom0_Insn, inst_register
from ...exec_st import CXT
from ... import operand3
from ....config_params import L0_SIZE_BYTES, MX_IO_DTYPE, MX_IO_BYTES

if TYPE_CHECKING:
    from ....npu import GtxNpu   # noqa: F401


# =============================================================================
# rs1 dim decode
# =============================================================================
def decode_mm_args(rs1: int) -> dict:
    """Decode packed ``rs1`` → ``{'row_A', 'col_A', 'col_B'}`` (0 ⇒ 0x10000)."""
    def dim16(v: int) -> int:
        d = v & 0xFFFF
        return d if d != 0 else 0x10000
    return {
        'row_A': dim16(rs1),
        'col_A': dim16(rs1 >> 16),
        'col_B': dim16(rs1 >> 48),
    }


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
# Core GEMM (mm / mmt / mmc / mmct + accumulate variants)
# =============================================================================
@inst_register.custom0(name='mm', funct7=0b0000000, funct3=0)
def _mm(npu, proc, inst: Custom0_Insn, cxt: CXT, *,
             accumulate: bool, transposed: bool, fp32_out: bool) -> int:
    # inst  input       output  description
    # ----  -----------  -------  --------------------------------------
    # mm	A, B	    C	    matrix multiplication (A*B), result fp32 / int32
    
    ws = npu.warp
    args = decode_mm_args(int(proc.state.XPR[inst.rs1]))
    M, K, N = args['row_A'], args['col_A'], args['col_B']

    lspr = npu.lspr[ws.current_nest][ws.current_spu]
    a_hw = lspr.get('SPM_ADDRA', 0)  // MX_IO_BYTES # FP32 offset
    b_hw = lspr.get('SPM_ADDRB', 0) // MX_IO_BYTES # FP32 offset

    l1h = npu.mem.view(cxt, 'l1', ws, np.float32)   # (*batch, HW)
    batch = l1h.shape[:-1]
    A = l1h[..., a_hw:a_hw + M * K].reshape(*batch, M, K).astype(np.float32)
    # B is stored transposed in L1 as (N, K) row-major (vendor gemm_core /
    # SystemC SPU::MM B-load): B_l1[j, k] at offset j*K + k. The GEMM is
    # C = A[M×K] @ B_l1[N×K]^T → (M, N).
    B = l1h[..., b_hw:b_hw + N * K].reshape(*batch, N, K).astype(np.float32)
    # mm	A, B	C	matrix multiplication (A*B), result fp32 / int32
    c_w = lspr.get('SPM_ADDRC', 0) // MX_IO_BYTES
    
    C = A @ np.swapaxes(B, -1, -2) 
    out = C.transpose(-2, -1) if transposed else C
    r_hw = lspr.get('SPM_ADDRC', 0) // MX_IO_BYTES                  # mm / mmc (deprecated) → FP16
    l1h[..., r_hw:r_hw + M * N] = out.reshape(*batch, M * N).astype(MX_IO_DTYPE)
    return 0

@inst_register.custom0(name='mmt', funct7=0b0000000, funct3=7)
def _mmt(npu, proc, inst: Custom0_Insn, cxt: CXT) -> int:
    # inst  input       output  description
    # ----  -----------  -------  --------------------------------------
    # mmt	A, B	    C	    matrix multiplication (A*B), transposed write fp32 / int32
    ws = npu.warp
    args = decode_mm_args(int(proc.state.XPR[inst.rs1]))
    M, K, N = args['row_A'], args['col_A'], args['col_B']

    lspr = npu.lspr[ws.current_nest][ws.current_spu]
    a_hw = lspr.get('SPM_ADDRA', 0)  // MX_IO_BYTES # FP32 offset
    b_hw = lspr.get('SPM_ADDRB', 0) // MX_IO_BYTES # FP32 offset

    l1h = npu.mem.view(cxt, 'l1', ws, np.float32)   # (*batch, HW)
    batch = l1h.shape[:-1]
    A = l1h[..., a_hw:a_hw + M * K].reshape(*batch, M, K).astype(np.float32)
    # B is stored transposed in L1 as (N, K) row-major (vendor gemm_core /
    # SystemC SPU::MM B-load): B_l1[j, k] at offset j*K + k. The GEMM is
    # C = A[M×K] @ B_l1[N×K]^T → (M, N).
    B = l1h[..., b_hw:b_hw + N * K].reshape(*batch, N, K).astype(np.float32)
    # mm	A, B	C	matrix multiplication (A*B), result fp32 / int32
    c_w = lspr.get('SPM_ADDRC', 0) // MX_IO_BYTES
    l1f = npu.mem.view(cxt, 'l1', ws, np.float32)
    # mm
    C = A @ B.transpose(-2, -1)                       # FP32 GEMM
    l1f[..., c_w:c_w + M * N] = C.reshape(*batch, M * N).astype(np.float32)
    return 0

@inst_register.custom0(name='mmc', funct7=0b0000001, funct3=0)
def _mmc(npu, proc, inst: Custom0_Insn, cxt: CXT) -> int:
    # inst  input       output  description
    # ----  -----------  -------  --------------------------------------
    # mmc	A, B, C	    C	    matrix multiplication and accumulation(A*B+C), result fp32 / int32

    ws = npu.warp
    args = decode_mm_args(int(proc.state.XPR[inst.rs1]))
    M, K, N = args['row_A'], args['col_A'], args['col_B']

    lspr = npu.lspr[ws.current_nest][ws.current_spu]
    a_hw = lspr.get('SPM_ADDRA', 0)  // 4 # FP32 offset
    b_hw = lspr.get('SPM_ADDRB', 0) // 4 # FP32 offset

    l1h = npu.mem.view(cxt, 'l1', ws, np.float32)   # (*batch, HW)
    batch = l1h.shape[:-1]
    A = l1h[..., a_hw:a_hw + M * K].reshape(*batch, M, K).astype(np.float32)
    B = l1h[..., b_hw:b_hw + N * K].reshape(*batch, N, K).astype(np.float32)
    
    AB = A @ np.swapaxes(B, -1, -2)                       # FP32 GEMM
    c_w = lspr.get('SPM_ADDRC', 0) // 4              # FP32 word offset
    l1f = npu.mem.view(cxt, 'l1', ws, np.float32)
    C = l1f[..., c_w:c_w + M * N].reshape(*batch, M, N)
    l1f[..., c_w:c_w + M * N] = (AB + C).reshape(*batch, M * N).astype(np.float32)
    return 0

@inst_register.custom0(name='mmct', funct7=0b0000001, funct3=7)
def _mmct(npu, proc, inst: Custom0_Insn, cxt: CXT) -> int:
        # inst  input       output  description
    # ----  -----------  -------  --------------------------------------
    # mmc	A, B, C	    C	    matrix multiplication and accumulation(A*B+C), result fp32 / int32

    ws = npu.warp
    args = decode_mm_args(int(proc.state.XPR[inst.rs1]))
    M, K, N = args['row_A'], args['col_A'], args['col_B']

    lspr = npu.lspr[ws.current_nest][ws.current_spu]
    a_hw = lspr.get('SPM_ADDRA', 0)  // MX_IO_BYTES # FP32 offset
    b_hw = lspr.get('SPM_ADDRB', 0) // MX_IO_BYTES # FP32 offset

    l1h = npu.mem.view(cxt, 'l1', ws, np.float32)   # (*batch, HW)
    batch = l1h.shape[:-1]
    A = l1h[..., a_hw:a_hw + M * K].reshape(*batch, M, K).astype(np.float32)
    B = l1h[..., b_hw:b_hw + N * K].reshape(*batch, N, K).astype(np.float32)
    
    AB = A @ np.swapaxes(B, -1, -2)                       # FP32 GEMM
    c_w = lspr.get('SPM_ADDRC', 0) // MX_IO_BYTES              # FP32 word offset
    l1f = npu.mem.view(cxt, 'l1', ws, np.float32)
    C = l1f[..., c_w:c_w + M * N].reshape(*batch, M, N)
    C = (AB + C).reshape(*batch, M * N).astype(np.float32)
    
    # Transposed write(mmct)
    C_T = C.transpose(-2, -1)
    
    r_w = lspr.get('SPM_ADDRR', 0) // MX_IO_BYTES
    l1io = npu.mem.view(cxt, 'l1', ws, MX_IO_DTYPE)
    l1io[..., r_w:r_w + M * N] = C_T.reshape(*batch, M * N).astype(MX_IO_DTYPE)
    return 0


# # =============================================================================
# # Reduction (mm.o sum / mm.v dot + accumulate variants) → L0 scalar + mxe_accum
# # =============================================================================
# def _mm_reduce(npu, proc, inst: Custom0_Insn, cxt: CXT, *,
#                accumulate: bool, is_dot: bool) -> int:
#     ws = npu.warp
#     args = decode_mm_args(int(proc.state.XPR[inst.rs1]))
#     vec = args['col_A']

#     lspr = npu.lspr[ws.current_nest][ws.current_spu]
#     a_hw = lspr.get('SPM_ADDRA', 0) // 2

#     l1h = npu.mem.view(cxt, 'l1', ws, np.float16)     # (*batch, HW)
#     A = l1h[..., a_hw:a_hw + vec].astype(np.float32)
#     if is_dot:
#         b_hw = lspr.get('SPM_ADDRB', 0) // 2
#         B = l1h[..., b_hw:b_hw + vec].astype(np.float32)
#         red = (A * B).sum(axis=-1)                       # dot, FP32
#     else:
#         red = A.sum(axis=-1)                             # sum, FP32

#     idx = _state_scope(cxt, ws)
#     if accumulate:
#         red = red + npu._mxe_accum[idx].astype(np.float32)
#     npu._mxe_accum[idx] = red.astype(npu._mxe_accum.dtype)   # always update (Pitfall B)

#     import os as _os, sys as _sys
#     if _os.environ.get("GTX_DEBUG_NORM"):
#         _slot = operand3(npu) & 0x1F
#         print(f"[MMRED] {'dot' if is_dot else 'sum'} cxt={cxt} vec={vec} "
#               f"a_hw={a_hw} A0={float(A.reshape(-1)[0]):.4f} red={float(np.asarray(red).reshape(-1)[0]):.4f} "
#               f"-> SVR{_slot} (op3={operand3(npu):#x})", file=_sys.stderr, flush=True)

#     # Result → L0 slot in the MX I/O width (FP32 LE by default), rest of the
#     # 32 B reg zeroed. Divergence from vendor (which stored an FP16/INT32
#     # scalar): the FP16 output cast was the dominant precision-loss source.
#     # Gated by config_params.MX_IO_DTYPE — flip to FP16 for vendor parity.
#     slot = operand3(npu) & 0x1F                           # rs3 result SVR addr[4:0]
#     off = (slot * 32) % L0_SIZE_BYTES
#     l0b = npu.mem.view(cxt, 'l0', ws, np.uint8)       # (*batch, L0_BYTES)
#     l0b[..., off:off + 32] = 0                            # zero the 32 B reg
#     # Write the scalar straight through the MX_IO_DTYPE view — it aliases the
#     # same bytes as the uint8 view, so one batched element assignment replaces
#     # the per-byte Python loop (and stays correct across every CONTEXT scope).
#     l0io = npu.mem.view(cxt, 'l0', ws, MX_IO_DTYPE)
#     l0io[..., off // MX_IO_BYTES] = red.astype(MX_IO_DTYPE)
#     return 0


