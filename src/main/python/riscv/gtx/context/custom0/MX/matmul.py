"""MM / MMC matrix-multiply handlers — context-resolved, broadcast-vectorized.

Encoding (RoCC custom0 0x0b). funct7 = family, funct3 = variant:

  family   funct7   variant  funct3  result
  ──────   ──────   ───────  ──────  ────────────────────────────────────
  MM       0x00     mm.s     0       A@B  [+bias] → ADDRC, FP32
                    mm.o     1       sum(A)       → L0 big-endian + mxe_accum
                    mm       2       A@B  [+bias] → ADDRR, FP16
                    mm.v     3       dot(A,B)     → L0 little-endian + mxe_accum
                    mm.t     7       (A@B)^T      → ADDRR, FP16 (N×M)
  MMC      0x01     mmc.{s,o,..}     same, but accumulate (+ ADDRC bias / mxe prior)

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

import torch

from ...disasm import Custom0_Insn, inst_register
from ...exec_st import CXT
from ....config_params import L0_SIZE_BYTES

if TYPE_CHECKING:
    from ....npu import GtxNpu   # noqa: F401


# =============================================================================
# Encoding constants
# =============================================================================
F7_MM: int = 0x00      # MM family  (collides with gem5 WRSPR — Pitfall F)
F7_MMC: int = 0x01     # MMC family (collides with gem5 RDSPR — Pitfall F)

F3_MM_S: int = 0           # FP32 result → ADDRC
F3_MM_O: int = 1           # sum(A) → L0 big-endian + mxe_accum
F3_MM: int = 2             # basic GEMM, FP16 → ADDRR
F3_MM_V: int = 3           # dot(A,B) → L0 little-endian + mxe_accum
F3_MM_T: int = 7           # transposed C^T → ADDRR


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
# Core GEMM (mm / mm.s / mm.t + accumulate variants)
# =============================================================================
def _mm_gemm(npu, proc, inst: Custom0_Insn, cxt: CXT, *,
             accumulate: bool, transposed: bool, fp32_out: bool) -> int:
    ws = npu.warp
    args = decode_mm_args(int(proc.state.XPR[inst.rs1]))
    M, K, N = args['row_A'], args['col_A'], args['col_B']

    lspr = npu.lspr[ws.current_nest][ws.current_spu]
    a_hw = lspr.get('SPM_ADDRA', 0) // 2     # FP16 halfword offsets
    b_hw = lspr.get('SPM_ADDRB', 0) // 2

    l1h = npu.mem.view(cxt, 'l1', ws, torch.float16)   # (*batch, HW)
    batch = l1h.shape[:-1]
    A = l1h[..., a_hw:a_hw + M * K].reshape(*batch, M, K).to(torch.float32)
    B = l1h[..., b_hw:b_hw + K * N].reshape(*batch, K, N).to(torch.float32)
    C = A @ B                                            # FP32 accumulate

    if accumulate:
        c_w = lspr.get('SPM_ADDRC', 0) // 4              # FP32 word offset
        l1f = npu.mem.view(cxt, 'l1', ws, torch.float32)
        C = C + l1f[..., c_w:c_w + M * N].reshape(*batch, M, N)

    if fp32_out:                                         # mm.s / mmc.s → ADDRC
        c_w = lspr.get('SPM_ADDRC', 0) // 4
        l1f = npu.mem.view(cxt, 'l1', ws, torch.float32)
        l1f[..., c_w:c_w + M * N] = C.reshape(*batch, M * N)
        return 0

    out = C.transpose(-2, -1) if transposed else C       # mm.t → (N, M)
    r_hw = lspr.get('SPM_ADDRR', 0) // 2
    l1h[..., r_hw:r_hw + M * N] = out.reshape(*batch, M * N).to(torch.float16)
    return 0


# =============================================================================
# Reduction (mm.o sum / mm.v dot + accumulate variants) → L0 scalar + mxe_accum
# =============================================================================
def _mm_reduce(npu, proc, inst: Custom0_Insn, cxt: CXT, *,
               accumulate: bool, is_dot: bool) -> int:
    ws = npu.warp
    args = decode_mm_args(int(proc.state.XPR[inst.rs1]))
    vec = args['col_A']

    lspr = npu.lspr[ws.current_nest][ws.current_spu]
    a_hw = lspr.get('SPM_ADDRA', 0) // 2

    l1h = npu.mem.view(cxt, 'l1', ws, torch.float16)     # (*batch, HW)
    A = l1h[..., a_hw:a_hw + vec].to(torch.float32)
    if is_dot:
        b_hw = lspr.get('SPM_ADDRB', 0) // 2
        B = l1h[..., b_hw:b_hw + vec].to(torch.float32)
        red = (A * B).sum(dim=-1)                        # dot, FP32
    else:
        red = A.sum(dim=-1)                              # sum, FP32

    idx = _state_scope(cxt, ws)
    if accumulate:
        red = red + npu._mxe_accum[idx].to(torch.float32)
    npu._mxe_accum[idx] = red.to(npu._mxe_accum.dtype)   # always update (Pitfall B)

    # FP16 scalar → L0 slot; o = big-endian, v = little-endian (asymmetric).
    slot = int(npu.gspr.get('GSPR_GTX_OPERAND3', 0)) & 0x1F
    off = (slot * 32) % L0_SIZE_BYTES
    l0b = npu.mem.view(cxt, 'l0', ws, torch.uint8)       # (*batch, L0_BYTES)
    l0b[..., off:off + 32] = 0
    bits = red.to(torch.float16).contiguous().view(torch.int16).to(torch.int64) & 0xFFFF
    hi = ((bits >> 8) & 0xFF).to(torch.uint8)
    lo = (bits & 0xFF).to(torch.uint8)
    if is_dot:                                           # little-endian (v)
        l0b[..., off] = lo
        l0b[..., off + 1] = hi
    else:                                                # big-endian (o)
        l0b[..., off] = hi
        l0b[..., off + 1] = lo
    return 0


# =============================================================================
# Handler entries — Pitfall F: rs1 index 0 ⇒ gem5 WRSPR/RDSPR, not MM ⇒ NOP.
# =============================================================================
@inst_register.custom0(name='mm.s', funct7=F7_MM, funct3=F3_MM_S)
def _mm_s(npu, proc, inst: Custom0_Insn, cxt: CXT) -> int:
    if inst.rs1 == 0:
        return 0
    return _mm_gemm(npu, proc, inst, cxt, accumulate=False, transposed=False, fp32_out=True)


@inst_register.custom0(name='mm.o', funct7=F7_MM, funct3=F3_MM_O)
def _mm_o(npu, proc, inst: Custom0_Insn, cxt: CXT) -> int:
    if inst.rs1 == 0:
        return 0
    return _mm_reduce(npu, proc, inst, cxt, accumulate=False, is_dot=False)


@inst_register.custom0(name='mm', funct7=F7_MM, funct3=F3_MM)
def _mm(npu, proc, inst: Custom0_Insn, cxt: CXT) -> int:
    if inst.rs1 == 0:
        return 0
    return _mm_gemm(npu, proc, inst, cxt, accumulate=False, transposed=False, fp32_out=False)


@inst_register.custom0(name='mm.v', funct7=F7_MM, funct3=F3_MM_V)
def _mm_v(npu, proc, inst: Custom0_Insn, cxt: CXT) -> int:
    if inst.rs1 == 0:
        return 0
    return _mm_reduce(npu, proc, inst, cxt, accumulate=False, is_dot=True)


@inst_register.custom0(name='mm.t', funct7=F7_MM, funct3=F3_MM_T)
def _mm_t(npu, proc, inst: Custom0_Insn, cxt: CXT) -> int:
    if inst.rs1 == 0:
        return 0
    return _mm_gemm(npu, proc, inst, cxt, accumulate=False, transposed=True, fp32_out=False)


@inst_register.custom0(name='mmc.s', funct7=F7_MMC, funct3=F3_MM_S)
def _mmc_s(npu, proc, inst: Custom0_Insn, cxt: CXT) -> int:
    if inst.rs1 == 0:
        return 0
    return _mm_gemm(npu, proc, inst, cxt, accumulate=True, transposed=False, fp32_out=True)


@inst_register.custom0(name='mmc.o', funct7=F7_MMC, funct3=F3_MM_O)
def _mmc_o(npu, proc, inst: Custom0_Insn, cxt: CXT) -> int:
    if inst.rs1 == 0:
        return 0
    return _mm_reduce(npu, proc, inst, cxt, accumulate=True, is_dot=False)


@inst_register.custom0(name='mmc', funct7=F7_MMC, funct3=F3_MM)
def _mmc(npu, proc, inst: Custom0_Insn, cxt: CXT) -> int:
    if inst.rs1 == 0:
        return 0
    return _mm_gemm(npu, proc, inst, cxt, accumulate=True, transposed=False, fp32_out=False)


@inst_register.custom0(name='mmc.v', funct7=F7_MMC, funct3=F3_MM_V)
def _mmc_v(npu, proc, inst: Custom0_Insn, cxt: CXT) -> int:
    if inst.rs1 == 0:
        return 0
    return _mm_reduce(npu, proc, inst, cxt, accumulate=True, is_dot=True)


@inst_register.custom0(name='mmc.t', funct7=F7_MMC, funct3=F3_MM_T)
def _mmc_t(npu, proc, inst: Custom0_Insn, cxt: CXT) -> int:
    if inst.rs1 == 0:
        return 0
    return _mm_gemm(npu, proc, inst, cxt, accumulate=True, transposed=True, fp32_out=False)
