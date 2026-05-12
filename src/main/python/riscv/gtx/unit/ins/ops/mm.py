"""MM / MMC op @handler entries + GEMM kernels + firmware dispatcher.

Single-file consolidation of the former three-layer split
(``gemm_core`` matmul / reduce / dot kernels + ``mm_engine`` decode &
variant dispatcher + ``ops/mm`` @handler decorators). Layout:

  1. GEMM kernels (``gemm_core`` / ``gemm_reduce_sum_a`` / ``gemm_dot``).
  2. L1 byte-level read/write helpers (FP16 / FP32 / bit reinterpret).
  3. Per-variant executors (``_exec_mm_basic`` / ``_s`` / ``_o`` / ``_v`` /
     ``_t``).
  4. ``decode_firmware_mm_args`` + ``firmware_mm`` dispatcher.
  5. @handler entries for the 5 MM (funct7=0x00) and 5 MMC (funct7=0x01)
     opcodes. Each handler applies Pitfall F (rs1==0 -> NOP) for
     gem5-simplified WRSPR collision safety, then forwards to
     ``firmware_mm`` with the right ``is_accumulate``/``variant`` pair.
"""
from __future__ import annotations

from typing import TYPE_CHECKING, Optional

import torch

from ...._registry import handler
from ....config_params import (
    GTX_L0_SIZE_BYTES, GTX_L1_SIZE_BYTES,
    GTX_NEST_NUM, GTX_SPU_NUM,
)
from ..encoding import (
    GSPR_GTX_OPERAND3,
    GTX_F3_MM, GTX_F3_MM_O, GTX_F3_MM_S, GTX_F3_MM_T, GTX_F3_MM_V,
    GTX_F7_RDSPR, GTX_F7_WRSPR,
    LSPR_SPM_ADDRA, LSPR_SPM_ADDRB, LSPR_SPM_ADDRC, LSPR_SPM_ADDRR,
)

if TYPE_CHECKING:
    from ...npu import GtxNpu   # noqa: F401


# =============================================================================
# 1. GEMM kernels
# =============================================================================
def _as_f32(x: torch.Tensor) -> torch.Tensor:
    """Return a contiguous FP32 view (cast if needed) for accumulation."""
    if x.dtype is torch.float32:
        return x.contiguous()
    return x.to(torch.float32).contiguous()


def gemm_core(
    A: torch.Tensor,
    B: torch.Tensor,
    *,
    has_bias: bool = False,
    bias_fp32: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """``C = A @ B [+ bias_fp32]`` — FP16 result with FP32 accumulate.

    Args:
        A: FP16 ``(M, K)`` tensor.
        B: FP16 ``(K, N)`` tensor.
        has_bias: when True, add ``bias_fp32`` to the FP32 accumulator
            before downcasting to FP16.
        bias_fp32: FP32 ``(M, N)`` bias staged from L1 ADDRC; required iff
            ``has_bias``.

    Returns:
        FP16 ``(M, N)`` result.
    """
    M, K = A.shape
    K2, N = B.shape
    if K != K2:
        raise ValueError(f"shape mismatch: A is (M={M}, K={K}), B is (K={K2}, N={N})")

    A_f32 = _as_f32(A)
    B_f32 = _as_f32(B)
    C_f32 = torch.matmul(A_f32, B_f32)

    if has_bias:
        if bias_fp32 is None:
            raise ValueError("has_bias=True requires bias_fp32 tensor")
        if tuple(bias_fp32.shape) != (M, N):
            raise ValueError(
                f"bias_fp32 shape {tuple(bias_fp32.shape)} != C shape ({M}, {N})"
            )
        if bias_fp32.dtype is not torch.float32:
            raise TypeError(
                f"bias_fp32 dtype must be float32, got {bias_fp32.dtype}"
            )
        C_f32 = C_f32 + bias_fp32

    return C_f32.to(torch.float16)


def gemm_reduce_sum_a(A: torch.Tensor, *, prior_accum: float = 0.0) -> float:
    """``MM_O`` / ``MMC_O`` scalar: ``sum(A) + prior_accum`` with FP32 reduce.

    Direct port of ``gtx_npu_mm.cc:200-211``. Returns a Python float.
    """
    A_f32 = _as_f32(A)
    s = torch.sum(A_f32) + torch.tensor(prior_accum, dtype=torch.float32,
                                        device=A_f32.device)
    return float(s.item())


def gemm_dot(A: torch.Tensor, B: torch.Tensor, *, prior_accum: float = 0.0) -> float:
    """``MM_V`` / ``MMC_V`` scalar: ``dot(A, B) + prior_accum`` with FP32 reduce."""
    if A.shape != B.shape:
        raise ValueError(f"shape mismatch: A {tuple(A.shape)} vs B {tuple(B.shape)}")
    A_f32 = _as_f32(A).flatten()
    B_f32 = _as_f32(B).flatten()
    s = torch.dot(A_f32, B_f32) + torch.tensor(prior_accum, dtype=torch.float32,
                                                device=A_f32.device)
    return float(s.item())


# =============================================================================
# 2. L1 byte-level read/write helpers
# =============================================================================
def _read_l1_fp16_matrix(npu, nest, spu, addr, rows, cols) -> torch.Tensor:
    """Read FP16 ``(rows, cols)`` from ``L1[addr:]`` little-endian (mod L1)."""
    l1 = npu.mem.l1_byte(nest, spu)
    nbytes = rows * cols * 2
    start = addr % GTX_L1_SIZE_BYTES
    if start + nbytes <= GTX_L1_SIZE_BYTES:
        return l1[start:start + nbytes].view(torch.float16).reshape(rows, cols)
    # Wrap-around: materialise into a contiguous uint8 buffer first.
    buf = torch.empty(nbytes, dtype=torch.uint8, device=l1.device)
    head = GTX_L1_SIZE_BYTES - start
    buf[:head] = l1[start:start + head]
    buf[head:] = l1[:nbytes - head]
    return buf.view(torch.float16).reshape(rows, cols)


def _write_l1_fp16_value(l1: torch.Tensor, off: int, fp16_raw: int) -> None:
    """Write a single FP16 little-endian to L1 at byte offset ``off``."""
    l1[off % GTX_L1_SIZE_BYTES] = fp16_raw & 0xFF
    l1[(off + 1) % GTX_L1_SIZE_BYTES] = (fp16_raw >> 8) & 0xFF


def _read_l1_fp32_bias(npu, nest, spu, addr, rows, cols) -> torch.Tensor:
    """Read FP32 ``(rows, cols)`` from L1 ADDRC region (little-endian)."""
    l1 = npu.mem.l1_byte(nest, spu)
    nbytes = rows * cols * 4
    start = addr % GTX_L1_SIZE_BYTES
    if start + nbytes <= GTX_L1_SIZE_BYTES:
        return l1[start:start + nbytes].view(torch.float32).reshape(rows, cols)
    buf = torch.empty(nbytes, dtype=torch.uint8, device=l1.device)
    head = GTX_L1_SIZE_BYTES - start
    buf[:head] = l1[start:start + head]
    buf[head:] = l1[:nbytes - head]
    return buf.view(torch.float32).reshape(rows, cols)


def _write_l1_fp32_value(l1: torch.Tensor, off: int, val_f32: float) -> None:
    """Write a single FP32 little-endian to L1 at byte offset ``off``."""
    raw32 = int(
        torch.tensor([val_f32], dtype=torch.float32).view(torch.int32).item()
    ) & 0xFFFFFFFF
    l1[off % GTX_L1_SIZE_BYTES] = raw32 & 0xFF
    l1[(off + 1) % GTX_L1_SIZE_BYTES] = (raw32 >> 8) & 0xFF
    l1[(off + 2) % GTX_L1_SIZE_BYTES] = (raw32 >> 16) & 0xFF
    l1[(off + 3) % GTX_L1_SIZE_BYTES] = (raw32 >> 24) & 0xFF


def _fp16_raw_bits(t: torch.Tensor) -> torch.Tensor:
    """Reinterpret an FP16 tensor as ``int16`` raw bit patterns."""
    return t.to(torch.float16).contiguous().view(torch.int16)


# =============================================================================
# 3. Per-variant executors (direct ports of the C++ exec_mm_* functions)
# =============================================================================
def _exec_mm_basic_variant(npu, nest, spu, args, is_accumulate):
    """``mm`` / ``mmc`` — A@B [+ bias] into ADDRR, FP16. No mxe_accum touch."""
    addr_a = npu.lspr[nest][spu].get(LSPR_SPM_ADDRA, 0)
    addr_b = npu.lspr[nest][spu].get(LSPR_SPM_ADDRB, 0)
    addr_r = npu.lspr[nest][spu].get(LSPR_SPM_ADDRR, 0)
    addr_c = npu.lspr[nest][spu].get(LSPR_SPM_ADDRC, 0)

    M, K, N = args['row_A'], args['col_A'], args['col_B']
    A = _read_l1_fp16_matrix(npu, nest, spu, addr_a, M, K)
    B = _read_l1_fp16_matrix(npu, nest, spu, addr_b, K, N)
    bias_fp32 = (_read_l1_fp32_bias(npu, nest, spu, addr_c, M, N)
                 if is_accumulate else None)
    C = gemm_core(A, B, has_bias=is_accumulate, bias_fp32=bias_fp32)

    # Writeback FP16 row-major LE to ADDRR.
    l1 = npu.mem.l1_byte(nest, spu)
    raw16 = _fp16_raw_bits(C).flatten()
    for i in range(raw16.numel()):
        _write_l1_fp16_value(l1, addr_r + i * 2, int(raw16[i]) & 0xFFFF)
    return 0


def _exec_mm_s_variant(npu, nest, spu, args, is_accumulate):
    """``mm_s`` / ``mmc_s`` — FP32 result into ADDRC. No mxe_accum touch."""
    addr_a = npu.lspr[nest][spu].get(LSPR_SPM_ADDRA, 0)
    addr_b = npu.lspr[nest][spu].get(LSPR_SPM_ADDRB, 0)
    addr_c = npu.lspr[nest][spu].get(LSPR_SPM_ADDRC, 0)

    M, K, N = args['row_A'], args['col_A'], args['col_B']
    A = _read_l1_fp16_matrix(npu, nest, spu, addr_a, M, K)
    B = _read_l1_fp16_matrix(npu, nest, spu, addr_b, K, N)
    A_f32 = A.to(torch.float32).contiguous()
    B_f32 = B.to(torch.float32).contiguous()
    C_f32 = torch.matmul(A_f32, B_f32)
    if is_accumulate:
        C_f32 = C_f32 + _read_l1_fp32_bias(npu, nest, spu, addr_c, M, N)

    l1 = npu.mem.l1_byte(nest, spu)
    flat = C_f32.flatten().tolist()
    for i, v in enumerate(flat):
        _write_l1_fp32_value(l1, addr_c + i * 4, float(v))
    return 0


def _exec_mm_o_variant(npu, nest, spu, args, is_accumulate):
    """``mm_o`` / ``mmc_o`` — scalar ``sum(A) [+ prior]``.

    Writes FP16 to L0 *big-endian*, updates ``mxe_accum`` unconditionally
    (Pitfall B).
    """
    addr_a = npu.lspr[nest][spu].get(LSPR_SPM_ADDRA, 0)
    col_A = args['col_A']

    A = _read_l1_fp16_matrix(npu, nest, spu, addr_a, 1, col_A).flatten()
    prior = float(npu._mxe_accum[nest, spu]) if is_accumulate else 0.0
    sum_f32 = gemm_reduce_sum_a(A, prior_accum=prior)
    npu._mxe_accum[nest, spu] = torch.tensor(
        sum_f32, dtype=torch.float32, device=npu._mxe_accum.device)

    l0_addr = int(npu.gspr.get(GSPR_GTX_OPERAND3, 0)) & 0x1F
    l0_off = (l0_addr * 32) % GTX_L0_SIZE_BYTES
    l0 = npu.mem.l0_byte(nest, spu)
    fp16_raw = int(_fp16_raw_bits(torch.tensor([sum_f32],
                                                dtype=torch.float32))[0]) & 0xFFFF
    # BIG-ENDIAN at L0 (HIGH byte first; asymmetric vs MM_V).
    l0[l0_off] = (fp16_raw >> 8) & 0xFF
    l0[(l0_off + 1) % GTX_L0_SIZE_BYTES] = fp16_raw & 0xFF
    for i in range(1, 16):
        l0[(l0_off + i * 2) % GTX_L0_SIZE_BYTES] = 0
        l0[(l0_off + i * 2 + 1) % GTX_L0_SIZE_BYTES] = 0
    return 0


def _exec_mm_v_variant(npu, nest, spu, args, is_accumulate):
    """``mm_v`` / ``mmc_v`` — scalar ``dot(A, B) [+ prior]``.

    Writes FP16 to L0 *little-endian* (asymmetric vs MM_O), updates
    ``mxe_accum``.
    """
    addr_a = npu.lspr[nest][spu].get(LSPR_SPM_ADDRA, 0)
    addr_b = npu.lspr[nest][spu].get(LSPR_SPM_ADDRB, 0)
    vec_len = args['col_A']

    A = _read_l1_fp16_matrix(npu, nest, spu, addr_a, 1, vec_len).flatten()
    B = _read_l1_fp16_matrix(npu, nest, spu, addr_b, 1, vec_len).flatten()
    prior = float(npu._mxe_accum[nest, spu]) if is_accumulate else 0.0
    dot_f32 = gemm_dot(A, B, prior_accum=prior)
    npu._mxe_accum[nest, spu] = torch.tensor(
        dot_f32, dtype=torch.float32, device=npu._mxe_accum.device)

    l0_addr = int(npu.gspr.get(GSPR_GTX_OPERAND3, 0)) & 0x1F
    l0_off = (l0_addr * 32) % GTX_L0_SIZE_BYTES
    l0 = npu.mem.l0_byte(nest, spu)
    fp16_raw = int(_fp16_raw_bits(torch.tensor([dot_f32],
                                                dtype=torch.float32))[0]) & 0xFFFF
    # LITTLE-ENDIAN at L0 (gtx_npu_mm.cc:274-275).
    l0[l0_off] = fp16_raw & 0xFF
    l0[(l0_off + 1) % GTX_L0_SIZE_BYTES] = (fp16_raw >> 8) & 0xFF
    for i in range(1, 16):
        l0[(l0_off + i * 2) % GTX_L0_SIZE_BYTES] = 0
        l0[(l0_off + i * 2 + 1) % GTX_L0_SIZE_BYTES] = 0
    return 0


def _exec_mm_t_variant(npu, nest, spu, args, is_accumulate):
    """``mm_t`` / ``mmc_t`` — transposed C^T (N×M layout) into ADDRR."""
    addr_a = npu.lspr[nest][spu].get(LSPR_SPM_ADDRA, 0)
    addr_b = npu.lspr[nest][spu].get(LSPR_SPM_ADDRB, 0)
    addr_r = npu.lspr[nest][spu].get(LSPR_SPM_ADDRR, 0)
    addr_c = npu.lspr[nest][spu].get(LSPR_SPM_ADDRC, 0)

    M, K, N = args['row_A'], args['col_A'], args['col_B']
    A = _read_l1_fp16_matrix(npu, nest, spu, addr_a, M, K)
    B = _read_l1_fp16_matrix(npu, nest, spu, addr_b, K, N)
    bias_fp32 = (_read_l1_fp32_bias(npu, nest, spu, addr_c, M, N)
                 if is_accumulate else None)
    C = gemm_core(A, B, has_bias=is_accumulate, bias_fp32=bias_fp32)

    # Transposed writeback: out[j, i] = C[i, j] at byte offset (i + M*j)*2.
    l1 = npu.mem.l1_byte(nest, spu)
    raw16 = _fp16_raw_bits(C)
    for i in range(M):
        for j in range(N):
            _write_l1_fp16_value(l1, addr_r + (i + M * j) * 2,
                                  int(raw16[i, j]) & 0xFFFF)
    return 0


# =============================================================================
# 4. decode_firmware_mm_args + firmware_mm dispatcher
# =============================================================================
def decode_firmware_mm_args(rs1: int) -> dict:
    """Decode packed ``rs1`` → ``{'row_A', 'col_A', 'col_B'}``.

    ``rs1`` layout: ``colB[63:48] | <reserved[47:32]> | colA[31:16] | rowA[15:0]``.
    HW convention: 0 in any 16-bit field means ``0x10000``.
    """
    def dim16(v: int) -> int:
        d = v & 0xFFFF
        return d if d != 0 else 0x10000
    return {
        'row_A': dim16(rs1),
        'col_A': dim16(rs1 >> 16),
        'col_B': dim16(rs1 >> 48),
    }


def firmware_mm(npu: 'GtxNpu', proc, insn,
                *, is_accumulate: bool, variant: str) -> int:
    """Direct port of ``gtx_npu_mm.cc:333-389 firmware_mm_op``."""
    rs1 = proc.state.XPR[insn.rs1]
    args = decode_firmware_mm_args(rs1)

    nest = npu.warp.tmu_id if npu.warp.is_ploop else 0
    spu = npu.warp.curr_id if npu.warp.is_tloop else 0
    if nest >= GTX_NEST_NUM:
        nest = 0
    if spu >= GTX_SPU_NUM:
        spu = 0

    if variant in ('mm_o', 'mmc_o'):
        return _exec_mm_o_variant(npu, nest, spu, args, is_accumulate)
    if variant in ('mm_v', 'mmc_v'):
        return _exec_mm_v_variant(npu, nest, spu, args, is_accumulate)
    if variant in ('mm_s', 'mmc_s'):
        return _exec_mm_s_variant(npu, nest, spu, args, is_accumulate)
    if variant in ('mm_t', 'mmc_t'):
        return _exec_mm_t_variant(npu, nest, spu, args, is_accumulate)
    return _exec_mm_basic_variant(npu, nest, spu, args, is_accumulate)


# =============================================================================
# 5. @handler entries
#
# Pitfall F (gtx_npu_mm.cc rs1==0 guard): funct7=0x00 collides with
# gem5-simplified WRSPR. Each per-funct3 handler NOPs if ``insn.rs1 == 0``
# so the dispatch table can keep funct3-keyed entries without losing the
# WRSPR semantics.
# =============================================================================
@handler(kind='custom0', funct7=GTX_F7_WRSPR, funct3=GTX_F3_MM_S,
         mnemonic='mm_s', mask_funct3=True)
def _exec_mm_s(npu, proc, insn, xs1, xs2):
    """funct7=0x00 funct3=0 -> mm_s (FP32 result to ADDRC)."""
    if insn.rs1 == 0:
        return 0
    return firmware_mm(npu, proc, insn, is_accumulate=False, variant='mm_s')


@handler(kind='custom0', funct7=GTX_F7_WRSPR, funct3=GTX_F3_MM_O,
         mnemonic='mm_o', mask_funct3=True)
def _exec_mm_o(npu, proc, insn, xs1, xs2):
    """funct7=0x00 funct3=1 -> mm_o (scalar sum(A) to L0 BE + mxe_accum)."""
    if insn.rs1 == 0:
        return 0
    return firmware_mm(npu, proc, insn, is_accumulate=False, variant='mm_o')


@handler(kind='custom0', funct7=GTX_F7_WRSPR, funct3=GTX_F3_MM,
         mnemonic='mm', mask_funct3=True)
def _exec_mm(npu, proc, insn, xs1, xs2):
    """funct7=0x00 funct3=2 -> mm (basic GEMM, FP16 result to ADDRR)."""
    if insn.rs1 == 0:
        return 0
    return firmware_mm(npu, proc, insn, is_accumulate=False, variant='mm')


@handler(kind='custom0', funct7=GTX_F7_WRSPR, funct3=GTX_F3_MM_V,
         mnemonic='mm_v', mask_funct3=True)
def _exec_mm_v(npu, proc, insn, xs1, xs2):
    """funct7=0x00 funct3=3 -> mm_v (scalar dot(A,B) to L0 LE + mxe_accum)."""
    if insn.rs1 == 0:
        return 0
    return firmware_mm(npu, proc, insn, is_accumulate=False, variant='mm_v')


@handler(kind='custom0', funct7=GTX_F7_WRSPR, funct3=GTX_F3_MM_T,
         mnemonic='mm_t', mask_funct3=True)
def _exec_mm_t(npu, proc, insn, xs1, xs2):
    """funct7=0x00 funct3=7 -> mm_t (transposed C^T to ADDRR)."""
    if insn.rs1 == 0:
        return 0
    return firmware_mm(npu, proc, insn, is_accumulate=False, variant='mm_t')


# ----- MMC family (funct7=0x01, is_accumulate=True) --------------------------

@handler(kind='custom0', funct7=GTX_F7_RDSPR, funct3=GTX_F3_MM_S,
         mnemonic='mmc_s', mask_funct3=True)
def _exec_mmc_s(npu, proc, insn, xs1, xs2):
    """funct7=0x01 funct3=0 -> mmc_s (FP32 result to ADDRC, accumulate)."""
    if insn.rs1 == 0:
        return 0
    return firmware_mm(npu, proc, insn, is_accumulate=True, variant='mmc_s')


@handler(kind='custom0', funct7=GTX_F7_RDSPR, funct3=GTX_F3_MM_O,
         mnemonic='mmc_o', mask_funct3=True)
def _exec_mmc_o(npu, proc, insn, xs1, xs2):
    """funct7=0x01 funct3=1 -> mmc_o (mxe_accum chain: prior + sum(A))."""
    if insn.rs1 == 0:
        return 0
    return firmware_mm(npu, proc, insn, is_accumulate=True, variant='mmc_o')


@handler(kind='custom0', funct7=GTX_F7_RDSPR, funct3=GTX_F3_MM,
         mnemonic='mmc', mask_funct3=True)
def _exec_mmc(npu, proc, insn, xs1, xs2):
    """funct7=0x01 funct3=2 -> mmc (basic GEMM with ADDRC FP32 bias)."""
    if insn.rs1 == 0:
        return 0
    return firmware_mm(npu, proc, insn, is_accumulate=True, variant='mmc')


@handler(kind='custom0', funct7=GTX_F7_RDSPR, funct3=GTX_F3_MM_V,
         mnemonic='mmc_v', mask_funct3=True)
def _exec_mmc_v(npu, proc, insn, xs1, xs2):
    """funct7=0x01 funct3=3 -> mmc_v (mxe_accum chain: prior + dot(A,B))."""
    if insn.rs1 == 0:
        return 0
    return firmware_mm(npu, proc, insn, is_accumulate=True, variant='mmc_v')


@handler(kind='custom0', funct7=GTX_F7_RDSPR, funct3=GTX_F3_MM_T,
         mnemonic='mmc_t', mask_funct3=True)
def _exec_mmc_t(npu, proc, insn, xs1, xs2):
    """funct7=0x01 funct3=7 -> mmc_t (transposed C^T to ADDRR, accumulate)."""
    if insn.rs1 == 0:
        return 0
    return firmware_mm(npu, proc, insn, is_accumulate=True, variant='mmc_t')
