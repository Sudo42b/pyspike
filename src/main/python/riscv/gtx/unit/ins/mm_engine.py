"""MM dispatcher — PyTorch-native L1 read/write + gemm_core invocation.

Direct port of ``gtx_npu_mm.cc``. All compute and memory views use
:mod:`torch`; the byte-level L1 store is the GTX scratchpad backing
tensor (uint8 contiguous block) and FP16/FP32 values are deposited as
their native little-endian bit patterns via ``.view(dtype)``.

Public entry
    decode_firmware_mm_args(rs1)  → {'row_A','col_A','col_B'}
    firmware_mm(npu, proc, insn, *, is_accumulate, variant) → 0
"""
from __future__ import annotations

from typing import TYPE_CHECKING

import torch

from .gemm_core import gemm_core, gemm_reduce_sum_a, gemm_dot
from .encoding import (
    GSPR_GTX_OPERAND3,
    LSPR_SPM_ADDRA, LSPR_SPM_ADDRB, LSPR_SPM_ADDRC, LSPR_SPM_ADDRR,
)
from ...config_params import (
    GTX_NEST_NUM, GTX_SPU_NUM,
    GTX_L0_SIZE_BYTES, GTX_L1_SIZE_BYTES,
)

if TYPE_CHECKING:
    from ...npu import GtxNpu


# =========================================================================
# decode_firmware_mm_args — direct port of gtx_npu_mm.cc:347-355 dim16 lambda.
# =========================================================================
def decode_firmware_mm_args(rs1: int) -> dict:
    """Decode packed ``rs1`` → ``{'row_A','col_A','col_B'}``.

    ``rs1`` layout: ``colB[63:48] | <reserved[47:32]> | colA[31:16] | rowA[15:0]``.
    HW convention: 0 in any 16-bit field means ``0x10000`` (per-field
    independent; Pitfall C — *not* a whole-word check).
    """
    def dim16(v: int) -> int:
        d = v & 0xFFFF
        return d if d != 0 else 0x10000
    return {
        'row_A': dim16(rs1),
        'col_A': dim16(rs1 >> 16),
        'col_B': dim16(rs1 >> 48),
    }


# =========================================================================
# firmware_mm — main dispatcher invoked by ops/mm.py @handlers.
# =========================================================================
def firmware_mm(npu: 'GtxNpu', proc, insn,
                *, is_accumulate: bool, variant: str) -> int:
    """Direct port of ``gtx_npu_mm.cc:333-389 firmware_mm_op``.

    Args
        npu: GtxNpu instance (reads ``warp`` / ``lspr`` / ``mem`` / ``_mxe_accum``).
        proc: spike ``processor_t`` (reads XPR via ``proc.state.XPR[idx]``).
        insn: ``rocc_insn_t`` (reads ``insn.rs1`` register *index*, not value
            — Pitfall 4).
        is_accumulate: True for MMC family (``funct7=0x01``), False for MM
            (``funct7=0x00``).
        variant: one of ``'mm','mm_s','mm_o','mm_v','mm_t','mmc','mmc_s',
            'mmc_o','mmc_v','mmc_t'``.

    Returns: 0 (cycle count is vestigial in the functional model).
    """
    rs1 = proc.state.XPR[insn.rs1]
    args = decode_firmware_mm_args(rs1)

    # Pitfall G: explicit is_ploop / is_tloop guards (gtx_npu_mm.cc:338-339).
    nest = npu.warp.tmu_id if npu.warp.is_ploop else 0
    spu = npu.warp.curr_id if npu.warp.is_tloop else 0
    if nest >= GTX_NEST_NUM:
        nest = 0
    if spu >= GTX_SPU_NUM:
        spu = 0

    # Variant dispatch — Pitfall E: unknown variant falls through to basic.
    if variant in ('mm_o', 'mmc_o'):
        return _exec_mm_o_variant(npu, nest, spu, args, is_accumulate)
    if variant in ('mm_v', 'mmc_v'):
        return _exec_mm_v_variant(npu, nest, spu, args, is_accumulate)
    if variant in ('mm_s', 'mmc_s'):
        return _exec_mm_s_variant(npu, nest, spu, args, is_accumulate)
    if variant in ('mm_t', 'mmc_t'):
        return _exec_mm_t_variant(npu, nest, spu, args, is_accumulate)
    return _exec_mm_basic_variant(npu, nest, spu, args, is_accumulate)


# =========================================================================
# L1 byte-level read/write helpers — torch.uint8 buffers, FP16 LE bit pattern.
# Modular ``GTX_L1_SIZE_BYTES`` addressing matches C++ ``% GTX_L1_SIZE``.
# =========================================================================
def _read_l1_fp16_matrix(npu, nest, spu, addr, rows, cols) -> torch.Tensor:
    """Read FP16 ``(rows, cols)`` from ``L1[addr:]`` little-endian (mod L1).

    Fast path uses :func:`torch.Tensor.view` when the read is contiguous and
    aligned; falls back to a 2-byte gather for the wrap-around case.
    """
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


# =========================================================================
# Per-variant helpers — direct ports of the C++ exec_mm_* functions.
# =========================================================================

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

    # L0 base from gspr[OPERAND3] & 0x1F (low 5 bits).
    l0_addr = int(npu.gspr.get(GSPR_GTX_OPERAND3, 0)) & 0x1F
    l0_off = (l0_addr * 32) % GTX_L0_SIZE_BYTES
    l0 = npu.mem.l0_byte(nest, spu)
    fp16_raw = int(_fp16_raw_bits(torch.tensor([sum_f32], dtype=torch.float32))[0]) & 0xFFFF
    # BIG-ENDIAN at L0 (gtx_npu_mm.cc:217-218 — HIGH byte first; asymmetric vs MM_V).
    l0[l0_off] = (fp16_raw >> 8) & 0xFF
    l0[(l0_off + 1) % GTX_L0_SIZE_BYTES] = fp16_raw & 0xFF
    # Zero the remaining 15 FP16 slots.
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
    fp16_raw = int(_fp16_raw_bits(torch.tensor([dot_f32], dtype=torch.float32))[0]) & 0xFFFF
    # LITTLE-ENDIAN at L0 (gtx_npu_mm.cc:274-275).
    l0[l0_off] = fp16_raw & 0xFF
    l0[(l0_off + 1) % GTX_L0_SIZE_BYTES] = (fp16_raw >> 8) & 0xFF
    for i in range(1, 16):
        l0[(l0_off + i * 2) % GTX_L0_SIZE_BYTES] = 0
        l0[(l0_off + i * 2 + 1) % GTX_L0_SIZE_BYTES] = 0
    return 0


def _exec_mm_t_variant(npu, nest, spu, args, is_accumulate):
    """``mm_t`` / ``mmc_t`` — transposed C^T (N×M layout) into ADDRR. No mxe_accum touch."""
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
