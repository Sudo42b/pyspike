#
# Copyright 2026 WuXi EsionTech Co., Ltd.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#    http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#
"""MM engine -- spike-bound dispatcher for firmware_mm_op.

Direct port of vendor/gtx_cpp_reference/gtx/gtx_npu_mm.cc:333-389 (firmware_mm_op)
and per-variant exec_mm_* (lines 106-315).

Per CONTEXT D-01: spike-bound layer (reads npu/proc/insn). Pure GEMM kernel
delegated to gemm_core.py (Plan 02). @handler entries live in ops/mm.py (Plan 04).

Per RESEARCH Pitfall B: MM_O/MMC_O/MM_V/MMC_V touch npu._mxe_accum;
MM_S/MMC_S/MM/MMC/MM_T/MMC_T use ADDRC FP32 staging only.

Per RESEARCH Pitfall G: nest = warp.tmu_id if is_ploop else 0;
                        spu  = warp.curr_id if is_tloop else 0.

Phase 4 plan 03 Task 1.
"""
from __future__ import annotations
from typing import TYPE_CHECKING

import numpy as np

from .gemm_core import gemm_core, gemm_reduce_sum_a, gemm_dot
from .encoding import (
    GSPR_GTX_OPERAND3,
    LSPR_SPM_ADDRA, LSPR_SPM_ADDRB, LSPR_SPM_ADDRC, LSPR_SPM_ADDRR,
)
from .params import (
    GTX_NEST_NUM, GTX_SPU_NUM,
    GTX_L0_SIZE_BYTES, GTX_L1_SIZE_BYTES,
)

if TYPE_CHECKING:
    from .npu import GtxNpu


# =========================================================================
# decode_firmware_mm_args -- direct port of gtx_npu_mm.cc:347-355 dim16 lambda.
# =========================================================================
def decode_firmware_mm_args(rs1: int) -> dict:
    """Decode packed rs1 -> {row_A, col_A, col_B}.

    rs1 layout: colB[63:48] | <reserved[47:32]> | colA[31:16] | rowA[15:0].
    HW convention: 0 in any 16-bit field means 65536 (per-field independent;
    Pitfall C -- NOT a whole-word check).

    Verified against gtx_npu_mm.cc:347-355:
        auto dim16 = [](uint64_t v) -> uint32_t {
            uint32_t d = v & 0xFFFF;
            return d ? d : 0x10000;
        };
        uint32_t row_A = dim16(rs1);
        uint32_t col_A = dim16(rs1 >> 16);
        uint32_t col_B = dim16(rs1 >> 48);
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
# firmware_mm -- main dispatcher invoked by ops/mm.py @handlers.
# =========================================================================
def firmware_mm(npu: 'GtxNpu', proc, insn,
                *, is_accumulate: bool, variant: str) -> int:
    """Direct port of gtx_npu_mm.cc:333-389 firmware_mm_op.

    Args:
        npu: GtxNpu instance (read npu.warp / npu.lspr / npu.mem / npu._mxe_accum)
        proc: spike processor_t (read XPR via proc.state.XPR[idx])
        insn: rocc_insn_t (read insn.rs1 register INDEX, NOT value -- Pitfall 4)
        is_accumulate: True for MMC family (funct7=0x01), False for MM (funct7=0x00)
        variant: one of 'mm', 'mm_s', 'mm_o', 'mm_v', 'mm_t',
                          'mmc', 'mmc_s', 'mmc_o', 'mmc_v', 'mmc_t'

    Returns: 0 (cycle count vestigial in functional model)
    """
    # Pitfall 4: read register VALUE via proc.state.XPR[insn.rs1].
    # The xs1 arg from RoCC trampoline is unreliable when xs1 flag is 0.
    rs1 = proc.state.XPR[insn.rs1]
    args = decode_firmware_mm_args(rs1)

    # Pitfall G: explicit is_ploop/is_tloop guards (gtx_npu_mm.cc:338-339).
    nest = npu.warp.tmu_id if npu.warp.is_ploop else 0
    spu = npu.warp.curr_id if npu.warp.is_tloop else 0
    if nest >= GTX_NEST_NUM:
        nest = 0
    if spu >= GTX_SPU_NUM:
        spu = 0

    # Variant dispatch -- Pitfall E: unknown variant defaults to mm_basic.
    if variant in ('mm_o', 'mmc_o'):
        return _exec_mm_o_variant(npu, nest, spu, args, is_accumulate)
    if variant in ('mm_v', 'mmc_v'):
        return _exec_mm_v_variant(npu, nest, spu, args, is_accumulate)
    if variant in ('mm_s', 'mmc_s'):
        return _exec_mm_s_variant(npu, nest, spu, args, is_accumulate)
    if variant in ('mm_t', 'mmc_t'):
        return _exec_mm_t_variant(npu, nest, spu, args, is_accumulate)
    # variant in ('mm', 'mmc') OR unknown -> default to basic
    return _exec_mm_basic_variant(npu, nest, spu, args, is_accumulate)


# =========================================================================
# L1 byte-level read/write helpers.
#
# All FP16 in L1 is little-endian per gtx_npu_mm.cc:38 (low byte first).
# All FP32 staging in L1 ADDRC is little-endian per gtx_npu_mm.cc:88 (memcpy).
# Modular L1_SIZE addressing matches C++ `% GTX_L1_SIZE`.
# =========================================================================
def _read_l1_fp16_matrix(npu: 'GtxNpu', nest: int, spu: int,
                          addr: int, rows: int, cols: int) -> np.ndarray:
    """Read FP16 (rows, cols) from L1[addr:] little-endian, modular L1_SIZE."""
    l1 = npu.mem.l1_byte(nest, spu)
    out = np.zeros((rows, cols), dtype=np.float16)
    for i in range(rows):
        for j in range(cols):
            off = (addr + (i * cols + j) * 2) % GTX_L1_SIZE_BYTES
            lo = int(l1[off])
            hi = int(l1[(off + 1) % GTX_L1_SIZE_BYTES])
            raw = (hi << 8) | lo  # LE: low byte first in memory
            out[i, j] = np.frombuffer(np.uint16(raw).tobytes(), dtype=np.float16)[0]
    return out


def _write_l1_fp16_value(l1: np.ndarray, off: int, fp16_raw: int) -> None:
    """Write a single FP16 little-endian to L1 at given byte offset."""
    l1[off % GTX_L1_SIZE_BYTES] = fp16_raw & 0xFF
    l1[(off + 1) % GTX_L1_SIZE_BYTES] = (fp16_raw >> 8) & 0xFF


def _read_l1_fp32_bias(npu: 'GtxNpu', nest: int, spu: int,
                        addr: int, rows: int, cols: int) -> np.ndarray:
    """Read FP32 (rows, cols) from L1[addr:] (ADDRC FP32 staging).

    Direct port of gtx_npu_mm.cc:88 `std::memcpy(&bias, &spu.l1[c_off], 4)`.
    """
    l1 = npu.mem.l1_byte(nest, spu)
    out = np.zeros((rows, cols), dtype=np.float32)
    for i in range(rows):
        for j in range(cols):
            off = (addr + (i * cols + j) * 4) % GTX_L1_SIZE_BYTES
            b0 = int(l1[off])
            b1 = int(l1[(off + 1) % GTX_L1_SIZE_BYTES])
            b2 = int(l1[(off + 2) % GTX_L1_SIZE_BYTES])
            b3 = int(l1[(off + 3) % GTX_L1_SIZE_BYTES])
            raw32 = b0 | (b1 << 8) | (b2 << 16) | (b3 << 24)
            out[i, j] = np.frombuffer(np.uint32(raw32).tobytes(), dtype=np.float32)[0]
    return out


def _write_l1_fp32_value(l1: np.ndarray, off: int, val_f32: float) -> None:
    """Write a single FP32 little-endian to L1 at given byte offset.

    Direct port of gtx_npu_mm.cc:173 `std::memcpy(&spu.l1[off], &val, 4)`.
    """
    raw32 = int(np.float32(val_f32).view(np.uint32))
    l1[off % GTX_L1_SIZE_BYTES] = raw32 & 0xFF
    l1[(off + 1) % GTX_L1_SIZE_BYTES] = (raw32 >> 8) & 0xFF
    l1[(off + 2) % GTX_L1_SIZE_BYTES] = (raw32 >> 16) & 0xFF
    l1[(off + 3) % GTX_L1_SIZE_BYTES] = (raw32 >> 24) & 0xFF


# =========================================================================
# Per-variant helpers. Each is a direct port of the corresponding C++ exec_mm_*.
# =========================================================================

def _exec_mm_basic_variant(npu, nest, spu, args, is_accumulate):
    """Direct port of gtx_npu_mm.cc:106-140 (exec_mm / exec_mmc).

    Reads A, B from L1 ADDRA/ADDRB; bias from ADDRC if is_accumulate;
    writes C (FP16) to ADDRR. Does NOT touch mxe_accum.
    """
    addr_a = npu.lspr[nest][spu].get(LSPR_SPM_ADDRA, 0)
    addr_b = npu.lspr[nest][spu].get(LSPR_SPM_ADDRB, 0)
    addr_r = npu.lspr[nest][spu].get(LSPR_SPM_ADDRR, 0)
    addr_c = npu.lspr[nest][spu].get(LSPR_SPM_ADDRC, 0)

    M, K, N = args['row_A'], args['col_A'], args['col_B']
    A = _read_l1_fp16_matrix(npu, nest, spu, addr_a, M, K)
    B = _read_l1_fp16_matrix(npu, nest, spu, addr_b, K, N)
    bias_fp32 = None
    if is_accumulate:
        bias_fp32 = _read_l1_fp32_bias(npu, nest, spu, addr_c, M, N)
    C = gemm_core(A, B, has_bias=is_accumulate, bias_fp32=bias_fp32)

    # Write FP16 result to ADDRR row-major LE (gtx_npu_mm.cc:131-136).
    l1 = npu.mem.l1_byte(nest, spu)
    flat = C.flatten()
    for i, v in enumerate(flat):
        raw = int(v.view(np.uint16))
        _write_l1_fp16_value(l1, addr_r + i * 2, raw)
    return 0


def _exec_mm_s_variant(npu, nest, spu, args, is_accumulate):
    """Direct port of gtx_npu_mm.cc:150-176 (exec_mm_s / exec_mmc_s).

    Computes A @ B in FP32; writes FP32 result bytes to ADDRC (next mm/mmc reads).
    If is_accumulate: reads prior FP32 bias from ADDRC, adds to result before writeback.
    Does NOT touch mxe_accum.
    """
    addr_a = npu.lspr[nest][spu].get(LSPR_SPM_ADDRA, 0)
    addr_b = npu.lspr[nest][spu].get(LSPR_SPM_ADDRB, 0)
    addr_c = npu.lspr[nest][spu].get(LSPR_SPM_ADDRC, 0)

    M, K, N = args['row_A'], args['col_A'], args['col_B']
    A = _read_l1_fp16_matrix(npu, nest, spu, addr_a, M, K)
    B = _read_l1_fp16_matrix(npu, nest, spu, addr_b, K, N)
    # Explicit 3-loop FP32 (matches gemm_core ordering, but no FP16 cast).
    A_f32 = A.astype(np.float32)
    B_f32 = B.astype(np.float32)
    C_f32 = np.zeros((M, N), dtype=np.float32)
    for i in range(M):
        for j in range(N):
            s = np.float32(0.0)
            for k in range(K):
                s += A_f32[i, k] * B_f32[k, j]
            C_f32[i, j] = s
    if is_accumulate:
        prior = _read_l1_fp32_bias(npu, nest, spu, addr_c, M, N)
        C_f32 += prior

    l1 = npu.mem.l1_byte(nest, spu)
    flat = C_f32.flatten()
    for i, v in enumerate(flat):
        _write_l1_fp32_value(l1, addr_c + i * 4, float(v))
    return 0


def _exec_mm_o_variant(npu, nest, spu, args, is_accumulate):
    """Direct port of gtx_npu_mm.cc:186-225 (exec_mm_o / exec_mmc_o).

    Reads A as FP16 (col_A,) from ADDRA; computes scalar sum(A) [+ prior_accum];
    writes FP16 to L0 BIG-ENDIAN (gtx_npu_mm.cc:217-218); writes mxe_accum.

    Note (Pitfall B): mxe_accum is written unconditionally by both MM_O and MMC_O.
    """
    addr_a = npu.lspr[nest][spu].get(LSPR_SPM_ADDRA, 0)
    col_A = args['col_A']

    A = _read_l1_fp16_matrix(npu, nest, spu, addr_a, 1, col_A).flatten()

    prior = float(npu._mxe_accum[nest, spu]) if is_accumulate else 0.0
    sum_f32 = gemm_reduce_sum_a(A, prior_accum=prior)
    npu._mxe_accum[nest, spu] = np.float32(sum_f32)

    # gtx_npu_mm.cc:215 -- l0_addr from gspr[GSPR_GTX_OPERAND3] & 0x1F (low 5 bits).
    l0_addr = int(npu.gspr.get(GSPR_GTX_OPERAND3, 0)) & 0x1F
    l0_off = (l0_addr * 32) % GTX_L0_SIZE_BYTES
    l0 = npu.mem.l0_byte(nest, spu)
    fp16_raw = int(np.float16(sum_f32).view(np.uint16))
    # BIG-ENDIAN at L0 (gtx_npu_mm.cc:217-218 -- HIGH byte first; asymmetry vs MM_V LE!).
    l0[l0_off] = (fp16_raw >> 8) & 0xFF
    l0[(l0_off + 1) % GTX_L0_SIZE_BYTES] = fp16_raw & 0xFF
    # Zero remaining 15 FP16 slots (gtx_npu_mm.cc:220-223).
    for i in range(1, 16):
        l0[(l0_off + i * 2) % GTX_L0_SIZE_BYTES] = 0
        l0[(l0_off + i * 2 + 1) % GTX_L0_SIZE_BYTES] = 0
    return 0


def _exec_mm_v_variant(npu, nest, spu, args, is_accumulate):
    """Direct port of gtx_npu_mm.cc:233-281 (exec_mm_v / exec_mmc_v).

    Reads A and B as FP16 vectors (col,); computes scalar dot(A, B) [+ prior];
    writes FP16 to L0 LITTLE-ENDIAN (gtx_npu_mm.cc:274-275 -- asymmetry vs MM_O!);
    writes mxe_accum.
    """
    addr_a = npu.lspr[nest][spu].get(LSPR_SPM_ADDRA, 0)
    addr_b = npu.lspr[nest][spu].get(LSPR_SPM_ADDRB, 0)
    vec_len = args['col_A']  # MM_V uses col_A as vector length (gtx_npu_mm.cc:368)

    A = _read_l1_fp16_matrix(npu, nest, spu, addr_a, 1, vec_len).flatten()
    B = _read_l1_fp16_matrix(npu, nest, spu, addr_b, 1, vec_len).flatten()

    prior = float(npu._mxe_accum[nest, spu]) if is_accumulate else 0.0
    dot_f32 = gemm_dot(A, B, prior_accum=prior)
    npu._mxe_accum[nest, spu] = np.float32(dot_f32)

    l0_addr = int(npu.gspr.get(GSPR_GTX_OPERAND3, 0)) & 0x1F
    l0_off = (l0_addr * 32) % GTX_L0_SIZE_BYTES
    l0 = npu.mem.l0_byte(nest, spu)
    fp16_raw = int(np.float16(dot_f32).view(np.uint16))
    # MM_V is LITTLE-ENDIAN at L0 -- different from MM_O (gtx_npu_mm.cc:274-275).
    l0[l0_off] = fp16_raw & 0xFF
    l0[(l0_off + 1) % GTX_L0_SIZE_BYTES] = (fp16_raw >> 8) & 0xFF
    for i in range(1, 16):
        l0[(l0_off + i * 2) % GTX_L0_SIZE_BYTES] = 0
        l0[(l0_off + i * 2 + 1) % GTX_L0_SIZE_BYTES] = 0
    return 0


def _exec_mm_t_variant(npu, nest, spu, args, is_accumulate):
    """Direct port of gtx_npu_mm.cc:289-315 (exec_mm_t / exec_mmc_t).

    Computes C = A @ B [+bias from ADDRC]; writes C^T (transposed, N x M layout)
    to ADDRR FP16 little-endian. Does NOT touch mxe_accum.

    Pitfall D: output L1 region is N x M not M x N. Writeback offset is
    `addr_r + (i + M*j)*2` per gtx_npu_mm.cc:308.
    """
    addr_a = npu.lspr[nest][spu].get(LSPR_SPM_ADDRA, 0)
    addr_b = npu.lspr[nest][spu].get(LSPR_SPM_ADDRB, 0)
    addr_r = npu.lspr[nest][spu].get(LSPR_SPM_ADDRR, 0)
    addr_c = npu.lspr[nest][spu].get(LSPR_SPM_ADDRC, 0)

    M, K, N = args['row_A'], args['col_A'], args['col_B']
    A = _read_l1_fp16_matrix(npu, nest, spu, addr_a, M, K)
    B = _read_l1_fp16_matrix(npu, nest, spu, addr_b, K, N)
    bias_fp32 = None
    if is_accumulate:
        bias_fp32 = _read_l1_fp32_bias(npu, nest, spu, addr_c, M, N)
    C = gemm_core(A, B, has_bias=is_accumulate, bias_fp32=bias_fp32)

    # Transposed writeback: out[j, i] = C[i, j] at offset (i + M*j) * 2.
    l1 = npu.mem.l1_byte(nest, spu)
    for i in range(M):
        for j in range(N):
            raw = int(C[i, j].view(np.uint16))
            off = (addr_r + (i + M * j) * 2)
            _write_l1_fp16_value(l1, off, raw)
    return 0
