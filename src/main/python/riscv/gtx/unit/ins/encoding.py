"""GTX RoCC instruction encoding constants.

Direct port of vendor/gtx_cpp_reference/gtx/gtx_npu.h

Phase 2 (D-11): full funct7 set + custom1 funct3 + opcode constants.
"""
from enum import Enum

class CUSTOM_OPCODE(Enum):
    # ----- RoCC custom opcodes -----
    CUSTOM0 = 0x0b # custom-0
    CUSTOM1 = 0x2b # custom-1

# ----- funct7 (custom0) -- gtx_npu.h:266-273 -----
GTX_F7_WRSPR: int = 0x00        # WRSPR (gem5) / MM ISS-full (rs1!=0 -> P4 MM)
GTX_F7_RDSPR: int = 0x01        # RDSPR (gem5) / MMC ISS-full (rs1!=0 -> P4 MMC)
GTX_F7_WSPLIT: int = 0x02
GTX_F7_WJOIN: int = 0x03        # WJOIN custom0 firmware variant (no exit)
GTX_F7_DISPATCH_MM: int = 0x04
GTX_F7_DISPATCH_VEC: int = 0x05
GTX_F7_DISPATCH_ACT: int = 0x06
GTX_F7_DISPATCH_DMA: int = 0x07

# ----- ISS-full funct7 (custom0) -- gtx_npu.h:276-282 (P2 subset) -----
GTX_ISS_F7_RDSPR_ISS: int = 0x48        # 0b1001000
GTX_ISS_F7_WRSPR_ISS: int = 0x49        # 0b1001001
GTX_ISS_F7_OPSET: int = 0x4A            # P3 (DMA staging)
GTX_ISS_F7_DEBUG_WR: int = 0x7D
GTX_ISS_F7_DEBUG_RD: int = 0x7E

# ----- custom1 funct3 (warp control) -- gtx_npu_custom1.cc:21-26 -----
WARP_F3_START_T: int = 0b000
WARP_F3_END_T: int = 0b001
WARP_F3_START_S: int = 0b010
WARP_F3_END_S: int = 0b011
WARP_F3_START_P: int = 0b110
WARP_F3_END_P: int = 0b111
WARP_F3_SPLIT: int = 0b100
WARP_F3_JOIN: int = 0b101

# ----- Mode constants -- gtx_npu.h:289-292 -----
GTX_OP_MM: int = 0
GTX_OP_VECTOR: int = 1
GTX_OP_ACTIVATION: int = 2
GTX_OP_DMA: int = 3

# ----- OPSET staging GSPR addresses (gtx_params.h:36-44) -----
# OPERAND0..2 are general-purpose staging registers (rs3/rs4/rs5 source).
# OPERAND3 / OPERAND4 are the OPSET staging slots cleared after every
# non-OPSET custom0 instruction (see writeback.state_writeback).
GSPR_GTX_OPERAND0: int = 0x000
GSPR_GTX_OPERAND1: int = 0x001
GSPR_GTX_OPERAND2: int = 0x002
GSPR_GTX_OPERAND3: int = 0x003   # OPSET staging slot 3 (gtx_params.h:40)
GSPR_GTX_OPERAND4: int = 0x004
GSPR_GTX_OPERAND5: int = 0x005   # OPSET staging slot 4 (gtx_params.h:44)
GSPR_GTX_OPCODE:   int = 0x012   # firmware sub-opcode staging (low byte mask)

# ----- LSPR SPM base-address registers (gtx_params.h:64-67) -----
LSPR_SPM_ADDRA: int = 0x900
LSPR_SPM_ADDRB: int = 0x901
LSPR_SPM_ADDRC: int = 0x902
LSPR_SPM_ADDRR: int = 0x903

# ----- NSPR reset-relevant register addresses (gtx_npu_core.cc:80-109) -----
NSPR_THREAD_MASK: int = 0x400    # 16-bit mask — defaults to 0xFFFF (all SPUs)
NSPR_SHARED_MASK: int = 0x401
NSPR_TYPE: int = 0x402           # 0=FP8, 1=FP16 (default), 2=INT8
NSPR_OP_MODE: int = 0x403
NSPR_CLEAR: int = 0x700
NSPR_SDLE_STATUS: int = 0x780
NSPR_SMU_DEBUG: int = 0x781
NSPR_CREDIT_COUNT: int = 0x782

# ----- Loop-control GSPR addresses (used by SPR router in plan 02) -----
# Provided here for cross-module use; addresses 0x100..0x105.
GSPR_STARTP: int = 0x100
GSPR_ENDP: int = 0x101
GSPR_STARTS: int = 0x102
GSPR_ENDS: int = 0x103
GSPR_STARTT: int = 0x104
GSPR_ENDT: int = 0x105

# ----- ISS funct7 (custom0) -- DMA section, P3 -----
GTX_ISS_F7_DMA_TPOSE: int = 0x38      # tpose (transpose)
GTX_ISS_F7_DMA_FILL: int = 0x39       # fill
GTX_ISS_F7_DMA_LD_ST: int = 0x40      # firmware DMA load/store/copy
GTX_ISS_F7_DMA_3D: int = 0x41         # SVR + 3D variants (load_svr/store_svr/load_3d/store_3d)
GTX_ISS_F7_DMA_MCAST_S2L: int = 0x42  # disasm-only stub in P3
GTX_ISS_F7_DMA_LD_SVR_L1: int = 0x43  # load_svr_l1 alias
GTX_ISS_F7_DMA_MCAST_GS: int = 0x44   # disasm-only stub (mcast_g2s/mcast_s2s/copy_mem share funct7)
GTX_ISS_F7_DMA_ST_SVR_L1: int = 0x45  # store_svr_l1 alias
GTX_ISS_F7_CREDIT_LD: int = 0x50      # credit_ld -- per-NEST/SPU counter inc/dec (P8 NEG fix)
GTX_ISS_F7_CREDIT_ST: int = 0x51      # credit_st -- per-NEST/SPU counter inc/dec (P8 NEG fix)
GTX_ISS_F7_CREDIT_LD_CHK: int = 0x52  # credit_ld_chk -- flush trigger when is_sloop (P8 MTDMA-01)
GTX_ISS_F7_CREDIT_ST_CHK: int = 0x53  # credit_st_chk -- flush trigger when is_sloop

# ----- MM funct3 (custom0 funct7=0x00 MM-family, funct7=0x01 MMC-family) -----
# Verified against vendor/gtx_cpp_reference/gtx/gtx_npu_disasm.inc:39-50
# and gtx_npu_mm.cc:357-376 dispatch switch.
# NOTE: funct3=4..6 are reserved; gtx_npu_mm.cc:373-376 falls through to basic mm.
GTX_F3_MM_S: int = 0      # mm_s / mmc_s -- FP32 result to ADDRC
GTX_F3_MM_O: int = 1      # mm_o / mmc_o -- scalar sum(A) to L0 BE + mxe_accum
GTX_F3_MM:   int = 2      # mm   / mmc   -- basic GEMM, ADDRC FP32 bias staging
GTX_F3_MM_V: int = 3      # mm_v / mmc_v -- scalar dot(A,B) to L0 LE + mxe_accum
GTX_F3_MM_T: int = 7      # mm_t / mmc_t -- transposed C^T to ADDRR (NxM layout!)


# =========================================================================
# Phase 5: VEC funct7 constants
# Source: vendor/gtx_cpp_reference/gtx/gtx_npu_disasm.inc:67-142
# =========================================================================
GTX_F7_VEC_SASMD: int = 0x10        # SASMD scalar arith (add/sub/mul/div x VS/IS)
GTX_F7_VEC_FMADD: int = 0x11        # fmadd_vss / fmadd_iss (P5 stub; lower priority)
GTX_F7_VEC_MINMAX: int = 0x13       # max_vs / min_vs / max_is / min_is (disasm.inc:80-84)
GTX_F7_VEC_ARITH: int = 0x18        # SASMD vector arith (add/sub/mul/div x VV/II)
# AUTHORITATIVE CORRECTION (Plan 05-02 deviation Rule 1):
# Plan 01 seeded GTX_F7_VEC_DOT_SUM=0x13 from a draft note; vendor
# `gtx_npu.h:308` defines GTX_ISS_F7_DOT_SUM = 0b0011010 = 0x1A. funct7=0x13
# is actually MIN/MAX scalar arith (disasm.inc:80-84). DOT/SUM lives at
# 0x1A with `dot_vvs` at funct3=0 and `sum_vs` (vsum) at funct3=1
# (disasm.inc:101-104; firmware_vec_op.cc:632-637).
GTX_F7_VEC_DOT_SUM: int = 0x1A      # dot funct3=0, vsum funct3=1 (gtx_npu_vec.cc:632-637)
GTX_F7_VEC_FMADD_VV: int = 0x19     # vector fmadd (P5 stub; lower priority)
GTX_F7_VEC_MATH: int = 0x1C         # sqrt/exp/log
GTX_F7_VEC_SIGN: int = 0x1D         # abs/neg/sgn/step
GTX_F7_VEC_ROUND: int = 0x1E        # ceil/trunc/floor/rne
GTX_F7_VEC_CLAMP: int = 0x1F        # clamp_min_v/clamp_max_v/accum_v/arange_v + bitwise

# =========================================================================
# Phase 5: format_cvt funct7 constants (RESEARCH Adjustment 1: include FP64<->FP16)
# Source: vendor/gtx_cpp_reference/gtx/gtx_npu_disasm.inc:144-148
# =========================================================================
GTX_F7_SCVT_QH: int = 0x20          # FP16<->FP8  (sub_op&1 selects direction)
GTX_F7_SCVT_IH: int = 0x21          # FP16<->INT8 (sub_op&1 selects direction)
GTX_F7_SCVT_HN: int = 0x22          # INT32->FP16 normalize (1-direction only)
GTX_F7_FCVT_SH: int = 0x24          # FP16<->FP32
GTX_F7_FCVT_DH: int = 0x25          # FP16<->FP64

# =========================================================================
# Phase 5: ACT funct7 constants (8 ISS activations split across 5 funct7 values)
# Source: vendor/gtx_cpp_reference/gtx/gtx_npu_disasm.inc:152-157
# =========================================================================
GTX_F7_ACT_PRELU: int = 0x28        # prelu funct3=3 / prelu_i funct3=7
GTX_F7_ACT_GELU: int = 0x2A         # gelu funct3=0 / gelu_i funct3=4
GTX_F7_ACT_TANH: int = 0x2C         # tanh funct3=0 / tanh_i funct3=4
GTX_F7_ACT_SIGM: int = 0x2D         # sigm funct3=0 / sigm_i funct3=4
GTX_F7_ACT_SOFTMAX: int = 0x2F      # esum funct3=1, softmax funct3=2; _imm at funct3=5/6

# =========================================================================
# Phase 5: POOL funct7 constants
# Source: vendor/gtx_cpp_reference/gtx/gtx_npu_disasm.inc:160-161
# =========================================================================
GTX_F7_POOL_MAX: int = 0x30         # pool_m
GTX_F7_POOL_AVG: int = 0x31         # pool_a

# =========================================================================
# Phase 5: GTX_ACT_* enum values (op_id passed to firmware_act + act_core)
# Source: vendor/gtx_cpp_reference/gtx/gtx_npu.h:371-377 (verified verbatim)
# =========================================================================
GTX_ACT_RELU: int = 0
GTX_ACT_TANH: int = 1
GTX_ACT_SOFTMAX: int = 2
GTX_ACT_GELU: int = 3
GTX_ACT_SIGMOID: int = 4
GTX_ACT_PRELU: int = 5
GTX_ACT_ESUM: int = 6

# Set of activation op_ids whose direction is REVERSED (read ADDRR, write ADDRA).
# Engine-internal CONSISTENCY CHECK only. The @handler-entry `is_reversed` literal
# is the source-of-truth (CONTEXT D-06). Do NOT use this as a routing primary.
ACT_OPS_REVERSED: frozenset = frozenset({GTX_ACT_TANH, GTX_ACT_GELU,
                                          GTX_ACT_SIGMOID, GTX_ACT_PRELU})

# =========================================================================
# Phase 5: GTX_VEC_* enum values (vec_op passed to vec_engine + vec_core)
# Source: vendor/gtx_cpp_reference/gtx/gtx_npu.h:382-405 (verified verbatim)
# Plan-stage draft used 0..9; vendor lock-in is 0..23 with full op list.
# =========================================================================
GTX_VEC_ADD: int = 0
GTX_VEC_SUB: int = 1
GTX_VEC_MUL: int = 2
GTX_VEC_DIV: int = 3
GTX_VEC_FMADD: int = 4
GTX_VEC_VSUM: int = 5
GTX_VEC_VEXP: int = 6
GTX_VEC_VSQRT: int = 7
GTX_VEC_VLN: int = 8
GTX_VEC_VABS: int = 9
GTX_VEC_VNEG: int = 10
GTX_VEC_MAX: int = 11
GTX_VEC_MIN: int = 12
GTX_VEC_SIGN: int = 13
GTX_VEC_STEP: int = 14
GTX_VEC_CEIL: int = 15
GTX_VEC_TRUNC: int = 16
GTX_VEC_FLOOR: int = 17
GTX_VEC_RNE: int = 18
GTX_VEC_ACCUM: int = 19
GTX_VEC_CLAMP_MAX: int = 20
GTX_VEC_CLAMP_MIN: int = 21
GTX_VEC_ARANGE: int = 22
GTX_VEC_DOT: int = 23