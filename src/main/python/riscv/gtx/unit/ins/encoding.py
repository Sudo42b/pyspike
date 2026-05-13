"""GTX RoCC instruction encoding constants.

Direct port of vendor/gtx_cpp_reference/gtx/gtx_npu.h

Phase 2 (D-11): full funct7 set + custom1 funct3 + opcode constants.
"""
from enum import Enum

# ----- RoCC custom opcodes -----
CUSTOM0 = 0x0b # custom-0 0b001011
CUSTOM1 = 0x2b # custom-1 0b101011

# ----- funct7 (custom0) -------
GTX_F7_WRSPR: int = 0x00        # 0b1001001
GTX_F7_RDSPR: int = 0x01        # 0b1001010
GTX_F7_WSPLIT: int = 0x02
GTX_F7_WJOIN: int = 0x03        # 0b1001011
GTX_F7_DISPATCH_MM: int = 0x04
GTX_F7_DISPATCH_VEC: int = 0x05
GTX_F7_DISPATCH_ACT: int = 0x06
GTX_F7_DISPATCH_DMA: int = 0x07

# ----- ISS-full funct7 (custom0) -- gtx_npu.h -----
GTX_ISS_F7_RDSPR_ISS: int = 0x48        # 0b1001000
GTX_ISS_F7_WRSPR_ISS: int = 0x49        # 0b1001001
GTX_ISS_F7_OPSET: int = 0x4A            # 0b1001010
GTX_ISS_F7_CPSVR: int = 0x4B            # 0b1001011
GTX_ISS_F7_MVSVR: int = 0x4C            # 0b1001100
GTX_ISS_F7_DEBUG_WR: int = 0x7D         # 0x7D
GTX_ISS_F7_DEBUG_RD: int = 0x7E         # 0x7E

# ----- DISPATCH funct7 sub-opcode -----
# ============================================================================
# DISPATCH funct7 sub-opcodes — Simplified encoding (gem5 gtx_insn.h)
#
# When funct3=DISPATCH, funct7 selects the operation category:
# ============================================================================
GTX_OP_MM: int = 0
GTX_OP_VECTOR: int = 1
GTX_OP_ACTIVATION: int = 2
GTX_OP_DMA: int = 3

# ============================================================================
# DISPATCH funct7 sub-opcodes — Full ISS encoding (from GTX_extension.h)
#
# These are the funct7 values from the ISS for fine-grained opcode dispatch.
# Spike supports BOTH encodings: simplified (0-3) from gem5 and full ISS.
# ============================================================================
GTX_ISS_F7_MM: int = 0b0000000        # MM variants/ mm.s, mm.o, mm.v, mm, mm.t
GTX_ISS_F7_MMC: int = 0b0000001       # MMC (accumulate) / mmc.s, mmc.o, mmc.v, mmc, mmc.t
GTX_ISS_F7_IM2COL_N: int = 0b0001000  # IM2COL normal 
GTX_ISS_F7_IM2COL_D: int = 0b0001001  # IM2COL depthwise
# Scalar Calulations
GTX_ISS_F7_SCALAR_ARITH:int = 0b0010000 # ADD/SUB/MUL/DIV scalar add.vs, sub.vs, mul.vs, div.vs, add.is, sub.is, mul.is, div.is
GTX_ISS_F7_FMADD_S:int = 0b0010001     # Scalar FMADD fmadd.vss, fmadd.iss
GTX_ISS_F7_MINMAX_S:int = 0b0010011    # Scalar MIN/MAX min.vs, max.vs, max.is, min.is
GTX_ISS_F7_DOT_SUM:int = 0b0011010     # DOT/SUM
# Vector Calculations
GTX_ISS_F7_VEC_ARITH:int = 0b0011000   # ADD/SUB/MUL/DIV vector add.vv, sub.vv, mul.vv, div.vv, add.ii, sub.ii, mul.ii, div.ii
GTX_ISS_F7_FMADD_V:int = 0b0011001     # Vector FMADD fmadd.vvv, fmadd.iii
GTX_ISS_F7_MATH_V:int = 0b0011100      # SQRT/EXP/LN, sqrt.v, exp.v, ln.v
GTX_ISS_F7_SIGN_V:int = 0b0011101      # ABS/NEG/SIGN/STEP abs.v, neg.v, sign.v, step.v
GTX_ISS_F7_ROUND_V:int = 0b0011110     # CEIL/TRUNC/FLOOR/RNE, ceil.v, trunc.v, floor.v, rne.v, ceil.i, trunc.i, floor.i, rne.i
GTX_ISS_F7_CLAMP_V:int = 0b0011111     # CLAMP/ACCUM/ARANGE/LOGIC, clamp.min, clamp.max, accum, arange, and.ii, or.ii, not.i, shift.i
# Format Conversion
GTX_ISS_F7_SCVT_QH:int = 0b0100000     # FP8↔FP16 scvt.qh, scvt.hq
GTX_ISS_F7_SCVT_IH:int = 0b0100001     # INT8↔FP16, scvt.ih, scvt.hi
GTX_ISS_F7_SCVT_HN:int = 0b0100010     # INT32→FP16, scvt.hn
GTX_ISS_F7_FCVT_SH:int = 0b0100100     # FP32↔FP16, fcvt.sh, fcvt.hs
GTX_ISS_F7_FCVT_DH:int = 0b0100101     # FP64↔FP16, fcvt.dh, fcvt.hd
# Activation functions
GTX_ISS_F7_PRELU:int = 0b0101000       # PReLU, prelu, prelu.i
GTX_ISS_F7_GELU:int = 0b0101010        # GeLU gelu, gelu.i
GTX_ISS_F7_TANH:int = 0b0101100        # Tanh tanh, tanh.i
GTX_ISS_F7_SIGM:int = 0b0101101        # Sigmoid sigmoid, sigm.i
# Softmax
GTX_ISS_F7_SOFTMAX:int = 0b0101111     # Softmax/ESUM esum, softmax, esum.i, softmax.i
# Pooling 
GTX_ISS_F7_POOL_MAX:int = 0b0110000    # Max pooling, pool.m
GTX_ISS_F7_POOL_AVG:int = 0b0110001    # Average pooling, pool.a
# Memory Operation
GTX_ISS_F7_TPOSE:int = 0b0111000       # Transpose, tpose
GTX_ISS_F7_FILL:int = 0b0111001        # Memory fill, fill
# DMA
GTX_ISS_F7_DMA_LOAD:int = 0b1000000    # DMA LOAD/STORE/COPY, load, store, copy
GTX_ISS_F7_DMA_3D:int = 0b1000001      # DMA 3D / SVR, load.svr, store.svr
GTX_ISS_F7_MCAST_S2L:int = 0b1000010   # Multicast SRAM→L1, mcast.s2l
GTX_ISS_F7_MCAST_G2S:int = 0b1000100   # Multicast/Copy, mcast.g2s, mcast.s2s, copy.mem
# SPR
# ----- ISS funct7 (custom0) -- DMA section-----
GTX_ISS_F7_CREDIT_LD: int = 0x50      # credit.ld -- per-NEST/SPU counter inc/dec 0b1010000
GTX_ISS_F7_CREDIT_ST: int = 0x51      # credit.st -- per-NEST/SPU counter inc/dec 0b1010001
GTX_ISS_F7_CREDIT_LD_CHK: int = 0x52  # credit.ld.chk -- flush trigger when is_sloop 0b1010010
GTX_ISS_F7_CREDIT_ST_CHK: int = 0x53  # credit.st.chk -- flush trigger when is_sloop 0b1010011
GTX_ISS_F7_DMA_LD_ST: int = 0x40      # firmware DMA load/store/copy
GTX_ISS_F7_DMA_3D: int = 0x41         # SVR + 3D variants (load_svr/store_svr/load_3d/store_3d)
GTX_ISS_F7_DMA_MCAST_S2L: int = 0x42  # disasm-only stub in P3
GTX_ISS_F7_DMA_LD_SVR_L1: int = 0x43  # load_svr_l1 alias
GTX_ISS_F7_DMA_MCAST_GS: int = 0x44   # disasm-only stub (mcast_g2s/mcast_s2s/copy_mem share funct7)
GTX_ISS_F7_DMA_ST_SVR_L1: int = 0x45  # store_svr_l1 alias
# MicroCode
GTX_ISS_F7_MEXEC:int = 0b1110000       # Macro execute
GTX_ISS_F7_MBAR:int = 0b1110100        # Memory barrier (NOP)
GTX_ISS_F7_MSYNC:int = 0b1110101       # Memory sync
GTX_ISS_F7_EOM:int = 0b1110111         # End of model
GTX_ISS_F7_BAR:int = 0b1111000         # Barrier
# 
GTX_ISS_F7_WAIT:int = 0b1111001        # Wait
GTX_ISS_F7_INTR:int = 0b1111011        # Interrupt
GTX_ISS_F7_FLUSH:int = 0b1111100       # Flush
GTX_ISS_F7_HALT:int = 0b1111111        # Halt

# ── LOAD_SVR / STORE_SVR (L0↔L1 transfer) ─────────────────────────────
GTX_ISS_F7_LOAD_SVR:int = 0b1000011    # 0x43 — L1→L0 (32B)
GTX_ISS_F7_STORE_SVR:int = 0b1000101   # 0x45 — L0→L1 (32B)

#  ── L0-based _IMM operation groups ─────────────────────────
GTX_ISS_F7_SCALAR_IMM:int = 0b1010100  # 0x54 — scalar _IS arith
GTX_ISS_F7_VECTOR_IMM:int = 0b1011000  # 0x58 — vector _II arith
GTX_ISS_F7_VFUNC_IMM:int = 0b1011010   # 0x5A — L0 math functions
GTX_ISS_F7_BITWISE_IMM:int = 0b1011011 # 0x5B — L0 bitwise
GTX_ISS_F7_ACT_IMM:int = 0b1011100     # 0x5C — L0 activations
GTX_ISS_F7_SOFTMAX_IMM:int = 0b1011101 # 0x5D — L0 ESUM/SOFTMAX

# ============================================================================
# Activation sub-opcodes (passed via GSPR_GTX_OPCODE or encoded in operands)
# ============================================================================
GTX_ACT_RELU:int    = 0
GTX_ACT_TANH:int    = 1
GTX_ACT_SOFTMAX:int = 2
GTX_ACT_GELU:int    = 3
GTX_ACT_SIGMOID:int = 4
GTX_ACT_PRELU:int   = 5
GTX_ACT_ESUM:int    = 6

# ============================================================================
# Vector sub-opcodes (passed via GSPR_GTX_OPCODE)
# ============================================================================
GTX_VEC_ADD:int   = 0
GTX_VEC_SUB:int   = 1
GTX_VEC_MUL:int   = 2
GTX_VEC_DIV:int   = 3
GTX_VEC_FMADD:int = 4
GTX_VEC_VSUM:int  = 5
GTX_VEC_VEXP:int  = 6
GTX_VEC_VSQRT:int = 7
GTX_VEC_VLN:int   = 8
GTX_VEC_VABS:int  = 9
GTX_VEC_VNEG:int  = 10
GTX_VEC_MAX:int   = 11
GTX_VEC_MIN:int   = 12
GTX_VEC_SIGN:int      = 13
GTX_VEC_STEP:int      = 14
GTX_VEC_CEIL:int      = 15
GTX_VEC_TRUNC:int     = 16
GTX_VEC_FLOOR:int     = 17
GTX_VEC_RNE:int       = 18
GTX_VEC_ACCUM:int     = 19
GTX_VEC_CLAMP_MAX:int = 20
GTX_VEC_CLAMP_MIN:int = 21
GTX_VEC_ARANGE:int    = 22
GTX_VEC_DOT:int       = 23

# ============================================================================
# MM variant sub-opcodes (via GSPR_GTX_OPCODE low bits)
#   [7]: has_bias (0=MM, 1=MMC)
#   [2:0]: variant (0=basic, 1=_S, 2=_O, 3=_V, 4=_T)
# ============================================================================
GTX_MM_BASIC = 0
GTX_MM_S     = 1  # FP32 output to L1
GTX_MM_O     = 2  # L0 output + mxe_accum
GTX_MM_V     = 3  # dot product → L0
GTX_MM_T     = 4  # transposed output

# ============================================================================
# _IMM sub-opcodes (via GSPR_GTX_OPCODE, within funct7 groups)
# ============================================================================
# Scalar _IS arith (funct7=0x54)
GTX_IMM_ADD   = 0
GTX_IMM_SUB   = 1
GTX_IMM_MUL   = 2
GTX_IMM_DIV   = 3
GTX_IMM_FMADD = 4
# Vector _II arith (funct7=0x58)
GTX_IMM_MAX   = 5
GTX_IMM_MIN   = 6
# L0 math functions (funct7=0x5A) — same numbering as VFUNC_IMM in ISS
GTX_IMM_SQRT  = 0
GTX_IMM_EXP   = 1
GTX_IMM_LN    = 2
GTX_IMM_ABS   = 3
GTX_IMM_NEG   = 4
GTX_IMM_SIGN  = 5
GTX_IMM_STEP  = 6
GTX_IMM_CEIL  = 7
GTX_IMM_TRUNC = 8
GTX_IMM_FLOOR = 9
GTX_IMM_RNE   = 10

# L0 bitwise (funct7=0x5B)
GTX_IMM_AND   = 0
GTX_IMM_OR    = 1
GTX_IMM_NOT   = 2
GTX_IMM_SHIFT = 3

# L0 activations (funct7=0x5C)
GTX_IMM_ACT_PRELU   = 0
GTX_IMM_ACT_GELU    = 1
GTX_IMM_ACT_TANH    = 2
GTX_IMM_ACT_SIGM    = 3
GTX_IMM_ACT_ESUM    = 4
GTX_IMM_ACT_SOFTMAX = 5

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

# ============================================================================
# custom1() — Warp control dispatch (custom-1 opcode 0x2b)
#
# Encoding: funct3 (bits[14:12]) selects the warp control variant:
#   funct3=000: START_T   funct3=001: END_T
#   funct3=010: START_S   funct3=011: END_S
#   funct3=100: SPLIT     funct3=101: JOIN
#   funct3=110: START_P   funct3=111: END_P
#
# In RoCC, bits[14:12] are {xd,xs1,xs2} flags, not funct3.
# We reconstruct funct3 from these bits and read registers directly.
# ============================================================================
WARP_F3_START_T: int = 0b000
WARP_F3_END_T: int = 0b001

WARP_F3_START_S: int = 0b010
WARP_F3_END_S: int = 0b011

WARP_F3_START_P: int = 0b110
WARP_F3_END_P: int = 0b111

WARP_F3_SPLIT: int = 0b100
WARP_F3_JOIN: int = 0b101
