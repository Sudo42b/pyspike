"""GTX RoCC instruction encoding constants.

Direct port of vendor/gtx_cpp_reference/gtx/gtx_npu.h — single source for all
funct7 / funct3 / sub-opcode constants consumed by the custom0 handlers.

Access namespace-qualified (``encoding.F7_OPSET``) — the ``GTX_`` prefix
is dropped since the module name already scopes these.
"""

# ----- RoCC custom opcodes -----
CUSTOM0 = 0x0b  # custom-0 0b001011
CUSTOM1 = 0x2b  # custom-1 0b101011

# ----- funct7 (custom0) -- vendor gtx_npu.h:266-277 -----
# Firmware path (collides with MM/MMC funct7=0x00/0x01 — protected by the
# rs1==0 guard in MX/matmul.py "Pitfall F"):
F7_WRSPR: int = 0x00        # firmware WRSPR (gtx_npu.h:266)
F7_RDSPR: int = 0x01        # firmware RDSPR (gtx_npu.h:267)
# ISS-full path (distinct funct7 — no collision):
F7_RDSPR_ISS: int = 0x48        # 0b1001000 (gtx_npu.h:276)
F7_WRSPR_ISS: int = 0x49        # 0b1001001 (gtx_npu.h:277)
F7_OPSET: int = 0x4A            # 0b1001010
F7_CPSVR: int = 0x4B            # 0b1001011
F7_MVSVR: int = 0x4C            # 0b1001100

F7_DISPATCH_MM: int = 0x04
F7_DISPATCH_VEC: int = 0x05
F7_DISPATCH_ACT: int = 0x06
F7_DISPATCH_DMA: int = 0x07

# ----- DISPATCH funct7 sub-opcode (gem5 simplified) -----
OP_MM: int = 0
OP_VECTOR: int = 1
OP_ACTIVATION: int = 2
OP_DMA: int = 3

# ----- Full ISS funct7 dispatch (gtx_npu.h) -----
# NOTE: MM (0b0000000) / MMC (0b0000001) intentionally alias F7_WRSPR /
# F7_RDSPR — vendor funct7 collision protected by the rs1==0 guard.
F7_IM2COL_N: int = 0b0001000
F7_IM2COL_D: int = 0b0001001
# Scalar calculations
F7_SCALAR_ARITH: int = 0b0010000
F7_FMADD_S: int = 0b0010001
F7_MINMAX_S: int = 0b0010011
F7_DOT_SUM: int = 0b0011010
# Vector calculations
F7_VEC_ARITH: int = 0b0011000
F7_FMADD_V: int = 0b0011001
F7_MATH_V: int = 0b0011100
F7_SIGN_V: int = 0b0011101
F7_ROUND_V: int = 0b0011110
F7_CLAMP_V: int = 0b0011111
# Format conversion
F7_SCVT_QH: int = 0b0100000
F7_SCVT_IH: int = 0b0100001
F7_SCVT_HN: int = 0b0100010
F7_FCVT_SH: int = 0b0100100
F7_FCVT_DH: int = 0b0100101
# Activation
F7_PRELU: int = 0b0101000
F7_GELU: int = 0b0101010
F7_TANH: int = 0b0101100
F7_SIGM: int = 0b0101101
F7_SOFTMAX: int = 0b0101111
# Pooling
F7_POOL_MAX: int = 0b0110000
F7_POOL_AVG: int = 0b0110001
# Memory operation
F7_TPOSE: int = 0b0111000
F7_FILL: int = 0b0111001
# DMA
F7_DMA_LOAD: int = 0b1000000
F7_MCAST_S2L: int = 0b1000010
F7_MCAST_G2S: int = 0b1000100
# SPR / DMA section funct7
F7_CREDIT_LD: int = 0x50      # credit.ld
F7_CREDIT_ST: int = 0x51      # credit.st
F7_CREDIT_LD_CHK: int = 0x52  # credit.ld.chk
F7_CREDIT_ST_CHK: int = 0x53  # credit.st.chk
F7_DMA_LD_ST: int = 0x40      # firmware DMA load/store/copy
F7_DMA_3D: int = 0x41         # SVR + 3D variants
F7_DMA_LD_SVR_L1: int = 0x43  # load.svr_l1 alias
F7_DMA_ST_SVR_L1: int = 0x45  # store.svr_l1 alias
# Microcode / sync
F7_MEXEC: int = 0b1110000
F7_MBAR: int = 0b1110100
F7_MSYNC: int = 0b1110101
F7_EOM: int = 0b1110111
F7_BAR: int = 0b1111000
F7_WAIT: int = 0b1111001
F7_INTR: int = 0b1111011
F7_FLUSH: int = 0b1111100
F7_HALT: int = 0b1111111

# LOAD_SVR / STORE_SVR (L0↔L1 transfer)
F7_LOAD_SVR: int = 0b1000011    # 0x43 — L1→L0 (32B)
F7_STORE_SVR: int = 0b1000101   # 0x45 — L0→L1 (32B)

# L0-based _IMM operation groups
F7_SCALAR_IMM: int = 0b1010100  # 0x54
F7_VECTOR_IMM: int = 0b1011000  # 0x58
F7_VFUNC_IMM: int = 0b1011010   # 0x5A
F7_BITWISE_IMM: int = 0b1011011  # 0x5B
F7_ACT_IMM: int = 0b1011100     # 0x5C
F7_SOFTMAX_IMM: int = 0b1011101  # 0x5D

# ----- Activation sub-opcodes -----
ACT_RELU: int = 0
ACT_TANH: int = 1
ACT_SOFTMAX: int = 2
ACT_GELU: int = 3
ACT_SIGMOID: int = 4
ACT_PRELU: int = 5
ACT_ESUM: int = 6

# Activations that swap LSPR direction (rd=ADDRR, wr=ADDRA).
ACT_OPS_REVERSED: frozenset = frozenset({
    ACT_PRELU, ACT_GELU, ACT_TANH, ACT_SIGMOID,
})

# ----- Vector sub-opcodes (via GSPR_GTX_OPCODE) -----
VEC_ADD: int = 0
VEC_SUB: int = 1
VEC_MUL: int = 2
VEC_DIV: int = 3
VEC_FMADD: int = 4
VEC_VSUM: int = 5
VEC_VEXP: int = 6
VEC_VSQRT: int = 7
VEC_VLN: int = 8
VEC_VABS: int = 9
VEC_VNEG: int = 10
VEC_MAX: int = 11
VEC_MIN: int = 12
VEC_SIGN: int = 13
VEC_STEP: int = 14
VEC_CEIL: int = 15
VEC_TRUNC: int = 16
VEC_FLOOR: int = 17
VEC_RNE: int = 18
VEC_ACCUM: int = 19
VEC_CLAMP_MAX: int = 20
VEC_CLAMP_MIN: int = 21
VEC_ARANGE: int = 22
VEC_DOT: int = 23

# ----- MM variant sub-opcodes -----
MM_BASIC = 0
MM_S = 1
MM_O = 2
MM_V = 3
MM_T = 4

# ----- _IMM sub-opcodes -----
IMM_ADD = 0
IMM_SUB = 1
IMM_MUL = 2
IMM_DIV = 3
IMM_FMADD = 4
IMM_MAX = 5
IMM_MIN = 6
IMM_SQRT = 0
IMM_EXP = 1
IMM_LN = 2
IMM_ABS = 3
IMM_NEG = 4
IMM_SIGN = 5
IMM_STEP = 6
IMM_CEIL = 7
IMM_TRUNC = 8
IMM_FLOOR = 9
IMM_RNE = 10
IMM_AND = 0
IMM_OR = 1
IMM_NOT = 2
IMM_SHIFT = 3
IMM_ACT_PRELU = 0
IMM_ACT_GELU = 1
IMM_ACT_TANH = 2
IMM_ACT_SIGM = 3
IMM_ACT_ESUM = 4
IMM_ACT_SOFTMAX = 5

# ----- MM funct3 (custom0 funct7=0x00 MM / 0x01 MMC) -----
F3_MM_S: int = 0      # mm.s / mmc.s -- FP32 result to ADDRC
F3_MM_O: int = 1      # mm.o / mmc.o -- scalar sum(A) to L0 BE + mxe_accum
F3_MM: int = 2        # mm   / mmc   -- basic GEMM
F3_MM_V: int = 3      # mm.v / mmc.v -- scalar dot(A,B) to L0 LE + mxe_accum
F3_MM_T: int = 7      # mm.t / mmc.t -- transposed C^T to ADDRR

# ----- VEC funct7 -----
F7_VEC_SASMD: int = 0x10
F7_VEC_FMADD: int = 0x11
F7_VEC_MINMAX: int = 0x13
F7_VEC_ARITH: int = 0x18
F7_VEC_DOT_SUM: int = 0x1A
F7_VEC_FMADD_VV: int = 0x19
F7_VEC_MATH: int = 0x1C
F7_VEC_SIGN: int = 0x1D
F7_VEC_ROUND: int = 0x1E
F7_VEC_CLAMP: int = 0x1F

# ----- format_cvt funct7 -----
F7_SCVT_QH: int = 0x20
F7_SCVT_IH: int = 0x21
F7_SCVT_HN: int = 0x22
F7_FCVT_SH: int = 0x24
F7_FCVT_DH: int = 0x25

# ----- ACT funct7 -----
F7_ACT_PRELU: int = 0x28
F7_ACT_GELU: int = 0x2A
F7_ACT_TANH: int = 0x2C
F7_ACT_SIGM: int = 0x2D
F7_ACT_SOFTMAX: int = 0x2F

# ----- POOL funct7 -----
F7_POOL_MAX: int = 0x30
F7_POOL_AVG: int = 0x31

# ----- custom1 warp funct3 -----
WARP_F3_START_T: int = 0b000
WARP_F3_END_T: int = 0b001
WARP_F3_START_S: int = 0b010
WARP_F3_END_S: int = 0b011
WARP_F3_SPLIT: int = 0b100
WARP_F3_JOIN: int = 0b101
WARP_F3_START_P: int = 0b110
WARP_F3_END_P: int = 0b111
