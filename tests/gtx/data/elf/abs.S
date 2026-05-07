# P6 VRF-03: abs (vec unary) hand-written firmware (Plan 03 Task 2b).
#
# NOT a vendor 1:1 .c-build (RESEARCH finding #3 hand-written .S strategy).
# Per-op encoding:
#   funct7=GTX_F7_VEC_SIGN=0x1D (sign/abs/neg/step family), funct3=0,
#   sub_op=GTX_VEC_VABS=9 packed into rs1 (per vec_core abs_v dispatch).
#
# Default L1 = zero-init -> abs(zeros) = zeros.

.section .text._start, "ax"
.global _start
_start:
    addi  sp, sp, -16

    # WRSPR LSPR_SPM_ADDRA (0x900) = 0
    li    x1, 0x900
    li    x2, 0
    .insn r 0x0b, 0, 0x49, x0, x1, x2

    # WRSPR LSPR_SPM_ADDRR (0x903) = 0x100
    li    x1, 0x903
    li    x2, 0x100
    .insn r 0x0b, 0, 0x49, x0, x1, x2

    # vec sign-family abs: funct7=0x1D funct3=0
    # rs1 non-zero (Pitfall F); GTX_VEC_VABS=9 packed in low bits.
    li    x1, 9
    .insn r 0x0b, 0, 0x1d, x0, x1, x0

    # WJOIN
    .insn r 0x2b, 0b101, 0, x0, x0, x0

    j .
