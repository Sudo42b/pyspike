# P6 VRF-03: mul_vv hand-written firmware (Plan 03 Task 2b).
#
# NOT a vendor 1:1 .c-build (RESEARCH finding #3 hand-written .S strategy).
# Per-op encoding:
#   funct7=GTX_F7_VEC_ARITH=0x18 (vector SASMD VV arith), funct3=2,
#   sub_op=GTX_VEC_MUL=2 (per encoding.py).
#
# Default L1 = zero-init -> mul(zeros, zeros) = zeros.

.section .text._start, "ax"
.global _start
_start:
    addi  sp, sp, -16

    # WRSPR LSPR_SPM_ADDRA (0x900) = 0
    li    x1, 0x900
    li    x2, 0
    .insn r 0x0b, 0, 0x49, x0, x1, x2

    # WRSPR LSPR_SPM_ADDRB (0x901) = 0x100
    li    x1, 0x901
    li    x2, 0x100
    .insn r 0x0b, 0, 0x49, x0, x1, x2

    # WRSPR LSPR_SPM_ADDRR (0x903) = 0x200
    li    x1, 0x903
    li    x2, 0x200
    .insn r 0x0b, 0, 0x49, x0, x1, x2

    # vec arith VV mul: funct7=0x18 funct3=2 (GTX_VEC_MUL=2)
    # rs1 non-zero (Pitfall F).
    li    x1, 0x10
    .insn r 0x0b, 2, 0x18, x0, x1, x0

    # WJOIN
    .insn r 0x2b, 0b101, 0, x0, x0, x0

    j .
