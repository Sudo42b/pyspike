# P6 VRF-03: add_vv hand-written firmware (Plan 03 Task 2b).
#
# NOT a vendor 1:1 .c-build (RESEARCH finding #3 hand-written .S strategy).
# Per-op encoding:
#   funct7=GTX_F7_VEC_ARITH=0x18 (vector SASMD VV arith), funct3=0,
#   sub_op=GTX_VEC_ADD=0 (encoded into rs1 packing per vec_engine).
#
# Default L1 = zero-init -> add(zeros, zeros) = zeros.

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

    # vec arith VV add: funct7=0x18 funct3=0 (GTX_VEC_ADD=0)
    # rs1 non-zero (Pitfall F).
    li    x1, 0x10
    .insn r 0x0b, 0, 0x18, x0, x1, x0

    # WJOIN
    .insn r 0x2b, 0b101, 0, x0, x0, x0

    j .
