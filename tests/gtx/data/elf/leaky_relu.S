# P6 VRF-03: leaky_relu hand-written firmware (Plan 03 Task 2a).
#
# NOT a vendor 1:1 .c-build (RESEARCH finding #3 hand-written .S strategy).
# Per-op encoding:
#   funct7=GTX_F7_DISPATCH_ACT=0x06 (firmware forward), funct3=0,
#   sub_op=GTX_ACT_PRELU=5 (staged via GSPR_GTX_OPCODE=0x004; PRELU covers
#   leaky-ReLU as the parametric variant per vendor n1s16_leaky_relu.c).
#
# Default L1 = zero-init -> leaky_relu(0) = 0.

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

    # WRSPR GSPR_GTX_OPCODE (0x004) = sub_op = GTX_ACT_PRELU = 5
    li    x1, 0x004
    li    x2, 5
    .insn r 0x0b, 0, 0x49, x0, x1, x2

    # firmware DISPATCH_ACT funct7=0x06 funct3=0 (forward)
    # rs1 non-zero (Pitfall F).
    li    x1, 0x10
    .insn r 0x0b, 0, 0x06, x0, x1, x0

    # WJOIN
    .insn r 0x2b, 0b101, 0, x0, x0, x0

    j .
