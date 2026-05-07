# P6 VRF-03: sum (vsum reduction) hand-written firmware (Plan 03 Task 2b).
#
# NOT a vendor 1:1 .c-build (RESEARCH finding #3 hand-written .S strategy).
# Per-op encoding:
#   funct7=GTX_F7_VEC_DOT_SUM=0x1A (vec reduction family), funct3=1
#   selects vsum (sum_vs at funct3=1 per disasm.inc:101-104,
#   firmware_vec_op.cc:632-637).
#
# Reduction: reads ADDRA, writes scalar to ADDRR.
# Default L1 = zero-init -> vsum(zeros) = 0.

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

    # vsum reduction: funct7=0x1A funct3=1
    # rs1 non-zero (Pitfall F).
    li    x1, 0x10
    .insn r 0x0b, 1, 0x1a, x0, x1, x0

    # WJOIN
    .insn r 0x2b, 0b101, 0, x0, x0, x0

    j .
