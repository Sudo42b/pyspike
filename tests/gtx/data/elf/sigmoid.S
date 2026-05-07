# P6 VRF-03: sigmoid hand-written firmware (Plan 03 Task 2a).
#
# NOT a vendor 1:1 .c-build (RESEARCH finding #3 hand-written .S strategy).
# Per-op reversed dispatch: funct7=GTX_F7_ACT_SIGM=0x2D, funct3=0
# (sigm: reads ADDRR, writes ADDRA -- reversed direction).
#
# No GSPR_GTX_OPCODE staging needed (per-op funct7 is the dispatch primary).
# Default L1 = zero-init -> output = sigmoid(0) = 0.5; but encoded form is
# whatever the reversed-write produces. Golden source = vendor _ref.txt
# (single-row 32-byte truncation; P4/P5 precedent).

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

    # sigmoid (ISS direct reversed): funct7=0x2D funct3=0
    # rs1 non-zero (Pitfall F).
    li    x1, 0x10
    .insn r 0x0b, 0, 0x2d, x0, x1, x0

    # WJOIN
    .insn r 0x2b, 0b101, 0, x0, x0, x0

    j .
