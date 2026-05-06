# Minimal MM firmware for P4 strict mode regression (D-09 fallback).
#
# Behavior:
#   1. addi sp, sp, -16            ; sp init = 0x80100000 (set in GtxNpu.reset)
#   2. WRSPR LSPR_SPM_ADDRA = 0    ; A region at L1[0..]    (ISS-full funct7=0x49)
#   3. WRSPR LSPR_SPM_ADDRB = 0x200; B region at L1[0x200..]
#   4. WRSPR LSPR_SPM_ADDRR = 0x400; result at L1[0x400..]
#   5. li x1, 0x0004_0000_0004_0004 ; packed rs1 = colB<<48 | colA<<16 | rowA = 4x4x4
#   6. .insn r 0x0b, 2, 0x00, x10, x1, x0  ; custom0 funct7=0x00 funct3=2 (mm)
#                                          ; rs1=x1 (!=0) -> routes to MM (NOT WRSPR)
#   7. .insn r 0x2b, 0b101, 0, x0, x0, x0  ; WJOIN (raises SystemExit(0))
#   8. j .                                  ; safety loop
#
# Notes:
# - Test-fixture L1 is zero-initialized (Plan 05 subprocess L1=0 path; non-trivial
#   operand staging deferred to P6). Result = mm(zeros @ zeros) = zeros.
# - Uses ISS-full WRSPR (funct7=0x49) not gem5-simplified (funct7=0x00) to
#   avoid reentering the funct7=0x00 dispatch path before MM @handler is wired.

.section .text._start, "ax"
.global _start
_start:
    addi  sp, sp, -16

    # WRSPR LSPR_SPM_ADDRA (0x900) = 0
    li    x1, 0x900
    li    x2, 0
    .insn r 0x0b, 0, 0x49, x0, x1, x2

    # WRSPR LSPR_SPM_ADDRB (0x901) = 0x200
    li    x1, 0x901
    li    x2, 0x200
    .insn r 0x0b, 0, 0x49, x0, x1, x2

    # WRSPR LSPR_SPM_ADDRR (0x903) = 0x400
    li    x1, 0x903
    li    x2, 0x400
    .insn r 0x0b, 0, 0x49, x0, x1, x2

    # rs1 = (colB=4 << 48) | (colA=4 << 16) | (rowA=4) = 0x0004_0000_0004_0004
    li    x1, 0x0004000000040004

    # custom0 funct7=0x00 funct3=2 (mm). rs1=x1 (nonzero) -> routes to MM.
    .insn r 0x0b, 2, 0x00, x10, x1, x0

    # WJOIN -- ends simulation cleanly
    .insn r 0x2b, 0b101, 0, x0, x0, x0

    # Safety loop (never reached)
    j .
