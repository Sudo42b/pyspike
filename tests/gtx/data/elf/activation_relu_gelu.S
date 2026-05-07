# Minimal ACT firmware for P5 strict-mode regression (D-04 + Plan 06 fixture).
#
# Behavior (zero-init L1 strategy mirroring P4 mm_basic.S):
#   1. addi sp, sp, -16                      ; sp init = 0x80100000 (set in GtxNpu.reset)
#   2. WRSPR LSPR_SPM_ADDRA = 0              ; A region at L1[0..]   (ISS-full funct7=0x49)
#   3. WRSPR LSPR_SPM_ADDRR = 0x100          ; R region at L1[0x100..]
#   4. custom0 funct7=0x06 funct3=0          ; firmware DISPATCH_ACT, sub_op=GTX_ACT_RELU
#                                              -- forward: reads ADDRA (zero), writes ADDRR
#   5. custom0 funct7=0x2A funct3=0          ; gelu (ISS direct): reversed
#                                              -- reversed: reads ADDRR, writes ADDRA
#   6. WJOIN custom1                         ; raises SystemExit(0)
#   7. j .                                   ; safety loop
#
# Notes:
# - Uses ISS-full WRSPR (funct7=0x49) not gem5-simplified (funct7=0x00) -- mirrors
#   mm_basic.S P4 04-01 D-3. Avoids reentering funct7=0x00 dispatch path.
# - L1 zero-init (P4 mm_basic.S strategy + Blocker 1 Option B).
#   Inputs to RELU and GELU are all zero.
# - RELU(0) = 0 (forward writes 0 to ADDRR; ADDRR was already 0).
# - GELU(0) = 0 (reversed writes 0 to ADDRA; ADDRA was already 0).
# - Net mutation = nothing visible. Plan 06 strict compare uses zero-byte golden.
# - Non-trivial operand staging deferred to P6 (mirror P4 04-01 Blocker 1 Option B).

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

    # rs1 must be nonzero so funct7=0x06 routes to firmware DISPATCH_ACT
    # (mirrors mm_basic.S Pitfall F nonzero-rs1 guard for funct7=0x00 collision)
    li    x1, 0x10
    # custom0 funct7=0x06 funct3=0 -- firmware DISPATCH_ACT, sub_op=GTX_ACT_RELU(0)
    .insn r 0x0b, 0, 0x06, x0, x1, x0

    # gelu (ISS direct): funct7=0x2A funct3=0 (reversed: ADDRR -> ADDRA)
    li    x1, 0x10
    .insn r 0x0b, 0, 0x2a, x0, x1, x0

    # WJOIN -- ends simulation cleanly (custom1 funct3=0b101)
    .insn r 0x2b, 0b101, 0, x0, x0, x0

    # Safety loop (never reached)
    j .
