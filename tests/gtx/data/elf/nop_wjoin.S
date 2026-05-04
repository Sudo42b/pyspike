# Minimal NOP+WJOIN firmware for Phase 2 skeleton regression test (D-22).
#
# Behavior:
#   1. addi sp, sp, -16   ;; uses sp init = 0x80100000 (set in GtxNpu.reset) ->
#                         ;; sp = 0x800FFFF0 (valid DRAM, won't trap)
#   2. .insn r 0x2b, ...  ;; custom1 funct3=0b101 = WJOIN
#   3. j .                ;; safety loop (WJOIN raises SystemExit; never reached)
#
# .insn r encoding for custom1 funct3=0b101:
#   Format: .insn r OPCODE, FUNCT3, FUNCT7, RD, RS1, RS2
#     OPCODE = 0x2b (custom-1)
#     FUNCT3 = 0b101 = 5  (encodes {xd=1, xs1=0, xs2=1} = WJOIN)
#     FUNCT7 = 0           (ignored for warp ops)
#     RD = x0, RS1 = x0, RS2 = x0  (no reg I/O)
#
# Build: see Makefile (riscv64-unknown-elf-gcc -nostdlib -Ttext=0x80000000 ...)

.section .text._start, "ax"
.global _start
_start:
    addi  sp, sp, -16
    .insn r 0x2b, 0b101, 0, x0, x0, x0
    j .
