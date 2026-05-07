# P6 VRF-03: relu hand-written firmware (Plan 03 Task 2a).
#
# NOT a vendor 1:1 .c-build (RESEARCH finding #3: vendor sources require
# gtx-firmware/include + custom -march=rv64g_xgtxnpu, infeasible for v1 CI).
# Hand-written .S using GTX RoCC opcodes directly (P5 activation_relu_gelu.S
# precedent direct extension). Built with stock /opt/riscv toolchain
# (-march=rv64imafd_zicsr_zifencei -O0 implicit; -nostdlib -nostartfiles).
#
# Per-op encoding:
#   funct7=0x06 (DISPATCH_ACT firmware forward), funct3=0,
#   sub_op=GTX_ACT_RELU=0 (staged via GSPR_GTX_OPCODE=0x004).
#
# Default L1 = zero-init -> output = relu(zeros) = zeros.

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

    # WRSPR GSPR_GTX_OPCODE (0x004) = sub_op = GTX_ACT_RELU = 0
    li    x1, 0x004
    li    x2, 0
    .insn r 0x0b, 0, 0x49, x0, x1, x2

    # firmware DISPATCH_ACT funct7=0x06 funct3=0 (forward)
    # rs1 must be nonzero so funct7=0x06 routes to DISPATCH_ACT (Pitfall F).
    li    x1, 0x10
    .insn r 0x0b, 0, 0x06, x0, x1, x0

    # WJOIN (custom1 funct3=0b101) -- raises SystemExit(0)
    .insn r 0x2b, 0b101, 0, x0, x0, x0

    # Safety loop (never reached)
    j .
