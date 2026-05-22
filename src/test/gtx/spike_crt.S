/* spike_crt.S — Minimal CRT for GTX NPU tests on Spike
 *
 * Sets stack, clears BSS, calls main(), terminates via HTIF tohost.
 * Compatible with linker script at ORIGIN = 0x80000000.
 */

    .section .text.init
    .global _start
_start:
    /* Set global pointer (PC-relative, no absolute relocation) */
    .option push
    .option norelax
    la      gp, _gp
    .option pop

    /* Set stack pointer */
    la      sp, __C_STACK_TOP__

    /* Enable the FP unit (mstatus.FS = Initial). Kernels that use host
     * float (ROPE/TIMESTEP_EMBEDDING sincos) emit hard-float insns; without
     * FS!=Off the first FP op traps to mtvec (=0, unset) and spins forever. */
    li      t0, 0x2000
    csrs    mstatus, t0

    /* Clear BSS */
    la      t0, __BSS_START__
    la      t1, __BSS_END__
1:  bgeu    t0, t1, 2f
    sd      zero, 0(t0)
    addi    t0, t0, 8
    j       1b
2:

    /* Call main */
    call    main

    /* Signal pass/fail to Spike via HTIF tohost */
    /* tohost = (a0 == 0) ? 1 : (a0 << 1) | 1   (1 = PASS) */
    la      t0, tohost
    slli    a0, a0, 1
    ori     a0, a0, 1
    sd      a0, 0(t0)

    /* Spin forever (Spike will stop when it sees tohost) */
3:  j       3b
