/* exit_shim — unify termination across SystemC-ISS / spike / pyspike.
 * gtx-firmware startup.c runs main() then _Exit() (infinite busy-loop):
 *   - ISS halts on the self-jump.
 *   - spike/pyspike need BOTH `tohost` and `fromhost` HTIF symbols to exit+dump.
 * --wrap=main: run real kernel (result -> DDR), write tohost so spike/pyspike
 * exit; startup's _Exit busy-loop then halts the ISS. One elf, all three sims.
 * Both symbols are referenced here so --gc-sections keeps them. */
extern int __real_main(void);
volatile unsigned long tohost   __attribute__((used, aligned(64)));
volatile unsigned long fromhost __attribute__((used, aligned(64)));
int __wrap_main(void) {
    int rc = __real_main();
    if (fromhost) { fromhost = 0; }                 /* keep fromhost referenced */
    tohost = ((unsigned long)rc << 1) | 1u;          /* HTIF exit */
    return rc;
}
