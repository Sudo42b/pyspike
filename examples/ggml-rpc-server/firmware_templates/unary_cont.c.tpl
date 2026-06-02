//==================================================================
// {{OP_NAME}} (generated) — CONT/DUP same-dtype DDR→DDR copy.
// Layout: (N_ROWS rows × ROW_BYTES each) packed contiguous bytes.
// Pure shared-section __copy_mem; no L1/L2 staging, no SPU body.
//==================================================================

#include "gtx_kernel.h"

#define NESTS               1

#define ROW_BYTES           {{ROW_BYTES}}
#define N_ROWS              {{N_ROWS}}

#define BASE_DDR_INPUT      0x1000000
#define BASE_DDR_RESULT     0xf000000

#if ROW_BYTES == 0 || N_ROWS == 0
#error "cont needs non-zero ROW_BYTES and N_ROWS"
#endif
#if ROW_BYTES > 65535
#error "cont ROW_BYTES exceeds 16-bit DMA length cap"
#endif
#if N_ROWS > 65535
#error "cont N_ROWS exceeds 16-bit DMA height cap"
#endif

GTX_KERNEL_BODY(
    /* SHARED_BODY */ {
        __copy_mem(
            GTX_MAIN_ADDR(BASE_DDR_INPUT),
            GTX_MAIN_ADDR(BASE_DDR_RESULT),
            (uint32_t)ROW_BYTES,
            (uint16_t)ROW_BYTES,
            (uint16_t)N_ROWS,
            (uint16_t)ROW_BYTES,
            0);
    },
    /* THREAD_BODY */ { /* shared-only */ }
)

int main(void) {
    GTX_LAUNCH_SHARED(NESTS);
    return 0;
}
