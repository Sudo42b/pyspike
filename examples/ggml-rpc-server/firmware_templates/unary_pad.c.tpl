//==================================================================
// {{OP_NAME}} (generated) — pad with zeros (right/bottom), via GTX_LAUNCH<<<1,0>>>.
// Shared-only kernel: no per-SPU body; the __pad intrinsic does the work
// inside the shared section.
//==================================================================

#include "gtx_kernel.h"
#include "gtx_csr.h"

#define NESTS               1
#define SRC_ROWS            {{SRC_ROWS}}
#define SRC_COLS            {{SRC_COLS}}
#define PAD_RIGHT           {{PAD_RIGHT}}
#define PAD_BOTTOM          {{PAD_BOTTOM}}

#define BASE_DDR_A          0x1000000
#define BASE_DDR_R          0xf000000
#define L2_PAD              0x000000

#define FP16_B              2
#define SRC_ROW_BYTES       (SRC_COLS * FP16_B)
#define DST_COLS            (SRC_COLS + PAD_RIGHT)
#define DST_ROWS            (SRC_ROWS + PAD_BOTTOM)
#define DST_ROW_BYTES       (DST_COLS * FP16_B)
#define DST_BYTES           (DST_ROWS * DST_ROW_BYTES)
#define CHANNELS            1
#define FP16_DTYPE_SHIFT    1
#define PAD_FILL_ZERO       0ULL

GTX_KERNEL_BODY(
    /* SHARED_BODY */ {
        __pad(GTX_MAIN_ADDR(BASE_DDR_A), L2_PAD,
            (uint32_t)SRC_ROW_BYTES,
            (uint16_t)SRC_ROW_BYTES,
            (uint16_t)SRC_ROWS,
            0, PAD_BOTTOM, 0, PAD_RIGHT,
            PAD_FILL_ZERO,
            (uint32_t)(SRC_ROWS * SRC_ROW_BYTES),
            CHANNELS, 2, FP16_DTYPE_SHIFT);

        __store(L2_PAD, GTX_MAIN_ADDR(BASE_DDR_R),
            (uint32_t)DST_BYTES,
            (uint16_t)DST_BYTES,
            1, (uint16_t)DST_BYTES);
    },
    /* THREAD_BODY */ { /* shared-only: no per-SPU work */ }
)

int main(void) {
    GTX_LAUNCH_SHARED(NESTS);
    return 0;
}
