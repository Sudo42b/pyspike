// =================================================================
// GGML_OP_PAD — Pad tensor with zeros (n1s16, 126x255 FP16 -> 127x256 FP16)
// ggml_pad(ctx, src0, 1, 1, 0, 0) appends one column and one row of zeros.
// Uses the direct Level-3 __pad memory intrinsic in shared mode, then stores
// the padded L2 tile back to DDR.
// =================================================================

#include "intrin.h"
#include "gtx/address.h"
#include "gtx_csr.h"
#include <stdint.h>

#ifndef NEST_ID
#define NEST_ID             0
#endif

#define BASE_DDR_A          0x1000000
#define BASE_DDR_R          0xf000000

#define L2_PAD              0x000000

#define SRC_ROWS            126
#define SRC_COLS            255
#define PAD_RIGHT           1
#define PAD_BOTTOM          1
#define FP16_B              2
#define SRC_ROW_BYTES       (SRC_COLS * FP16_B)          // 510
#define DST_COLS            (SRC_COLS + PAD_RIGHT)       // 256
#define DST_ROWS            (SRC_ROWS + PAD_BOTTOM)      // 127
#define DST_ROW_BYTES       (DST_COLS * FP16_B)          // 512
#define DST_BYTES           (DST_ROWS * DST_ROW_BYTES)   // 65024
#define CHANNELS            1
#define FP16_DTYPE_SHIFT    1
#define PAD_FILL_ZERO       0ULL

int main(void)
{
    uint8_t nest_id = NEST_ID;

    __split();
    {
        __start_plan(nest_id);
            __start_shared();
                __pad(
                    GTX_MAIN_ADDR(BASE_DDR_A),
                    L2_PAD,
                    (uint32_t)SRC_ROW_BYTES,
                    (uint16_t)SRC_ROW_BYTES,
                    (uint16_t)SRC_ROWS,
                    0,
                    PAD_BOTTOM,
                    0,
                    PAD_RIGHT,
                    PAD_FILL_ZERO,
                    (uint32_t)(SRC_ROWS * SRC_ROW_BYTES),
                    CHANNELS,
                    2,
                    FP16_DTYPE_SHIFT);

                __store(L2_PAD, GTX_MAIN_ADDR(BASE_DDR_R),
                    (uint32_t)DST_BYTES,
                    (uint16_t)DST_BYTES,
                    1,
                    (uint16_t)DST_BYTES);
            __end_shared();
        __end_plan(nest_id);
    }
    __join();

    return 0;
}
