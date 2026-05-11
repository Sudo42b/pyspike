// =================================================================
// GGML_OP_REPEAT — Repeat source tensor to target tensor (n1s16)
// Self-check shape [530,1030]: src0 is [265,515] FP16 and dst is
// [530,1030] FP16. ggml_repeat tiles inner dimension first, then repeats
// full source-row blocks. Uses canonical split/plan/shared DDR->L2->DDR DMA
// movement without SPU ALU.
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

#define FP16_B              2

#define SRC_COLS            265
#define SRC_ROWS            515
#define DST_COLS            530
#define DST_ROWS            1030

#define L2_A                0x000000
#define L2_RESULT           0x080000

#define SRC_ROW_BYTES       (SRC_COLS * FP16_B)        // 530
#define DST_ROW_BYTES       (DST_COLS * FP16_B)        // 1060
#define SRC_BYTES           (SRC_ROWS * SRC_ROW_BYTES) // 272950
#define DST_BYTES           (DST_ROWS * DST_ROW_BYTES) // 1091800

int main(void)
{
    uint8_t nest_id = NEST_ID;

    __split();
    {
        __start_plan(nest_id);
            __start_shared();
                __load(GTX_MAIN_ADDR(BASE_DDR_A), L2_A,
                    (uint32_t)SRC_ROW_BYTES,
                    (uint16_t)SRC_ROW_BYTES,
                    SRC_ROWS,
                    (uint16_t)SRC_ROW_BYTES);

                // Build the full repeated destination in L2 using 2D copies.
                // Each copy transfers all source rows with source-row stride and
                // destination-row stride, avoiding thousands of narrow DDR stores.
                __copy(L2_A,
                    L2_RESULT,
                    (uint32_t)SRC_ROW_BYTES,
                    (uint16_t)SRC_ROW_BYTES,
                    SRC_ROWS,
                    (uint16_t)DST_ROW_BYTES);

                __copy(L2_A,
                    L2_RESULT + SRC_ROW_BYTES,
                    (uint32_t)SRC_ROW_BYTES,
                    (uint16_t)SRC_ROW_BYTES,
                    SRC_ROWS,
                    (uint16_t)DST_ROW_BYTES);

                __copy(L2_A,
                    L2_RESULT + (uint32_t)SRC_ROWS * DST_ROW_BYTES,
                    (uint32_t)SRC_ROW_BYTES,
                    (uint16_t)SRC_ROW_BYTES,
                    SRC_ROWS,
                    (uint16_t)DST_ROW_BYTES);

                __copy(L2_A,
                    L2_RESULT + (uint32_t)SRC_ROWS * DST_ROW_BYTES + SRC_ROW_BYTES,
                    (uint32_t)SRC_ROW_BYTES,
                    (uint16_t)SRC_ROW_BYTES,
                    SRC_ROWS,
                    (uint16_t)DST_ROW_BYTES);

                __store(L2_RESULT, GTX_MAIN_ADDR(BASE_DDR_R),
                    (uint32_t)DST_ROW_BYTES,
                    (uint16_t)DST_ROW_BYTES,
                    DST_ROWS,
                    (uint16_t)DST_ROW_BYTES);
            __end_shared();
        __end_plan(nest_id);
    }
    __join();

    return 0;
}
