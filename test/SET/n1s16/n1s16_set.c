// =================================================================
// GGML_OP_SET — Set a 2D top-left region of dst with src1 data (n1s16)
// Verifier shape [1024,64]: src0 is [1024,64] FP16 and src1 is [512,32] FP16.
// ggml_set_2d copies src0 to dst, then overwrites the first src1 rows with
// a row stride equal to the full dst row width. Uses canonical split/plan/shared
// DDR->L2->DDR DMA movement without SPU ALU.
// =================================================================

#include "intrin.h"
#include "gtx/address.h"
#include "gtx_csr.h"
#include <stdint.h>

#ifndef NEST_ID
#define NEST_ID             0
#endif

#define BASE_DDR_SRC0       0x1000000
#define BASE_DDR_SRC1       0x2000000
#define BASE_DDR_RESULT     0xf000000

#define L2_RESULT           0x000000

#define ROWS                257
#define COLS                4096
#define FP16_B              2
#define PATCH_ROWS          (ROWS / 2)
#define PATCH_COLS          (COLS / 2)

#define DST_ROW_BYTES       (COLS * FP16_B)
#define PATCH_ROW_BYTES     (PATCH_COLS * FP16_B)
#define TOTAL_BYTES         (ROWS * DST_ROW_BYTES)

int main(void)
{
    uint8_t nest_id = NEST_ID;

    __split();
    {
        __start_plan(nest_id);
            __start_shared();
                // Copy the full destination base tensor to L2 row-wise so each
                // 2D DMA length stays within the 16-bit transfer field.
                __load(GTX_MAIN_ADDR(BASE_DDR_SRC0), L2_RESULT,
                    (uint32_t)DST_ROW_BYTES,
                    (uint16_t)DST_ROW_BYTES,
                    ROWS,
                    (uint16_t)DST_ROW_BYTES);

                // Overlay src1 into the top-left patch. The source patch rows are
                // contiguous, while destination rows keep the full src0 stride.
                __load(GTX_MAIN_ADDR(BASE_DDR_SRC1), L2_RESULT,
                    (uint32_t)PATCH_ROW_BYTES,
                    (uint16_t)PATCH_ROW_BYTES,
                    PATCH_ROWS,
                    (uint16_t)DST_ROW_BYTES);

                __store(L2_RESULT, GTX_MAIN_ADDR(BASE_DDR_RESULT),
                    (uint32_t)DST_ROW_BYTES,
                    (uint16_t)DST_ROW_BYTES,
                    ROWS,
                    (uint16_t)DST_ROW_BYTES);
            __end_shared();
        __end_plan(nest_id);
    }
    __join();

    return 0;
}
