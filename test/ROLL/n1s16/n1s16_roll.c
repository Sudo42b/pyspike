// =================================================================
// GGML_OP_ROLL — Circular shift along ne0 (n1s16, 16 rows x 32 FP16)
// generate_data uses ggml_roll(ctx, src0, 1, 0, 0, 0):
//   dst[row][0]     = src[row][31]
//   dst[row][1..31] = src[row][0..30]
// Uses canonical split/plan/shared DDR->L2->DDR strided DMA.
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

#define L2_R                0x000000

#define ROWS                257
#define COLS                4096
#define SHIFT0              1
#define FP16_B              2
#define ROW_BYTES           (COLS * FP16_B)                  // 64
#define HEAD_BYTES          ((COLS - SHIFT0) * FP16_B)       // 62
#define TAIL_BYTES          (SHIFT0 * FP16_B)                // 2
#define TOTAL_BYTES         (ROWS * ROW_BYTES)               // 1024

int main(void)
{
    uint8_t nest_id = NEST_ID;

    __split();
    {
        __start_plan(nest_id);
            __start_shared();
                // Move the wrapped tail element of each row to dst column 0.
                __load(GTX_MAIN_ADDR(BASE_DDR_A) + HEAD_BYTES, L2_R,
                    (uint32_t)ROW_BYTES,
                    (uint16_t)TAIL_BYTES,
                    (uint16_t)ROWS,
                    (uint16_t)ROW_BYTES);

                // Move the row head to dst columns 1..31.
                __load(GTX_MAIN_ADDR(BASE_DDR_A), L2_R + TAIL_BYTES,
                    (uint32_t)ROW_BYTES,
                    (uint16_t)HEAD_BYTES,
                    (uint16_t)ROWS,
                    (uint16_t)ROW_BYTES);

                __store(L2_R, GTX_MAIN_ADDR(BASE_DDR_R),
                    (uint32_t)ROW_BYTES,
                    (uint16_t)ROW_BYTES,
                    (uint16_t)ROWS,
                    (uint16_t)ROW_BYTES);
            __end_shared();
        __end_plan(nest_id);
    }
    __join();

    return 0;
}
