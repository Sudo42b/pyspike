// =================================================================
// GGML_OP_CPY — Copy tensor with type conversion (n1s16, 32x1023 FP16)
// For the generated FP16->FP16 case this is a direct data movement kernel.
// dst = copy(src0)
// Uses canonical split/plan/shared DDR->L2->DDR bulk DMA.
// =================================================================

#include "intrin.h"
#include "gtx/address.h"
#include "gtx_csr.h"

#ifndef NEST_ID
#define NEST_ID             0
#endif

#define BASE_DDR_A          0x1000000
#define BASE_DDR_R          0xf000000

#define L2_A                0x000000

#define ROWS                32
#define COLS                1023
#define FP16_B              2
#define TOTAL_BYTES         (ROWS * COLS * FP16_B)  // 65472

int main(void)
{
    uint8_t nest_id = NEST_ID;

    __split();
    {
        __start_plan(nest_id);
            __start_shared();
                __load(GTX_MAIN_ADDR(BASE_DDR_A), L2_A,
                    (uint32_t)TOTAL_BYTES,
                    (uint16_t)TOTAL_BYTES,
                    1,
                    (uint16_t)TOTAL_BYTES);

                __store(L2_A, GTX_MAIN_ADDR(BASE_DDR_R),
                    (uint32_t)TOTAL_BYTES,
                    (uint16_t)TOTAL_BYTES,
                    1,
                    (uint16_t)TOTAL_BYTES);
            __end_shared();
        __end_plan(nest_id);
    }
    __join();

    return 0;
}
