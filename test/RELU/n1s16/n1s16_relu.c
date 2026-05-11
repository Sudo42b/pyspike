//==================================================================
// n1s16_relu — element-wise ReLU with 1 NEST x 16 SPUs
// dst[row] = max(0, src0[row]), 256 rows x 1024 FP16 elements
// Uses GTX clamp-min vector path for max(x, 0) on each FP16 row tile.
// Shared DDR<->L2 transfers are row-shaped so the 16-bit DMA length field
// never truncates the 1024x256 tensor byte count.
//==================================================================

#include "intrin.h"
#include "gtx/address.h"
#include "gtx_csr.h"

#define NEST_NUM            1
#define SPU_NUM_PER_NEST    16
#define DTYPE               2       // FP16

#define WIDTH               1024
#define HEIGHT              256

#define BASE_DDR_A          0x1000000
#define BASE_DDR_RESULT     0xf000000

#define L2_A                0x000000
#define L2_RESULT           0x100000

#define BANK_A              0x00000
#define BANK_B              0x20000
#define BANK_C              0x30000
#define BANK_R              0x50000

#define ROW_BYTES           (WIDTH * DTYPE)
#define ROWS_PER_NEST       (HEIGHT / NEST_NUM)
#define ROWS_PER_SPU        (ROWS_PER_NEST / SPU_NUM_PER_NEST)


int main(void) {


    __split();

    {
        uint8_t nest_id = 0;

        __start_plan(nest_id);

            __start_shared();
                uint32_t nest_off = (uint32_t)nest_id * ROWS_PER_NEST * ROW_BYTES;

                __load_cr(GTX_MAIN_ADDR(BASE_DDR_A) + nest_off, L2_A,
                    (uint32_t)ROW_BYTES,
                    (uint16_t)ROW_BYTES,
                    (uint16_t)ROWS_PER_NEST, (uint32_t)ROW_BYTES,
                    1, 0xFFFF, 0xBEEF);

                __credit_chk(0xFFFF);

                __store_cr(L2_RESULT, GTX_MAIN_ADDR(BASE_DDR_RESULT) + nest_off,
                    (uint32_t)ROW_BYTES,
                    (uint16_t)ROW_BYTES,
                    (uint16_t)ROWS_PER_NEST, (uint32_t)ROW_BYTES,
                    1, 0xFFFF);
            __end_shared();

            for (uint8_t tid = 0; tid < SPU_NUM_PER_NEST; tid++) {
                uint16_t tid_mask = (uint16_t)(1u << tid);
                uint16_t nest_mask = (uint16_t)(1u << nest_id);

                __start_thread(tid);
                    __set_spm_addr(BANK_R, BANK_C, BANK_B, BANK_A);
                    __credit_chk(0xBEEF);
                    for (uint8_t r = 0; r < ROWS_PER_SPU; r++) {
                        uint32_t row_off = (uint32_t)(tid * ROWS_PER_SPU + r) * ROW_BYTES;

                        // L2 -> L1: src0 row to Bank A
                        if (r == ROWS_PER_SPU - 1) {
                            __load_cr(L2_A + row_off, BANK_A,
                                ROW_BYTES, (uint16_t)ROW_BYTES, 1, (uint16_t)ROW_BYTES,
                                1, tid_mask, nest_mask);
                        } else {
                            __load(L2_A + row_off, BANK_A,
                                ROW_BYTES, (uint16_t)ROW_BYTES, 1, (uint16_t)ROW_BYTES);
                        }

                        // ReLU: max(x, 0) = clamp_min(x, 0)
                        // __clamp_min operates element-wise on Bank A -> Bank R
                        __clamp_min(WIDTH, 0x0000, 0);

                        // L1 -> L2: result from Bank R
                        if (r == ROWS_PER_SPU - 1) {
                            __store_cr(BANK_R, L2_RESULT + row_off, ROW_BYTES, (uint16_t)ROW_BYTES, 1, (uint16_t)ROW_BYTES, 1, tid_mask);
                        } else {
                            __store(BANK_R, L2_RESULT + row_off, ROW_BYTES, (uint16_t)ROW_BYTES, 1, (uint16_t)ROW_BYTES);
                        }
                    }
                __end_thread(tid);
            }

        __end_plan(nest_id);
    }

    __join();
    return 0;
}
