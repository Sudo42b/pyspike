//==================================================================
// n1s16_sigmoid — element-wise sigmoid with 1 NEST x 16 SPUs
// dst[row] = sigmoid(src0[row]), HEIGHT rows x WIDTH FP16 elements
//==================================================================

#include "intrin.h"
#include "gtx/address.h"
#include "gtx_csr.h"

#define NEST_NUM            1
#define SPU_NUM_PER_NEST    16
#define DTYPE               4       // FP16

#define WIDTH               32
#define HEIGHT              128

#define BASE_DDR_A          0x1000000
#define BASE_DDR_RESULT     0xf000000

#define L2_A                0x000000
#define L2_RESULT           0x002000

#define BANK_A              0x00000
#define BANK_B              0x20000
#define BANK_C              0x30000
#define BANK_R              0x50000

#define ROW_BYTES           (WIDTH * DTYPE)
#define ROWS_PER_NEST       (HEIGHT / NEST_NUM)
#define ROWS_PER_SPU_BASE   (ROWS_PER_NEST / SPU_NUM_PER_NEST)
#define ROWS_PER_SPU_REM    (ROWS_PER_NEST % SPU_NUM_PER_NEST)


int main(void) {


    __split();

    {
        uint8_t nest_id = 0;

        __start_plan(nest_id);

            __start_shared();
                uint32_t nest_off = (uint32_t)nest_id * ROWS_PER_NEST * ROW_BYTES;

                __load_cr(GTX_MAIN_ADDR(BASE_DDR_A) + nest_off, L2_A,
                    (uint32_t)(ROWS_PER_NEST * ROW_BYTES),
                    (uint16_t)(ROWS_PER_NEST * ROW_BYTES),
                    1, (uint16_t)(ROWS_PER_NEST * ROW_BYTES),
                    1, 0xFFFF, 0xBEEF);

                __credit_chk(0xFFFF);

                __store_cr(L2_RESULT, GTX_MAIN_ADDR(BASE_DDR_RESULT) + nest_off,
                    (uint32_t)(ROWS_PER_NEST * ROW_BYTES),
                    (uint16_t)(ROWS_PER_NEST * ROW_BYTES),
                    1, (uint16_t)(ROWS_PER_NEST * ROW_BYTES),
                    1, 0xFFFF);
            __end_shared();

            for (uint8_t tid = 0; tid < SPU_NUM_PER_NEST; tid++) {
                __start_thread(tid);
                    __set_spm_addr(BANK_R, BANK_C, BANK_B, BANK_A);
                    __credit_chk(0xBEEF);
                    uint16_t rows_this_spu = ROWS_PER_SPU_BASE + (tid < ROWS_PER_SPU_REM ? 1 : 0);
                    uint16_t row_start = (uint16_t)tid * ROWS_PER_SPU_BASE + (tid < ROWS_PER_SPU_REM ? tid : ROWS_PER_SPU_REM);
                    for (uint16_t r = 0; r < rows_this_spu; r++) {
                        uint32_t row_off = (uint32_t)(row_start + r) * ROW_BYTES;

                        // Load input to Bank R (sigm reads from R, writes to A)
                        __load(L2_A + row_off, BANK_R,
                            ROW_BYTES, (uint16_t)ROW_BYTES, 1, (uint16_t)ROW_BYTES);
                        if (r == rows_this_spu - 1) __credit_ld((1u << tid), (1u << nest_id));

                        // sigmoid: Bank R -> Bank A
                        __sigm(WIDTH);

                        // Store result from Bank A
                        if (r == rows_this_spu - 1) {
                            __store_cr(BANK_A, L2_RESULT + row_off, ROW_BYTES, (uint16_t)ROW_BYTES, 1, (uint16_t)ROW_BYTES, 1, 0x1 << tid);
                        } else {
                            __store(BANK_A, L2_RESULT + row_off, ROW_BYTES, (uint16_t)ROW_BYTES, 1, (uint16_t)ROW_BYTES);
                        }
                    }
                __end_thread(tid);
            }

        __end_plan(nest_id);
    }

    __join();
    return 0;
}
