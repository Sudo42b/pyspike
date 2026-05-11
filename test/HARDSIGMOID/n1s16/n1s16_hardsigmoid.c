//==================================================================
// n1s16_hardsigmoid — element-wise hardsigmoid with 1 NEST x 16 SPUs
// dst[row] = min(max(x+3, 0), 6) / 6, 64 rows x 8 FP16 elements
// FP16 constants: 3.0=0x4200, 0.0=0x0000, 6.0=0x4600, 1/6=0x3155
//==================================================================

#include "intrin.h"
#include "gtx/address.h"
#include "gtx_csr.h"

#define NEST_NUM            1
#define SPU_NUM_PER_NEST    16
#define DTYPE               2       // FP16

#define WIDTH               127
#define HEIGHT              31

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
#define ROWS_BASE_PER_SPU   (ROWS_PER_NEST / SPU_NUM_PER_NEST)
#define ROWS_REM_PER_SPU    (ROWS_PER_NEST % SPU_NUM_PER_NEST)
#define ACTIVE_SPU_NUM      ((ROWS_PER_NEST < SPU_NUM_PER_NEST) ? ROWS_PER_NEST : SPU_NUM_PER_NEST)
#define ACTIVE_SPU_MASK     ((ACTIVE_SPU_NUM >= SPU_NUM_PER_NEST) ? 0xFFFFu : ((1u << ACTIVE_SPU_NUM) - 1u))


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
                    1, ACTIVE_SPU_MASK, 0xBEEF);

                __credit_chk(ACTIVE_SPU_MASK);

                __store_cr(L2_RESULT, GTX_MAIN_ADDR(BASE_DDR_RESULT) + nest_off,
                    (uint32_t)(ROWS_PER_NEST * ROW_BYTES),
                    (uint16_t)(ROWS_PER_NEST * ROW_BYTES),
                    1, (uint16_t)(ROWS_PER_NEST * ROW_BYTES),
                    1, ACTIVE_SPU_MASK);
            __end_shared();

            for (uint8_t tid = 0; tid < SPU_NUM_PER_NEST; tid++) {
                __start_thread(tid);
                    if (tid < ACTIVE_SPU_NUM) {
                        uint32_t tid_mask = (uint32_t)(1u << tid);
                        uint32_t nest_mask = (uint32_t)(1u << nest_id);

                        __set_spm_addr(BANK_R, BANK_C, BANK_B, BANK_A);
                        __credit_chk(0xBEEF);
                        uint8_t row_count = (uint8_t)(ROWS_BASE_PER_SPU + ((tid < ROWS_REM_PER_SPU) ? 1u : 0u));
                        uint32_t row_start = (uint32_t)tid * ROWS_BASE_PER_SPU +
                            ((tid < ROWS_REM_PER_SPU) ? (uint32_t)tid : (uint32_t)ROWS_REM_PER_SPU);

                        for (uint8_t r = 0; r < row_count; r++) {
                            uint32_t row_off = (row_start + (uint32_t)r) * ROW_BYTES;

                            // L2 -> L1: src0 row to Bank A
                            __load(L2_A + row_off, BANK_A,
                                ROW_BYTES, (uint16_t)ROW_BYTES, 1, (uint16_t)ROW_BYTES
    );
                            if (r == row_count - 1) __credit_ld(tid_mask, nest_mask);

                            // hardsigmoid = min(max(x+3, 0), 6) / 6
                            // A has x; R = x + 3
                            __add_vs(WIDTH, 0x4200, 0);
                            // R -> A
                            __copy(BANK_R, BANK_A, ROW_BYTES, ROW_BYTES, 1, ROW_BYTES);
                            // R = max(0, x+3)
                            __clamp_min(WIDTH, 0x0000, 0);
                            // R -> A
                            __copy(BANK_R, BANK_A, ROW_BYTES, ROW_BYTES, 1, ROW_BYTES);
                            // R = min(6, relu(x+3))
                            __clamp_max(WIDTH, 0x4600, 0);
                            // R -> A
                            __copy(BANK_R, BANK_A, ROW_BYTES, ROW_BYTES, 1, ROW_BYTES);
                            // R = result / 6
                            __mul_vs(WIDTH, 0x3155, 0);

                            // L1 -> L2: result from Bank R
                            if (r == row_count - 1) {
                                __store_cr(BANK_R, L2_RESULT + row_off, ROW_BYTES, (uint16_t)ROW_BYTES, 1, (uint16_t)ROW_BYTES, 1, tid_mask);
                            } else {
                                __store(BANK_R, L2_RESULT + row_off, ROW_BYTES, (uint16_t)ROW_BYTES, 1, (uint16_t)ROW_BYTES);
                            }
                        }
                    }
                __end_thread(tid);
            }

        __end_plan(nest_id);
    }

    __join();
    return 0;
}
