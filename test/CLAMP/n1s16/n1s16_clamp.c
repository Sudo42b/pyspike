//==================================================================
// n1s16_clamp — element-wise clamp with 1 NEST x 16 SPUs
// dst[row] = clamp(src0[row], min_val, max_val), 64 rows x 8 FP16
//          = max(min_val, min(max_val, src0))
//
// min_val and max_val (2 FP16) read from DDR at BASE_DDR_B before __split.
// Each SPU processes 1 row (8 elements = 1 SVR chunk).
// Uses __clamp_max / __clamp_min (element-wise Bank A -> R).
//==================================================================

#include "intrin.h"
#include "gtx/address.h"
#include "gtx_csr.h"

#define NEST_NUM            1
#define SPU_NUM_PER_NEST    16
#define DTYPE               2       // FP16

#define WIDTH               8
#define HEIGHT              512

#define BASE_DDR_A          0x1000000
#define BASE_DDR_B          0x2000000   // [min_val, max_val] (2 FP16)
#define BASE_DDR_RESULT     0xf000000

#define L2_A                0x000000
#define L2_RESULT           0x002000

#define BANK_A              0x00000
#define BANK_B              0x20000
#define BANK_C              0x30000
#define BANK_R              0x50000

#define ROW_BYTES           (WIDTH * DTYPE)     // 16
#define ROWS_PER_NEST       (HEIGHT / NEST_NUM) // 64
#define ROWS_PER_SPU_QUOT   (ROWS_PER_NEST / SPU_NUM_PER_NEST)
#define ROWS_PER_SPU_REM    (ROWS_PER_NEST % SPU_NUM_PER_NEST)
#define ACTIVE_SPU_COUNT    ((ROWS_PER_NEST < SPU_NUM_PER_NEST) ? ROWS_PER_NEST : SPU_NUM_PER_NEST)
#define ACTIVE_SPU_MASK     ((ACTIVE_SPU_COUNT >= SPU_NUM_PER_NEST) ? 0xFFFFu : ((1u << ACTIVE_SPU_COUNT) - 1u))
#define ROWS_THIS_SPU(tid)  ((uint32_t)((tid) < ROWS_PER_SPU_REM ? (ROWS_PER_SPU_QUOT + 1) : ROWS_PER_SPU_QUOT))
#define ROW_START_ROW(tid)  ((uint32_t)(((tid) < ROWS_PER_SPU_REM) ? ((tid) * (ROWS_PER_SPU_QUOT + 1)) : (ROWS_PER_SPU_REM * (ROWS_PER_SPU_QUOT + 1) + ((tid) - ROWS_PER_SPU_REM) * ROWS_PER_SPU_QUOT)))


int main(void) {

    // Read clamp parameters from DDR via CPU (before __split)
    volatile uint16_t *params = (volatile uint16_t *)GTX_MAIN_ADDR(BASE_DDR_B);
    uint16_t min_val = params[0];
    uint16_t max_val = params[1];

    __split();

    {
        uint8_t nest_id = 0;

        __start_plan(nest_id);

            __start_shared();
                uint32_t nest_off = (uint32_t)nest_id * ROWS_PER_NEST * ROW_BYTES;

                // Load src0 to L2_A with credit to all SPUs
                __load_cr(GTX_MAIN_ADDR(BASE_DDR_A) + nest_off, L2_A,
                    (uint32_t)(ROWS_PER_NEST * ROW_BYTES),
                    (uint16_t)(ROWS_PER_NEST * ROW_BYTES),
                    1, (uint16_t)(ROWS_PER_NEST * ROW_BYTES),
                    1, ACTIVE_SPU_MASK, 0xBEEF);

                // Wait all SPUs done
                __credit_chk(ACTIVE_SPU_MASK);

                // Store result to DDR
                __store_cr(L2_RESULT, GTX_MAIN_ADDR(BASE_DDR_RESULT) + nest_off,
                    (uint32_t)(ROWS_PER_NEST * ROW_BYTES),
                    (uint16_t)(ROWS_PER_NEST * ROW_BYTES),
                    1, (uint16_t)(ROWS_PER_NEST * ROW_BYTES),
                    1, ACTIVE_SPU_MASK);
            __end_shared();

            for (uint8_t tid = 0; tid < SPU_NUM_PER_NEST; tid++) {
                uint16_t tid_mask = (uint16_t)(0x1u << tid);
                uint32_t rows_this_spu = ROWS_THIS_SPU(tid);
                uint32_t row_start = ROW_START_ROW(tid);

                __start_thread(tid);
                    __set_spm_addr(BANK_R, BANK_C, BANK_B, BANK_A);
                    if (rows_this_spu > 0) {
                        __credit_chk(0xBEEF);
                    }
                    for (uint32_t r = 0; r < rows_this_spu; r++) {
                        uint32_t row_off = (row_start + r) * ROW_BYTES;

                        // Load row from L2 to Bank A
                        if (r == rows_this_spu - 1) {
                            __load_cr(L2_A + row_off, BANK_A,
                                ROW_BYTES, (uint16_t)ROW_BYTES, 1, (uint16_t)ROW_BYTES,
                                1, tid_mask, nest_id);
                        } else {
                            __load(L2_A + row_off, BANK_A,
                                ROW_BYTES, (uint16_t)ROW_BYTES, 1, (uint16_t)ROW_BYTES);
                        }

                        // clamp = max(min_val, min(max_val, x))
                        // A has x; clamp_max: R = min(A, max_val)
                        __clamp_max(WIDTH, max_val, 0);
                        // R -> A
                        __copy(BANK_R, BANK_A, ROW_BYTES, ROW_BYTES, 1, ROW_BYTES);
                        // clamp_min: R = max(A, min_val)
                        __clamp_min(WIDTH, min_val, 0);

                        // Store result L1 -> L2
                        if (r == rows_this_spu - 1) {
                            __store_cr(BANK_R, L2_RESULT + row_off,
                                ROW_BYTES, (uint16_t)ROW_BYTES, 1, (uint16_t)ROW_BYTES,
                                1, tid_mask);
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
