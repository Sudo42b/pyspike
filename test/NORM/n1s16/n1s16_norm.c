//==================================================================
// n1s16_norm — ggml NORM (per-row layer normalization)
// dst[row, col] = (x - mean(row)) / sqrt(var(row) + eps)
// 257 rows x 32 FP16 elements, 1 NEST x 16 SPUs
//==================================================================

#include "intrin.h"
#include "gtx/address.h"
#include <stdint.h>

#define NEST_NUM             1
#define SPU_NUM_PER_NEST     16
#define DTYPE                4       // FP16

#define WIDTH                32
#define HEIGHT               257

#define BASE_DDR_A           0x1000000
#define BASE_DDR_RESULT      0xf000000

#define L2_A                 0x000000
#define L2_RESULT            0x008000

#define BANK_A               0x00000
#define BANK_B               0x20000
#define BANK_C               0x30000
#define BANK_R               0x50000

#define ROW_BYTES            (WIDTH * DTYPE)
#define ROWS_PER_NEST        (HEIGHT / NEST_NUM)
#define ROWS_PER_SPU_BASE    (ROWS_PER_NEST / SPU_NUM_PER_NEST)
#define ROWS_REMAINDER       (ROWS_PER_NEST % SPU_NUM_PER_NEST)
#define NEST_DATA_BYTES      (ROWS_PER_NEST * ROW_BYTES)

// generate_data.cpp uses DEFAULT_EPS = 1e-5f for ggml_norm.
#define FP16_EPS_1E_NEG_5    0x37280000

int main(void) {
    uint8_t nest_id = 0;
    uint32_t nest_off = (uint32_t)nest_id * NEST_DATA_BYTES;
    uint16_t active_tid_mask = 0xFFFFu;

    __split();

    {
        __start_plan(nest_id);

            __start_shared();
                __load_cr(GTX_MAIN_ADDR(BASE_DDR_A) + nest_off, L2_A,
                    NEST_DATA_BYTES,
                    (uint16_t)NEST_DATA_BYTES,
                    1, (uint16_t)NEST_DATA_BYTES,
                    1, active_tid_mask, 0xBEEF);

                __credit_chk(active_tid_mask);

                __store_cr(L2_RESULT, GTX_MAIN_ADDR(BASE_DDR_RESULT) + nest_off,
                    NEST_DATA_BYTES,
                    (uint16_t)NEST_DATA_BYTES,
                    1, (uint16_t)NEST_DATA_BYTES,
                    1, active_tid_mask);
            __end_shared();

            for (uint8_t tid = 0; tid < SPU_NUM_PER_NEST; tid++) {
                uint16_t tid_mask = (uint16_t)(0x1u << tid);
                uint8_t rows_for_tid = (uint8_t)(ROWS_PER_SPU_BASE + (tid < ROWS_REMAINDER ? 1 : 0));
                uint32_t first_row = (uint32_t)tid * ROWS_PER_SPU_BASE
                    + (uint32_t)(tid < ROWS_REMAINDER ? tid : ROWS_REMAINDER);

                __start_thread(tid);
                    __set_spm_addr(BANK_R, BANK_C, BANK_B, BANK_A);
                    __credit_chk(0xBEEF);

                    for (uint8_t r = 0; r < rows_for_tid; r++) {
                        uint32_t row_off = (first_row + r) * ROW_BYTES;

                        if (r == rows_for_tid - 1) {
                            __load_cr(L2_A + row_off, BANK_A,
                                ROW_BYTES, (uint16_t)ROW_BYTES, 1, (uint16_t)ROW_BYTES,
                                1, tid_mask, nest_id);
                        } else {
                            __load(L2_A + row_off, BANK_A,
                                ROW_BYTES, (uint16_t)ROW_BYTES, 1, (uint16_t)ROW_BYTES);
                        }

                        __layernorm(WIDTH, BANK_A, BANK_B, BANK_R, FP16_EPS_1E_NEG_5);

                        if (r == rows_for_tid - 1) {
                            __store_cr(BANK_R, L2_RESULT + row_off,
                                ROW_BYTES, (uint16_t)ROW_BYTES, 1, (uint16_t)ROW_BYTES,
                                1, tid_mask);
                        } else {
                            __store(BANK_R, L2_RESULT + row_off,
                                ROW_BYTES, (uint16_t)ROW_BYTES, 1, (uint16_t)ROW_BYTES);
                        }
                    }
                __end_thread(tid);
            }

        __end_plan(nest_id);
    }

    __join();
    return 0;
}
