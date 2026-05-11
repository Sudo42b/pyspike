//==================================================================
// n1s16_log — element-wise natural logarithm, 1 NEST x 16 SPUs
// dst[row] = ln(src0[row]), 127 rows x 64 FP16 elements
//==================================================================

#include "intrin.h"
#include "gtx/address.h"
#include <stdint.h>

#define NEST_NUM            1
#define SPU_NUM_PER_NEST    16
#define DTYPE               2       // FP16

#define WIDTH               64
#define HEIGHT              127

#define BASE_DDR_A          0x1000000
#define BASE_DDR_RESULT     0xf000000

#define L2_A                0x000000
#define L2_RESULT           0x008000

#define BANK_A              0x00000
#define BANK_B              0x20000
#define BANK_C              0x30000
#define BANK_R              0x50000

#define FP16_QNAN           0xFE00

#define ROW_BYTES           (WIDTH * DTYPE)
#define ROWS_PER_NEST       (HEIGHT / NEST_NUM)
#define ROWS_BASE_PER_SPU   (ROWS_PER_NEST / SPU_NUM_PER_NEST)
#define ROWS_REMAINDER      (ROWS_PER_NEST % SPU_NUM_PER_NEST)
#define ACTIVE_SPU_COUNT    ((ROWS_PER_NEST < SPU_NUM_PER_NEST) ? ROWS_PER_NEST : SPU_NUM_PER_NEST)
#define ACTIVE_SPU_MASK     ((uint16_t)((1u << ACTIVE_SPU_COUNT) - 1u))


int main(void) {
    __split();

    {
        uint8_t nest_id = 0;
        uint32_t nest_off = (uint32_t)nest_id * ROWS_PER_NEST * ROW_BYTES;

        __start_plan(nest_id);

            __start_shared();
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
                uint16_t tid_mask = (uint16_t)(1u << tid);
                uint16_t rows_for_tid = (uint16_t)(ROWS_BASE_PER_SPU + ((tid < ROWS_REMAINDER) ? 1 : 0));
                uint16_t start_row = (uint16_t)(tid * ROWS_BASE_PER_SPU + ((tid < ROWS_REMAINDER) ? tid : ROWS_REMAINDER));

                __start_thread(tid);
                    if (rows_for_tid > 0) {
                        __set_spm_addr(BANK_R, BANK_C, BANK_B, BANK_A);
                        __credit_chk(0xBEEF);

                        for (uint16_t r = 0; r < rows_for_tid; r++) {
                            uint32_t row_off = (uint32_t)(start_row + r) * ROW_BYTES;

                            if (r == rows_for_tid - 1) {
                                __load_cr(L2_A + row_off, BANK_A,
                                    ROW_BYTES, (uint16_t)ROW_BYTES, 1, (uint16_t)ROW_BYTES,
                                    1, tid_mask, nest_id);
                            } else {
                                __load(L2_A + row_off, BANK_A,
                                    ROW_BYTES, (uint16_t)ROW_BYTES, 1, (uint16_t)ROW_BYTES);
                            }

                            // __ln_v returns -inf for both zero and negative FP16 inputs on ISS.
                            // ggml LOG follows logf: log(0) = -inf, log(x < 0) = NaN.
                            // Use __ln_v for the full vector, then patch only negative non-zero
                            // FP16 lanes to the ggml/CPU quiet-NaN reference encoding.
                            __copy(BANK_A, BANK_C, ROW_BYTES, ROW_BYTES, 1, ROW_BYTES);
                            __ln_v(WIDTH, 0);
                            volatile const uint16_t *src_bits = (volatile const uint16_t *)(uintptr_t)BANK_C;
                            volatile uint16_t *dst_bits = (volatile uint16_t *)(uintptr_t)BANK_R;
                            for (uint8_t c = 0; c < WIDTH; c++) {
                                uint16_t x_bits = src_bits[c];
                                if ((x_bits & 0x8000u) && (x_bits & 0x7FFFu)) {
                                    dst_bits[c] = FP16_QNAN;
                                }
                            }

                            if (r == rows_for_tid - 1) {
                                __store_cr(BANK_R, L2_RESULT + row_off,
                                    ROW_BYTES, (uint16_t)ROW_BYTES, 1, (uint16_t)ROW_BYTES,
                                    1, tid_mask);
                            } else {
                                __store(BANK_R, L2_RESULT + row_off,
                                    ROW_BYTES, (uint16_t)ROW_BYTES, 1, (uint16_t)ROW_BYTES);
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
