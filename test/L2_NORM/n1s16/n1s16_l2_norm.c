//==================================================================
// n1s16_l2_norm — per-row L2 normalization with 1 NEST x 16 SPUs
// dst[i] = x[i] / max(sqrt(sum(x^2)), eps)   (NO 1/N division)
// 512 rows x 8 FP16 elements
//==================================================================

#include "intrin.h"
#include "gtx/address.h"
#include "gtx_csr.h"

#define NEST_NUM            1
#define SPU_NUM_PER_NEST    16
#define DTYPE               4

#define WIDTH               8
#define HEIGHT              512

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
#define ROWS_PER_SPU        (ROWS_PER_NEST / SPU_NUM_PER_NEST)
#define ROWS_REMAINDER      (ROWS_PER_NEST % SPU_NUM_PER_NEST)

#define SVR_ADDR            0x800
#define SVR_WORD_ADDR(svr, word) (SVR_ADDR + (uint32_t)(svr) * 4u + (uint32_t)(word))

#define FP16_EPS            0x37280000      // fp16(1e-5), matches generate_data DEFAULT_EPS
#define FP16_ONE_PAIR       0x3C003C003C003C00ULL


int main(void) {


    __split();

    {
        uint8_t nest_id = 0;

        __start_plan(nest_id);

            __start_shared();
                uint32_t nest_off = (uint32_t)nest_id * ROWS_PER_NEST * ROW_BYTES;

                // Load src0 to L2 with credit to all SPUs
                __load_cr(GTX_MAIN_ADDR(BASE_DDR_A) + nest_off, L2_A,
                    (uint32_t)(ROWS_PER_NEST * ROW_BYTES),
                    (uint16_t)(ROWS_PER_NEST * ROW_BYTES),
                    1, (uint16_t)(ROWS_PER_NEST * ROW_BYTES),
                    1, 0xFFFF, 0xBEEF);

                // Wait all SPUs done
                __credit_chk(0xFFFF);

                // Store result to DDR
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
                    uint8_t rows_for_tid = ROWS_PER_SPU + (tid < ROWS_REMAINDER ? 1 : 0);
                    uint32_t first_row = (uint32_t)tid * ROWS_PER_SPU + (tid < ROWS_REMAINDER ? tid : ROWS_REMAINDER);

                    for (uint8_t r = 0; r < rows_for_tid; r++) {
                        uint32_t row_off = (first_row + r) * ROW_BYTES;

                        // L2 -> L1: load row to Bank A.  Use credit-DMA on the
                        // final row to acknowledge the shared DDR->L2 handoff.
                        if (r == rows_for_tid - 1) {
                            __load_cr(L2_A + row_off, BANK_A,
                                ROW_BYTES, (uint16_t)ROW_BYTES, 1, (uint16_t)ROW_BYTES,
                                1, (uint16_t)(0x1u << tid), nest_id);
                        } else {
                            __load(L2_A + row_off, BANK_A,
                                ROW_BYTES, (uint16_t)ROW_BYTES, 1, (uint16_t)ROW_BYTES);
                        }

                        // Save original x, then form x^2 in R.
                        __copy(BANK_A, BANK_C, ROW_BYTES, ROW_BYTES, 1, ROW_BYTES);
                        __copy(BANK_A, BANK_B, ROW_BYTES, ROW_BYTES, 1, ROW_BYTES);
                        __mul_vv(WIDTH);

                        // Reduce sum(x^2) into SVR[0] using a vector of ones in Bank B.
                        __copy(BANK_R, BANK_A, ROW_BYTES, ROW_BYTES, 1, ROW_BYTES);
                        __wrspr(SVR_WORD_ADDR(1, 0), 0, FP16_ONE_PAIR, 0);
                        __wrspr(SVR_WORD_ADDR(1, 1), 0, FP16_ONE_PAIR, 0);
                        __wrspr(SVR_WORD_ADDR(1, 2), 0, FP16_ONE_PAIR, 0);
                        __wrspr(SVR_WORD_ADDR(1, 3), 0, FP16_ONE_PAIR, 0);
                        __store_svr(BANK_B, 1);
                        __dot_product(WIDTH, 0);

                        // Build a denominator vector: max(sqrt(sum(x^2)), eps).
                        __store_svr(BANK_A, 0);
                        for (uint8_t k = 1; k < WIDTH; k++) {
                            __copy(BANK_A, BANK_A + (uint32_t)k * DTYPE, 0, DTYPE, 1, 0);
                        }
                        __sqrt_v(WIDTH);
                        __copy(BANK_R, BANK_A, ROW_BYTES, ROW_BYTES, 1, ROW_BYTES);
                        __max_vs(WIDTH, FP16_EPS, 1, 0);
                        __store_svr(BANK_A, 1);
                        for (uint8_t k = 1; k < WIDTH; k++) {
                            __copy(BANK_A, BANK_A + (uint32_t)k * DTYPE, 0, DTYPE, 1, 0);
                        }
                        __copy(BANK_A, BANK_B, ROW_BYTES, ROW_BYTES, 1, ROW_BYTES);

                        // Divide original x by the per-row denominator.
                        __copy(BANK_C, BANK_A, ROW_BYTES, ROW_BYTES, 1, ROW_BYTES);
                        __div_vv(WIDTH);

                        // Store result L1 -> L2
                        if (r == rows_for_tid - 1) {
                            __store_cr(BANK_R, L2_RESULT + row_off, ROW_BYTES, (uint16_t)ROW_BYTES, 1, (uint16_t)ROW_BYTES, 1, 0x1 << tid);
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
