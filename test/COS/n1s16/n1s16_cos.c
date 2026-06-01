//==================================================================
// n1s16_cos — element-wise cosine with 1 NEST x 16 SPUs
// dst[row] = cos(src0[row]), HEIGHT rows x WIDTH FP16 elements
//
// Uses Taylor series with range reduction:
//   1. Range reduce: n = round to nearest(x / (2*pi)), x_red = x - n * 2*pi
//   2. cos(x) ≈ 1 + x²*(-1/2 + x²*(1/24 + x²*(-1/720)))
//      Horner form: cos(x) = 1 + x²*(c1 + x²*(c2 + x²*c3))
//
// Each SPU processes 8 elements (< 16 SVR width), so one SVR chunk
// suffices per row. Load row into SVR[0], compute Taylor, store back.
//==================================================================

#include "intrin.h"
#include "gtx/address.h"
#include "gtx_csr.h"

#define NEST_NUM            1
#define SPU_NUM_PER_NEST    16
#define DTYPE               4       // FP16

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
#define ROWS_PER_SPU_BASE   (ROWS_PER_NEST / SPU_NUM_PER_NEST)
#define ROWS_REMAINDER      (ROWS_PER_NEST % SPU_NUM_PER_NEST)


// FP16 constants for range reduction
#define FP16_TWO_PI         0x40C90000  // 2*pi ≈ 6.2832
#define FP16_INV_2PI        0x3E230000  // 1/(2*pi) ≈ 0.15915

// FP16 Taylor coefficients for cosine (Horner form)
#define FP16_C0             0x3F800000  // 1.0
#define FP16_C1             0xBF000000  // -0.5
#define FP16_C2             0x3D2AA000  // 1/24 ≈ 0.04167
#define FP16_C3             0xBAB68000  // -1/720 ≈ -0.001389

int main(void) {


    __split();

    {
        uint8_t nest_id = 0;

        __start_plan(nest_id);

            __start_shared();
                uint32_t nest_off = (uint32_t)nest_id * ROWS_PER_NEST * ROW_BYTES;

                // Load src0 (with credit to all 16 SPUs)
                __load_cr(GTX_MAIN_ADDR(BASE_DDR_A) + nest_off, L2_A,
                    (uint32_t)(ROWS_PER_NEST * ROW_BYTES),
                    (uint16_t)(ROWS_PER_NEST * ROW_BYTES),
                    1, (uint16_t)(ROWS_PER_NEST * ROW_BYTES),
                    1, 0xFFFF, 0xBEEF);

                // Wait all SPUs done
                __credit_chk(0xFFFF);

                // Store result back to DDR
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
                    uint16_t rows_for_tid = (uint16_t)(ROWS_PER_SPU_BASE + (tid < ROWS_REMAINDER ? 1 : 0));
                    uint32_t row_start = (uint32_t)tid * ROWS_PER_SPU_BASE + (tid < ROWS_REMAINDER ? tid : ROWS_REMAINDER);
                    for (uint16_t r = 0; r < rows_for_tid; r++) {
                        uint32_t row_off = (uint32_t)(row_start + r) * ROW_BYTES;

                        // L2 -> L1: src0 row to Bank A
                        __load(L2_A + row_off, BANK_A,
                            ROW_BYTES, (uint16_t)ROW_BYTES, 1, (uint16_t)ROW_BYTES
);
                        if (r == rows_for_tid - 1) __credit_ld(tid, nest_id);

                        // Load 16 elements from Bank A into SVR[0]
                        // (only first 8 are valid, rest are garbage but harmless)
                        __load_svr(BANK_A, 0);              // SVR[0] = x

                        // Range reduction: n = round to nearest(x / (2*pi)), x_red = x - n * 2*pi
                        __mul_is(0, FP16_INV_2PI, 1, 0);   // SVR[1] = x / (2*pi)
                        __rne_i(1, 2);                      // SVR[2] = round to nearest(x/(2*pi))
                        __mul_is(2, FP16_TWO_PI, 3, 0);    // SVR[3] = n * 2*pi
                        __sub_ii(0, 3, 4);                  // SVR[4] = x_red

                        // Horner form: cos(x) = 1 + x²*(c1 + x²*(c2 + x²*c3))
                        // Step 1: x²
                        __mul_ii(4, 4, 5);                  // SVR[5] = x_red²

                        // Step 2: innermost — x²*c3
                        __mul_is(5, FP16_C3, 6, 0);        // SVR[6] = x² * c3

                        // Step 3: c2 + x²*c3
                        __add_is(6, FP16_C2, 7, 0);        // SVR[7] = c2 + x²*c3

                        // Step 4: x² * (c2 + x²*c3)
                        __mul_ii(5, 7, 8);                  // SVR[8] = x²*(c2 + x²*c3)

                        // Step 5: c1 + x²*(c2 + x²*c3)
                        __add_is(8, FP16_C1, 9, 0);        // SVR[9] = c1 + x²*(c2 + x²*c3)

                        // Step 6: x² * (c1 + x²*(c2 + x²*c3))
                        __mul_ii(5, 9, 10);                 // SVR[10] = x²*(c1 + ...)

                        // Step 7: 1 + x²*(c1 + ...) = cos(x_red)
                        __add_is(10, FP16_C0, 11, 0);      // SVR[11] = cos(x_red)

                        // Store result SVR to Bank R
                        __store_svr(BANK_R, 11);

                        // L1 -> L2: result from Bank R
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
