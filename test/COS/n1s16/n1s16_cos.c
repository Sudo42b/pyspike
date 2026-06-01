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


// Range reduction to [-pi/2, pi/2] (mirrors n1s16_sin): n = round(x/pi),
// x_red = x - n*pi, then cos(x) = (-1)^n * cos(x_red). The 4-term Taylor is
// only accurate near 0, so reducing to [-pi/2,pi/2] (not [-pi,pi]) is required.
// Full FP32 precision (not fp16-widened) — a coarse pi propagates ~1e-3 through
// the range-reduction x - n*pi, which alone exceeds tolerance near x = ±pi/2.
#define FP16_PI             0x40490FDB  // pi   = 3.14159274
#define FP16_INV_PI         0x3EA2F983  // 1/pi = 0.318309873
#define FP16_HALF           0x3F000000  // 0.5
#define FP16_TWO            0x40000000  // 2.0

// FP32 Taylor coefficients for cosine (Horner form, even powers)
#define FP16_C0             0x3F800000  // 1.0
#define FP16_C1             0xBF000000  // -0.5
#define FP16_C2             0x3D2AAAAB  // 1/24    = 0.0416666679
#define FP16_C3             0xBAB60B61  // -1/720  = -0.00138888892
#define FP16_C4             0x37D00D01  // 1/40320 = 2.48015876e-5 (5th term: z=pi/2 err 9e-4→2.5e-5)

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

                        // Range reduction to [-pi/2, pi/2]: n = round(x/pi), x_red = x - n*pi
                        __mul_is(0, FP16_INV_PI, 1, 0);     // SVR[1] = x / pi
                        __rne_i(1, 1);                      // SVR[1] = n = round(x/pi)
                        __mul_is(1, FP16_PI, 2, 0);         // SVR[2] = n * pi
                        __sub_ii(0, 2, 3);                  // SVR[3] = x_red ∈ [-pi/2, pi/2]

                        // Sign correction: cos(x) = (-1)^n * cos(x_red)
                        // (-1)^n via parity of n: n - 2*ceil(n/2) ∈ {0,-1}, then *2+1.
                        __mul_is(1, FP16_HALF, 4, 0);       // SVR[4] = n/2
                        __ceil_i(4, 4);                     // SVR[4] = ceil(n/2)
                        __mul_is(4, FP16_TWO, 4, 0);        // SVR[4] = 2*ceil(n/2)
                        __sub_ii(1, 4, 4);                  // SVR[4] = n - 2*ceil(n/2) ∈ {0,-1}
                        __fmadd_iss(4, FP16_TWO, FP16_C0, 4, 0); // SVR[4] = (-1)^n (C0 = 1.0)

                        // Horner on x_red: cos(z)=1+z²*(c1+z²*(c2+z²*(c3+z²*c4)))
                        __mul_ii(3, 3, 5);                  // SVR[5] = z²
                        __mul_is(5, FP16_C4, 6, 0);         // SVR[6] = z²*c4
                        __add_is(6, FP16_C3, 6, 0);         // SVR[6] = c3 + z²*c4
                        __mul_ii(6, 5, 6);                  // SVR[6] = z²*(c3 + z²*c4)
                        __add_is(6, FP16_C2, 6, 0);         // SVR[6] = c2 + z²*(...)
                        __mul_ii(6, 5, 6);                  // SVR[6] = z²*(c2 + ...)
                        __add_is(6, FP16_C1, 6, 0);         // SVR[6] = c1 + z²*(...)
                        __mul_ii(6, 5, 6);                  // SVR[6] = z²*(c1 + ...)
                        __add_is(6, FP16_C0, 6, 0);         // SVR[6] = cos(z)
                        __mul_ii(6, 4, 6);                  // SVR[6] = (-1)^n * cos(z) = cos(x)

                        // Store result SVR to Bank R
                        __store_svr(BANK_R, 6);

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
