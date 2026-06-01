//==================================================================
// Copyright   : (C) Supergate - All Rights Reserved
// Project     : GSF / VTS
// Test        : n1s16_mean
// Description : GTX MEAN with 1 NEST x 16 SPUs = 16 SPUs
//               Mean of each row: dst[i] = sum(src0[i][j]) / WIDTH
//
//               src0: [HEIGHT=64, WIDTH=8] row-major FP16 at 0x1000000
//               dst:  [64] FP16 scalars at 0xf000000 (128 bytes)
//
//               Same row-reduction semantics as ggml_mean: for each row,
//               reduce ne0 elements and scale by 1/WIDTH = 1/8
//               (FP16: 0x3000).
//
//               Parallelism: each SPU handles four rows.
//                 NEST 0: rows [0 .. 63]
//                 SPU tid: rows [tid*4 .. tid*4+3]
//                 Each SPU: __sum(row)          -> SVR[0]
//                           __mul_is(SVR[0], 1/8) -> SVR[1]
//                           store SVR[1] scalar
//
//               L2 layout:
//                 src0: 64 rows x 16B = 1024B at L2_A
//                 result: 64 x 2B = 128B at L2_RESULT
//
// Author      : sw.lee
// Last Update : 2026/03/04
//==================================================================

#ifndef N1S16_MEAN_C
#define N1S16_MEAN_C

#include "intrin.h"
#include "gtx/address.h"
#include "gtx_csr.h"

// Hardware
#define NEST_NUM            1
#define SPU_NUM_PER_NEST    16
#define ACTIVE_SPU_MASK     0xFFFF
#define ROWS_PER_NEST       (HEIGHT / NEST_NUM)      // 64 / 1
#define ROWS_PER_SPU        (ROWS_PER_NEST / SPU_NUM_PER_NEST)

// Tensor dimensions
#define WIDTH               16
#define HEIGHT              80
#define DTYPE               4       // FP16
#define ROW_BYTES           (WIDTH * DTYPE)             // 16
#define OUT_BYTES_PER_NEST  (ROWS_PER_NEST * DTYPE)     // 128

// DDR addresses
#define BASE_DDR_A          0x1000000   // src0 [64 x 8] FP16
#define BASE_DDR_RESULT     0xf000000   // dst [64] FP16

// L2 SPM addresses
#define L2_A                0x000000    // 64 rows x 16B = 1024B
#define L2_RESULT           0x002000    // 64 x 2B = 128B

// L1 SPM bank addresses
#define BANK_A              0x00000
#define BANK_B              0x20000
#define BANK_C              0x30000
#define BANK_R              0x50000

// SVR / FP16 constants
#define FP16_INV8           0x3D800000      // 1/16 = 0.0625 in FP16
#define FP16_B              2

// Stack (required for multi-NEST)

int main(void) {

    //=============================================================
    // Stack save/restore setup (required for multi-NEST)
    //=============================================================

    __split();

    {
        uint8_t nest_id = 0;

        __start_plan(nest_id);

            //=====================================================
            // Shared: DDR <-> L2
            //=====================================================
            __start_shared();

                uint32_t nest_off_in  = (uint32_t)nest_id * ROWS_PER_NEST * ROW_BYTES;
                uint32_t nest_off_out = (uint32_t)nest_id * OUT_BYTES_PER_NEST;

                // Load 64 rows of src0 [64 x 16B = 1024B]
                // Credit to all 16 SPUs when load completes
                __load_cr(
                    GTX_MAIN_ADDR(BASE_DDR_A) + nest_off_in, L2_A,
                    (uint32_t)(ROWS_PER_NEST * ROW_BYTES),
                    (uint16_t)(ROWS_PER_NEST * ROW_BYTES),
                    1, (uint16_t)(ROWS_PER_NEST * ROW_BYTES),
                    1, ACTIVE_SPU_MASK, 0xBEEF
                );

                // Wait for all SPUs to finish
                __credit_chk(ACTIVE_SPU_MASK);

                // Store 64 result scalars (128B) to DDR
                __store_cr(
                    L2_RESULT, GTX_MAIN_ADDR(BASE_DDR_RESULT) + nest_off_out,
                    (uint32_t)OUT_BYTES_PER_NEST,
                    (uint16_t)OUT_BYTES_PER_NEST,
                    1, (uint16_t)OUT_BYTES_PER_NEST,
                    1, ACTIVE_SPU_MASK
                );

            __end_shared();

            //=====================================================
            // Threads: Each SPU computes mean of one row
            //=====================================================
            for (uint8_t tid = 0; tid < SPU_NUM_PER_NEST; tid++) {
                uint16_t tid_mask = (uint16_t)(0x1u << tid);

                __start_thread(tid);
                    __set_spm_addr(BANK_R, BANK_C, BANK_B, BANK_A);
                    __credit_chk(0xBEEF);   // one-time shared DDR->L2 handoff

                    for (uint8_t r = 0; r < ROWS_PER_SPU; r++) {

                        uint32_t row_idx = (uint32_t)(tid * ROWS_PER_SPU + r);

                        // Load this SPU's row [WIDTH=8 x 2B = 16B] -> Bank A
                        if (r == ROWS_PER_SPU - 1) {
                            __load_cr(
                                L2_A + row_idx * ROW_BYTES,
                                BANK_A,
                                ROW_BYTES, (uint16_t)ROW_BYTES,
                                1, (uint16_t)ROW_BYTES,
                                1, tid_mask, nest_id
                            );
                        } else {
                            __load(
                                L2_A + row_idx * ROW_BYTES,
                                BANK_A,
                                ROW_BYTES, (uint16_t)ROW_BYTES,
                                1, (uint16_t)ROW_BYTES
                            );
                        }

                        // Sum this row into SVR[0].
                        __sum(WIDTH, 0);

                        // Multiply by 1/8 to get mean: SVR[1] = SVR[0] * (1/8)
                        __mul_is(0, FP16_INV8, 1, 0);

                        // Store SVR[1] to BANK_C (32B), then store first 2B scalar to L2
                        __store_svr(BANK_C, 1);

                        // Store 2B scalar to L2_RESULT at (tid * ROWS_PER_SPU + r) * 2
                        if (r == ROWS_PER_SPU - 1) {
                            __store_cr(BANK_C, L2_RESULT + row_idx * FP16_B, FP16_B, (uint16_t)FP16_B, 1, (uint16_t)FP16_B, 1, tid_mask);
                        } else {
                            __store(BANK_C, L2_RESULT + row_idx * FP16_B, FP16_B, (uint16_t)FP16_B, 1, (uint16_t)FP16_B);
                        }

                    }
                __end_thread(tid);
            }

        __end_plan(nest_id);
    }

    __join();

    return 0;
}

#endif
