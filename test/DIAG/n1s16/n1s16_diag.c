//==================================================================
// Copyright   : (C) Supergate - All Rights Reserved
// Project     : GSF / VTS
// Test        : n1s16_diag (1 NEST x 16 SPUs = 16 SPUs)
// Description : Create diagonal matrix from vector using GTX DMA flow.
//
//               Input:  [N] FP16 diagonal values at 0x1000000
//               Output: [N][N] FP16 matrix at 0xf000000
//
//               One canonical split/plan loads the vector into L2, zeros the
//               result matrix in L2, lets 16 SPUs scatter diagonal values
//               through L1, then stores the complete matrix to DDR.
//
// Author      : sw.lee
//==================================================================

#ifndef N1S16_DIAG_C
#define N1S16_DIAG_C

#include <stdint.h>

#include "intrin.h"
#include "gtx/address.h"
#include "gtx_csr.h"

// Dimensions
#define NEST_ID             0
#define SPU_NUM_PER_NEST    16
#define N                   181
#define FP16_B              2
#define ROW_BYTES           (N * FP16_B)
#define INPUT_BYTES         (N * FP16_B)
#define OUTPUT_BYTES        (N * ROW_BYTES)
#define ELEMS_PER_SPU       (N / SPU_NUM_PER_NEST)
#define REM_ELEMS           (N % SPU_NUM_PER_NEST)
#define ACTIVE_TID_MASK     0xFFFFu

// DDR addresses
#define BASE_DDR_INPUT      0x1000000
#define BASE_DDR_RESULT     0xf000000

// L2 and L1 addresses
#define L2_A                0x000000
#define L2_RESULT           0x002000

#define BANK_A              0x00000
#define BANK_B              0x20000
#define BANK_C              0x30000
#define BANK_R              0x50000

int main(void) {
    __split();

    {
        __start_plan(NEST_ID);

            __start_shared();
                __fill(L2_RESULT, (uint32_t)OUTPUT_BYTES,
                    (uint16_t)OUTPUT_BYTES, 1, 0, 0);

                __load_cr(GTX_MAIN_ADDR(BASE_DDR_INPUT), L2_A,
                    (uint32_t)INPUT_BYTES,
                    (uint16_t)INPUT_BYTES,
                    1, (uint16_t)INPUT_BYTES,
                    1, ACTIVE_TID_MASK, 0xBEEF);

                __credit_chk(ACTIVE_TID_MASK);

                __store_cr(L2_RESULT, GTX_MAIN_ADDR(BASE_DDR_RESULT),
                    (uint32_t)OUTPUT_BYTES,
                    (uint16_t)OUTPUT_BYTES,
                    1, (uint16_t)OUTPUT_BYTES,
                    1, ACTIVE_TID_MASK);
            __end_shared();

            for (uint8_t tid = 0; tid < SPU_NUM_PER_NEST; tid++) {
                uint16_t tid_mask = (uint16_t)(0x1u << tid);
                uint32_t extra_before = (tid < REM_ELEMS) ? tid : REM_ELEMS;
                uint32_t elem_start = (uint32_t)tid * ELEMS_PER_SPU + extra_before;
                uint32_t elem_count = ELEMS_PER_SPU + ((tid < REM_ELEMS) ? 1u : 0u);

                __start_thread(tid);
                    __set_spm_addr(BANK_R, BANK_C, BANK_B, BANK_A);
                    __credit_chk(0xBEEF);

                    for (uint32_t e = 0; e < elem_count; e++) {
                        uint32_t idx = elem_start + e;
                        uint32_t src_off = idx * FP16_B;
                        uint32_t dst_off = idx * ROW_BYTES + idx * FP16_B;

                        if (e == elem_count - 1) {
                            __load_cr(L2_A + src_off, BANK_A,
                                FP16_B, (uint16_t)FP16_B,
                                1, (uint16_t)FP16_B,
                                1, 0xBEEF, 0xBEEF);
                        } else {
                            __load(L2_A + src_off, BANK_A,
                                FP16_B, (uint16_t)FP16_B,
                                1, (uint16_t)FP16_B);
                        }

                        if (e == elem_count - 1) {
                            __store_cr(BANK_A, L2_RESULT + dst_off,
                                FP16_B, (uint16_t)FP16_B,
                                1, (uint16_t)FP16_B,
                                1, tid_mask);
                        } else {
                            __store(BANK_A, L2_RESULT + dst_off,
                                FP16_B, (uint16_t)FP16_B,
                                1, (uint16_t)FP16_B);
                        }
                    }
                __end_thread(tid);
            }

        __end_plan(NEST_ID);
    }

    __join();

    return 0;
}

#endif
