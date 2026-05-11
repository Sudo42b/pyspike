//==================================================================
// Copyright   : (C) Supergate - All Rights Reserved
// Project     : GSF / VTS
// Test        : n1s16_out_prod (1 NEST x 16 SPUs = 16 SPUs)
// Description : OUT_PROD for generated shape [32,1]
//
//               ggml tensors:
//                 src0: [32, 1] FP16 at 0x1000000 (64B)
//                 src1: [16, 1] FP16 at 0x2000000 (32B)
//                 dst : [32,16] FP16 at 0xf000000 (1024B)
//
//               dst[i0 + i1*32] = src0[i0] * src1[i1]
//
//               Parallelization: each SPU handles one output column i1.
//
//               Each SPU loads src0 in two 16-element chunks and its own
//               src1 scalar through GTX DDR->L2->L1/SVR dataflow.  The src1
//               scalar is broadcast into a 16-lane SVR by a strided L2->L1
//               load, then multiplied with the src0 chunks to form one
//               contiguous 32-element dst column in L2.
//
// Author      : sw.lee
//==================================================================

#ifndef N1S16_OUT_PROD_C
#define N1S16_OUT_PROD_C

#include "intrin.h"
#include "gtx/address.h"
#include "gtx_csr.h"

// Hardware
#define NEST_NUM            1
#define SPU_NUM_PER_NEST    16

// Dimensions
#define A_LEN               32      // dst ne0 / src0 ne0
#define B_LEN               16      // dst ne1 / src1 ne0
#define FP16_B              2
#define SVR_BYTES           32
#define A_CHUNK_ELEMS       16
#define A_CHUNK_BYTES       (A_CHUNK_ELEMS * FP16_B)

// DDR addresses
#define BASE_DDR_A          0x1000000   // src0[32] FP16 = 64B
#define BASE_DDR_B          0x2000000   // src1[16] FP16 = 32B
#define BASE_DDR_RESULT     0xf000000   // dst[32][16] FP16 = 1024B

// L2 SPM addresses
#define L2_A                0x000000    // src0[32] = 64B (shared)
#define L2_B                0x001000    // src1[16] = 32B (shared)
#define L2_RESULT           0x002000    // dst[32][16] = 1024B

// L1 SPM bank addresses
#define BANK_A              0x00000
#define BANK_B              0x20000
#define BANK_C              0x30000
#define BANK_R              0x50000

// Byte sizes
#define A_BYTES             (A_LEN * FP16_B)            // 64
#define B_BYTES             (B_LEN * FP16_B)            // 32
#define COL_BYTES           A_BYTES                     // one dst column
#define RESULT_BYTES        (A_LEN * B_LEN * FP16_B)    // 1024

int main(void) {
    //=============================================================
    // Stack save/restore setup (required for multi-NEST)
    //=============================================================

    __split();

    {
        uint8_t nest_id = 0;

        __start_plan(nest_id);

            //=========================================================
            // Shared: DDR <-> L2
            //=========================================================
            __start_shared();

                // Load src1[16] (32B) to L2 before issuing the shared data-ready credit.
                __load(
                    GTX_MAIN_ADDR(BASE_DDR_B), L2_B,
                    (uint32_t)B_BYTES, (uint16_t)B_BYTES,
                    1, (uint16_t)B_BYTES
                );

                // Load src0[32] (64B) to L2 with credit to all 16 SPUs.
                __load_cr(
                    GTX_MAIN_ADDR(BASE_DDR_A), L2_A,
                    (uint32_t)A_BYTES, (uint16_t)A_BYTES,
                    1, (uint16_t)A_BYTES,
                    1, 0xFFFF, 0xBEEF
                );

                // Wait for all SPUs to finish
                __credit_chk(0xFFFF);

                // Store full dst[32,16] result (1024B)
                __store_cr(
                    L2_RESULT, GTX_MAIN_ADDR(BASE_DDR_RESULT),
                    (uint32_t)RESULT_BYTES,
                    (uint16_t)RESULT_BYTES,
                    1, (uint16_t)RESULT_BYTES,
                    1, 0xFFFF
                );

            __end_shared();

            //=========================================================
            // Threads: compute output columns
            //=========================================================
            for (uint8_t tid = 0; tid < SPU_NUM_PER_NEST; tid++) {

                __start_thread(tid);
                    __set_spm_addr(BANK_R, BANK_C, BANK_B, BANK_A);
                    __credit_chk(0xBEEF);   // wait for L2 load

                    uint32_t col_off = (uint32_t)tid * COL_BYTES;

                    // Broadcast this SPU's src1 scalar into all 16 FP16 lanes of BANK_B.
                    __load(L2_B + (uint32_t)tid * FP16_B, BANK_B, 0, (uint16_t)FP16_B, A_CHUNK_ELEMS, (uint16_t)FP16_B);
                    __load_svr(BANK_B, 1);

                    // First 16 src0 elements -> first half of dst column.
                    __load(L2_A, BANK_A, A_CHUNK_BYTES, (uint16_t)A_CHUNK_BYTES, 1, (uint16_t)A_CHUNK_BYTES);
                    __load_svr(BANK_A, 0);
                    __mul_ii(0, 1, 2);
                    __store_svr(BANK_R, 2);
                    __store(BANK_R, L2_RESULT + col_off, A_CHUNK_BYTES, (uint16_t)A_CHUNK_BYTES, 1, (uint16_t)A_CHUNK_BYTES);

                    // Last 16 src0 elements -> second half of dst column.
                    __load(L2_A + A_CHUNK_BYTES, BANK_A, A_CHUNK_BYTES, (uint16_t)A_CHUNK_BYTES, 1, (uint16_t)A_CHUNK_BYTES);
                    __credit_ld(0xBEEF, 0xBEEF);
                    __load_svr(BANK_A, 0);
                    __mul_ii(0, 1, 2);
                    __store_svr(BANK_R, 2);
                    __store_cr(BANK_R, L2_RESULT + col_off + A_CHUNK_BYTES, A_CHUNK_BYTES, (uint16_t)A_CHUNK_BYTES, 1, (uint16_t)A_CHUNK_BYTES, 1, (1u << tid));

                __end_thread(tid);
            }

        __end_plan(nest_id);
    }

    __join();

    return 0;
}

#endif
