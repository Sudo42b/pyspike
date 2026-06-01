//==================================================================
// Copyright   : (C) Supergate - All Rights Reserved
// Project     : GSF / VTS
// Test        : n1s16_geglu_quick
// Description : GEGLU_QUICK (Quick-GELU Gated Linear Unit) with
//               1 NEST x 16 SPUs = 16 SPUs
//
//               Input: 1024 rows x 256 FP16 (WIDTH_IN=256) at 0x1000000
//               Each row: [gate(128) | value(128)]
//               Output: 1024 rows x 128 FP16 (WIDTH_OUT=128) at 0xf000000
//
//               dst[row][i] = gelu_quick(gate[i]) * value[i]
//
//               gelu_quick(x) = x * sigmoid(1.702 * x)
//               1.702 in FP16 = 0x3ED1
//
//               Work partitioning:
//                 NEST i: its contiguous row block
//                 SPU j:  row block partitioned with remainder rows covered
//
//               Shared: load input rows to L2_A with 2D row transfers
//               Thread: load 1 row to Bank A, compute GEGLU_QUICK
//
//               Compute per SPU:
//                 1. Save x/gate (A[0..15]) -> C[0..15]
//                 2. Save g/value (A[16..31]) -> C[16..31]
//                 3. __mul_vs(WIDTH_OUT, 0x3ED1) -> 1.702*x in R  (A -> R)
//                 4. __sigm(WIDTH_OUT) -> sigmoid(1.702*x) in A  (R -> A)
//                 5. Copy sigmoid A -> B
//                 6. Restore gate: C -> A
//                 7. x * sigmoid(1.702*x): A * B -> R = gelu_quick(x)
//                 8. Copy gelu_quick R -> A
//                 9. Restore value: C+16 -> B
//                 10. final = gelu_quick(gate) * value: A * B -> R
//                 11. Store R (ROW_OUT_BYTES) to L2_RESULT
//
// Author      : sw.lee
// Last Update : 2026/03/04
//==================================================================

#ifndef N1S16_GEGLU_QUICK_C
#define N1S16_GEGLU_QUICK_C

#include "intrin.h"
#include "gtx/address.h"
#include "gtx_csr.h"

// Hardware
#define NEST_NUM            1
#define SPU_NUM_PER_NEST    16
#define DTYPE               4       // FP16

// Dimensions
#define WIDTH_IN            256     // input elements per row (gate + value)
#define WIDTH_OUT           128     // output elements per row (half of input)
#define HEIGHT              1024    // total rows

#define FP16_B              2
#define ROW_IN_BYTES        (WIDTH_IN * FP16_B)
#define ROW_OUT_BYTES       (WIDTH_OUT * FP16_B)
#define ROWS_PER_NEST       (HEIGHT / NEST_NUM)     // 64
#define ROWS_PER_SPU_BASE   (ROWS_PER_NEST / SPU_NUM_PER_NEST)
#define ROWS_PER_SPU_REM    (ROWS_PER_NEST % SPU_NUM_PER_NEST)

// DDR addresses
#define BASE_DDR_A          0x1000000   // input [HEIGHT x WIDTH_IN] FP16
#define BASE_DDR_RESULT     0xf000000   // output [HEIGHT x WIDTH_OUT] FP16

// L2 SPM addresses
#define L2_A                0x000000
#define L2_ALIGN            0x001000
#define L2_RESULT           ((L2_A + (ROWS_PER_NEST * ROW_IN_BYTES) + (L2_ALIGN - 1)) & ~(L2_ALIGN - 1))

// L1 SPM bank addresses
#define BANK_A              0x00000
#define BANK_B              0x20000
#define BANK_C              0x30000
#define BANK_R              0x50000

// Stack (required for multi-NEST)

// FP16 constant: 1.702 = 0x3ED1
#define FP16_1_702          0x3FDA2000

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

                // Input offset for this NEST
                uint32_t in_nest_off = (uint32_t)nest_id * ROWS_PER_NEST * ROW_IN_BYTES;

                // Load input rows to L2_A with credit.  Use one DMA row per
                // tensor row so the 16-bit length/stride fields stay bounded
                // when verifier retargets the static dimensions to larger
                // shapes such as [256,1024].
                __load_cr(
                    GTX_MAIN_ADDR(BASE_DDR_A) + in_nest_off, L2_A,
                    (uint32_t)ROW_IN_BYTES,
                    (uint16_t)ROW_IN_BYTES,
                    (uint16_t)ROWS_PER_NEST,
                    (uint16_t)ROW_IN_BYTES,
                    1, 0xFFFF, 0xBEEF   // credit to all 16 SPUs
                );

                // Wait for all SPUs to finish
                __credit_chk(0xFFFF);

                // Store result rows to DDR with the same bounded 2D row
                // transfer strategy used for the input load.
                uint32_t out_nest_off = (uint32_t)nest_id * ROWS_PER_NEST * ROW_OUT_BYTES;
                __store_cr(
                    L2_RESULT, GTX_MAIN_ADDR(BASE_DDR_RESULT) + out_nest_off,
                    (uint32_t)ROW_OUT_BYTES,
                    (uint16_t)ROW_OUT_BYTES,
                    (uint16_t)ROWS_PER_NEST,
                    (uint16_t)ROW_OUT_BYTES,
                    1, 0xFFFF
                );

            __end_shared();

            //=====================================================
            // Threads: L2 <-> L1, compute GEGLU_QUICK
            //=====================================================
            for (uint8_t tid = 0; tid < SPU_NUM_PER_NEST; tid++) {

                __start_thread(tid);
                        __set_spm_addr(BANK_R, BANK_C, BANK_B, BANK_A);
                        __credit_chk(0xBEEF);   // wait once for DDR->L2 load credit

                    uint16_t rows_this_spu = (uint16_t)ROWS_PER_SPU_BASE + (tid < ROWS_PER_SPU_REM ? 1 : 0);
                    uint16_t row_start = (uint16_t)tid * (uint16_t)ROWS_PER_SPU_BASE + (tid < ROWS_PER_SPU_REM ? tid : ROWS_PER_SPU_REM);

                    if (rows_this_spu == 0) {
                        __credit_ld((uint32_t)(1u << tid), (uint32_t)(1u << nest_id));
                        __credit_st((uint32_t)(1u << tid));
                    }

                    for (uint16_t r = 0; r < rows_this_spu; r++) {

                        uint32_t row_index = (uint32_t)row_start + (uint32_t)r;

                        // Load this SPU's input row from L2_A -> Bank A.
                        // Row contains [gate(ROW_OUT_BYTES) | value(ROW_OUT_BYTES)]
                        __load(
                            L2_A + row_index * ROW_IN_BYTES,
                            BANK_A,
                            ROW_IN_BYTES, (uint16_t)ROW_IN_BYTES,
                            1, (uint16_t)ROW_IN_BYTES
);
                        if (r == rows_this_spu - 1) __credit_ld((uint32_t)(1u << tid), (uint32_t)(1u << nest_id));

                        //=================================================
                        // GEGLU_QUICK computation:
                        // dst[i] = gelu_quick(x[i]) * g[i]
                        //
                        // gelu_quick(x) = x * sigmoid(1.702 * x)
                        // 1.702 in FP16 = 0x3ED1
                        //=================================================

                        // Step 1: Save x/gate (first 16B) to Bank C
                        __copy(BANK_A, BANK_C,
                            ROW_OUT_BYTES, ROW_OUT_BYTES, 1, ROW_OUT_BYTES);

                        // Step 2: Save g/value (last 16B) to Bank C + ROW_OUT_BYTES
                        __copy(BANK_A + ROW_OUT_BYTES, BANK_C + ROW_OUT_BYTES,
                            ROW_OUT_BYTES, ROW_OUT_BYTES, 1, ROW_OUT_BYTES);

                        // Step 3: 1.702 * x
                        // __mul_vs reads A, writes R: R = A * scalar
                        __mul_vs(WIDTH_OUT, FP16_1_702, 0);
                        // Now: R = 1.702 * x

                        // Step 4: sigmoid(1.702 * x)
                        // __sigm reads from Bank R -> writes Bank A
                        __sigm(WIDTH_OUT);
                        // Now: A = sigmoid(1.702 * gate)

                        // Step 5: Copy sigmoid result A -> B
                        __copy(BANK_A, BANK_B,
                            ROW_OUT_BYTES, ROW_OUT_BYTES, 1, ROW_OUT_BYTES);

                        // Step 6: Restore x/gate: C -> A
                        __copy(BANK_C, BANK_A,
                            ROW_OUT_BYTES, ROW_OUT_BYTES, 1, ROW_OUT_BYTES);

                        // Step 7: gelu_quick = x * sigmoid(1.702*x): A * B -> R
                        __mul_vv(WIDTH_OUT);
                        // Now: R = gelu_quick(gate)

                        // Step 8: Copy gelu_quick result R -> A
                        __copy(BANK_R, BANK_A,
                            ROW_OUT_BYTES, ROW_OUT_BYTES, 1, ROW_OUT_BYTES);

                        // Step 9: Restore g/value: C + ROW_OUT_BYTES -> B
                        __copy(BANK_C + ROW_OUT_BYTES, BANK_B,
                            ROW_OUT_BYTES, ROW_OUT_BYTES, 1, ROW_OUT_BYTES);

                        // Step 10: final = gelu_quick(gate) * value: A * B -> R
                        __mul_vv(WIDTH_OUT);

                        // Step 11: Store result from Bank R -> L2_RESULT
                        if (r == rows_this_spu - 1) {
                            __store_cr(BANK_R, L2_RESULT + row_index * ROW_OUT_BYTES, ROW_OUT_BYTES, (uint16_t)ROW_OUT_BYTES, 1, (uint16_t)ROW_OUT_BYTES, 1, 0x1 << tid);
                        } else {
                            __store(BANK_R, L2_RESULT + row_index * ROW_OUT_BYTES, ROW_OUT_BYTES, (uint16_t)ROW_OUT_BYTES, 1, (uint16_t)ROW_OUT_BYTES);
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
