//==================================================================
// Copyright   : (C) Supergate - All Rights Reserved
// Project     : GSF / VTS
// Test        : n1s16_reglu
// Description : ReGLU (ReLU Gated Linear Unit) with
//               1 NEST x 16 SPUs = 16 SPUs
//
//               Input: 64 rows x 16 FP16 (WIDTH_IN=16) at 0x1000000
//               Each row: [gate(8) | value(8)]
//               Output: 64 rows x 8 FP16 (WIDTH_OUT=8) at 0xf000000
//
//               dst[row][i] = relu(gate[i]) * value[i]
//                           = max(0, gate[i]) * value[i]
//
//               Work partitioning:
//                 NEST i: rows i*16 .. i*16+15
//                 SPU j:  row i*16 + j (1 row per SPU)
//
//               Shared: load 16 input rows (512B) per NEST to L2_A
//               Thread: load 1 row (32B) to Bank A, compute ReGLU
//
//               Compute per SPU:
//                 1. Save gate (A[0..15]) -> C[0..15]
//                 2. Save value (A[16..31]) -> C[16..31]
//                 3. __clamp_min(WIDTH_OUT, 0x0000, 0) -> relu(gate) in R
//                    (__clamp_min reads A, writes R with threshold=0)
//                 4. Copy relu(gate) R -> A
//                 5. Restore value: C+16 -> B
//                 6. final = relu(gate) * value: A * B -> R
//                 7. Store R (16B) to L2_RESULT
//
// Author      : sw.lee
// Last Update : 2026/03/04
//==================================================================

#ifndef N1S16_REGLU_C
#define N1S16_REGLU_C

#include "intrin.h"
#include "gtx/address.h"
#include "gtx_csr.h"

// Hardware
#define NEST_NUM            1
#define SPU_NUM_PER_NEST    16
#define DTYPE               2       // FP16

// Dimensions
#define WIDTH_IN            32      // input elements per row (gate + value)
#define WIDTH_OUT           16      // output elements per row (half of input)
#define HEIGHT              128     // total rows

#define FP16_B              2
#define ROW_IN_BYTES        (WIDTH_IN * FP16_B)     // 32
#define ROW_OUT_BYTES       (WIDTH_OUT * FP16_B)    // 16
#define ROWS_PER_NEST       (HEIGHT / NEST_NUM)     // 64
#define ROWS_PER_SPU        (ROWS_PER_NEST / SPU_NUM_PER_NEST)

// DDR addresses
#define BASE_DDR_A          0x1000000   // input [64 x 16] FP16
#define BASE_DDR_RESULT     0xf000000   // output [64 x 8] FP16

// L2 SPM addresses
#define L2_A                0x000000    // input: 16 rows x 32B = 512B
#define L2_RESULT           0x002000    // result: 16 rows x 16B = 256B

// L1 SPM bank addresses
#define BANK_A              0x00000
#define BANK_B              0x20000
#define BANK_C              0x30000
#define BANK_R              0x50000

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

                // Input offset for this NEST
                uint32_t in_nest_off = (uint32_t)nest_id * ROWS_PER_NEST * ROW_IN_BYTES;

                // Load 16 input rows (512B) to L2_A with credit
                __load_cr(
                    GTX_MAIN_ADDR(BASE_DDR_A) + in_nest_off, L2_A,
                    (uint32_t)(ROWS_PER_NEST * ROW_IN_BYTES),
                    (uint16_t)(ROWS_PER_NEST * ROW_IN_BYTES),
                    1,
                    (uint16_t)(ROWS_PER_NEST * ROW_IN_BYTES),
                    1, 0xFFFF, 0xBEEF   // credit to all 16 SPUs
                );

                // Wait for all SPUs to finish
                __credit_chk(0xFFFF);

                // Store result (256B) to DDR
                uint32_t out_nest_off = (uint32_t)nest_id * ROWS_PER_NEST * ROW_OUT_BYTES;
                __store_cr(
                    L2_RESULT, GTX_MAIN_ADDR(BASE_DDR_RESULT) + out_nest_off,
                    (uint32_t)(ROWS_PER_NEST * ROW_OUT_BYTES),
                    (uint16_t)(ROWS_PER_NEST * ROW_OUT_BYTES),
                    1,
                    (uint16_t)(ROWS_PER_NEST * ROW_OUT_BYTES),
                    1, 0xFFFF
                );

            __end_shared();

            //=====================================================
            // Threads: L2 <-> L1, compute ReGLU
            //=====================================================
            for (uint8_t tid = 0; tid < SPU_NUM_PER_NEST; tid++) {

                __start_thread(tid);
                    for (uint8_t r = 0; r < ROWS_PER_SPU; r++) {

                        __set_spm_addr(BANK_R, BANK_C, BANK_B, BANK_A);
                        __credit_chk(0xBEEF);   // wait for DDR->L2 load

                        // Load this SPU's input row (32B) from L2_A -> Bank A
                        // Row contains [gate(16B) | value(16B)]
                        __load(
                            L2_A + (uint32_t)(tid * ROWS_PER_SPU + r) * ROW_IN_BYTES,
                            BANK_A,
                            ROW_IN_BYTES, (uint16_t)ROW_IN_BYTES,
                            1, (uint16_t)ROW_IN_BYTES
);
                        if (r == ROWS_PER_SPU - 1) __credit_ld((uint32_t)(1u << tid), (uint32_t)(1u << nest_id));

                        //=================================================
                        // ReGLU computation:
                        // dst[i] = relu(gate[i]) * value[i]
                        //        = max(0, gate[i]) * value[i]
                        //=================================================

                        // Step 1: Save gate (first 16B) to Bank C
                        __copy(BANK_A, BANK_C,
                            ROW_OUT_BYTES, ROW_OUT_BYTES, 1, ROW_OUT_BYTES);

                        // Step 2: Save value (last 16B) to Bank C + ROW_OUT_BYTES
                        __copy(BANK_A + ROW_OUT_BYTES, BANK_C + ROW_OUT_BYTES,
                            ROW_OUT_BYTES, ROW_OUT_BYTES, 1, ROW_OUT_BYTES);

                        // Step 3: relu(gate) = clamp_min(gate, 0)
                        // __clamp_min reads A, writes R; threshold=0 (FP16 0x0000)
                        __clamp_min(WIDTH_OUT, 0x0000, 0);
                        // Now: R = relu(gate) = max(0, gate)

                        // Step 4: Copy relu(gate) R -> A
                        __copy(BANK_R, BANK_A,
                            ROW_OUT_BYTES, ROW_OUT_BYTES, 1, ROW_OUT_BYTES);

                        // Step 5: Restore value: C + ROW_OUT_BYTES -> B
                        __copy(BANK_C + ROW_OUT_BYTES, BANK_B,
                            ROW_OUT_BYTES, ROW_OUT_BYTES, 1, ROW_OUT_BYTES);

                        // Step 6: final = relu(gate) * value: A * B -> R
                        __mul_vv(WIDTH_OUT);

                        // Step 7: Store result from Bank R (16B) -> L2_RESULT
                        if (r == ROWS_PER_SPU - 1) {
                            __store_cr(BANK_R, L2_RESULT + (uint32_t)(tid * ROWS_PER_SPU + r) * ROW_OUT_BYTES, ROW_OUT_BYTES, (uint16_t)ROW_OUT_BYTES, 1, (uint16_t)ROW_OUT_BYTES, 1, 0x1 << tid);
                        } else {
                            __store(BANK_R, L2_RESULT + (uint32_t)(tid * ROWS_PER_SPU + r) * ROW_OUT_BYTES, ROW_OUT_BYTES, (uint16_t)ROW_OUT_BYTES, 1, (uint16_t)ROW_OUT_BYTES);
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
