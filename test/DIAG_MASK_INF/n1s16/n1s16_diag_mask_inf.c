//==================================================================
// Copyright   : (C) Supergate - All Rights Reserved
// Project     : GSF / VTS
// Test        : n1s16_diag_mask_inf (1 NEST x 16 SPUs = 16 SPUs)
// Description : Causal mask with -inf (upper triangular masking)
//
//               src:  [64][8] FP16 at 0x1000000 (1024B)
//               dst:  [64][8] FP16 at 0xf000000 (1024B)
//
//               For row i, column j:
//                 if j > i:  dst[i][j] = -inf (FP16 0xFC00)
//                 else:      dst[i][j] = src[i][j]
//
//               WIDTH=8, so for rows i >= 7, NO masking needed
//               (all j in 0..7 satisfy j <= i).
//               For rows i < 7: mask elements at indices j > i.
//
//               Parallelization: each SPU handles 1 row.
//               NEST i -> rows [i*16 .. i*16+15]
//               SPU tid -> row = i*16 + tid
//
//               Strategy:
//               - CPU creates a mask_buf[8][8] at 0x3000000 with
//                 precomputed masked rows (rows 0-7 only need masking).
//               - Rows 0-7: load from mask_buf (CPU pre-applied mask)
//               - Rows 8-63: copy unchanged from src.
//
//               Actually simpler: CPU pre-applies the mask to all 64 rows
//               in DDR, then GTX just copies the result. But that defeats
//               the purpose of using the accelerator.
//
//               Better approach: each SPU loads its row, and for rows
//               where masking is needed, overwrites the tail elements
//               with -inf using __fill + __copy.
//
//               For row i (global_tid):
//                 num_keep = min(i + 1, WIDTH)
//                 num_mask = WIDTH - num_keep
//                 If num_mask > 0: fill last num_mask elements with -inf
//
// Author      : sw.lee
//==================================================================

#ifndef N1S16_DIAG_MASK_INF_C
#define N1S16_DIAG_MASK_INF_C

#include "intrin.h"
#include "gtx/address.h"
#include "gtx_csr.h"

// Hardware
#define NEST_NUM            1
#define SPU_NUM_PER_NEST    16
#define ROWS_PER_NEST       (HEIGHT / NEST_NUM)
#define ROWS_PER_SPU_BASE   (ROWS_PER_NEST / SPU_NUM_PER_NEST)
#define ROWS_PER_SPU_REM    (ROWS_PER_NEST % SPU_NUM_PER_NEST)

// Dimensions
#define WIDTH               32
#define HEIGHT              128

// DDR addresses
#define BASE_DDR_A          0x1000000   // src[64][8] FP16 = 1024B
#define BASE_DDR_RESULT     0xf000000   // dst[64][8] FP16 = 1024B

// L2 SPM addresses
#define L2_A                0x000000    // WIDTH x HEIGHT input tile in L2
#define L2_RESULT           0x008000    // separate result tile; avoids aliasing L2_A for larger verifier shapes

// L1 SPM bank addresses
#define BANK_A              0x00000
#define BANK_B              0x20000
#define BANK_C              0x30000
#define BANK_R              0x50000

#define FP16_B              2
#define SVR_BYTES           32

#define ROW_BYTES           (WIDTH * FP16_B)                // 16
#define NEST_BYTES          (ROWS_PER_NEST * ROW_BYTES)     // 256

// FP16 -inf = 0xFC00
#define FP16_NEG_INF        0xFC00

// Stack (required for multi-NEST)

// SVR address base
#define SVR_ADDR            0x800

int main(void) {

    //=============================================================
    // Stack save/restore setup
    //=============================================================

    __split();

    {
        uint8_t nest_id = 0;

        __start_plan(nest_id);

            //=========================================================
            // Shared: DDR <-> L2
            //=========================================================
            __start_shared();

                uint32_t nest_off = (uint32_t)nest_id * NEST_BYTES;

                // Load src rows for this NEST [16 rows x 16B = 256B]
                __load_cr(
                    GTX_MAIN_ADDR(BASE_DDR_A) + nest_off, L2_A,
                    (uint32_t)NEST_BYTES,
                    (uint16_t)NEST_BYTES,
                    1, (uint16_t)NEST_BYTES,
                    1, 0xFFFF, 0xBEEF
                );

                // Wait for all SPUs to finish
                __credit_chk(0xFFFF);

                // Store result to DDR [256B]
                __store_cr(
                    L2_RESULT, GTX_MAIN_ADDR(BASE_DDR_RESULT) + nest_off,
                    (uint32_t)NEST_BYTES,
                    (uint16_t)NEST_BYTES,
                    1, (uint16_t)NEST_BYTES,
                    1, 0xFFFF
                );

            __end_shared();

            //=========================================================
            // Threads: apply causal mask
            //=========================================================
            for (uint8_t tid = 0; tid < SPU_NUM_PER_NEST; tid++) {
                uint16_t tid_mask = (uint16_t)(0x1u << tid);

                __start_thread(tid);
                    __set_spm_addr(BANK_R, BANK_C, BANK_B, BANK_A);
                    __credit_chk(0xBEEF);   // wait for L2 load
                    uint8_t rows_for_tid = (uint8_t)(ROWS_PER_SPU_BASE + ((tid < ROWS_PER_SPU_REM) ? 1 : 0));
                    uint16_t start_row = (uint16_t)(tid * ROWS_PER_SPU_BASE + ((tid < ROWS_PER_SPU_REM) ? tid : ROWS_PER_SPU_REM));

                    for (uint8_t r = 0; r < rows_for_tid; r++) {

                        uint16_t global_row = (uint16_t)(start_row + r);
                        uint32_t row_off = (uint32_t)global_row * ROW_BYTES;

                        // Load row from L2 to Bank A
                        if (r == rows_for_tid - 1) {
                            __load_cr(
                                L2_A + row_off, BANK_A,
                                ROW_BYTES, (uint16_t)ROW_BYTES,
                                1, (uint16_t)ROW_BYTES,
                                1, tid_mask, nest_id
                            );
                        } else {
                            __load(
                                L2_A + row_off, BANK_A,
                                ROW_BYTES, (uint16_t)ROW_BYTES,
                                1, (uint16_t)ROW_BYTES
                            );
                        }

                        // Copy row to Bank R (start with original data)
                        __copy(
                            BANK_A, BANK_R,
                            0, ROW_BYTES, 1, 0
                        );

                        // Apply mask: if global_row < WIDTH, overwrite tail with -inf
                        // num_keep = global_row + 1 (for row 0: keep 1, mask 7)
                        // num_mask = WIDTH - num_keep = 7 - global_row
                        // mask_offset = num_keep * FP16_B
                        // mask_bytes = num_mask * FP16_B
                        if (global_row + 1 < WIDTH) {
                            uint16_t num_keep = global_row + 1;
                            uint16_t num_mask = WIDTH - num_keep;
                            uint32_t mask_offset = (uint32_t)num_keep * FP16_B;
                            uint16_t mask_bytes = num_mask * FP16_B;

                            // Fill -inf pattern into Bank C
                            // NOTE: __fill is shared-only; use __wrspr+__store_svr in thread
                            uint64_t neginf_word =
                                ((uint64_t)FP16_NEG_INF << 48) |
                                ((uint64_t)FP16_NEG_INF << 32) |
                                ((uint64_t)FP16_NEG_INF << 16) |
                                ((uint64_t)FP16_NEG_INF);
                            __wrspr(SVR_ADDR + 0*4,     0, neginf_word, 0);
                            __wrspr(SVR_ADDR + 0*4 + 1, 0, neginf_word, 0);
                            __wrspr(SVR_ADDR + 0*4 + 2, 0, neginf_word, 0);
                            __wrspr(SVR_ADDR + 0*4 + 3, 0, neginf_word, 0);
                            __store_svr(BANK_C, 0);              // first 32B of -inf
                            __store_svr(BANK_C + SVR_BYTES, 0);  // second 32B covers WIDTH=31 tail

                            // Copy -inf values to the tail of the row in Bank R
                            __copy(
                                BANK_C,
                                BANK_R + mask_offset,
                                0, mask_bytes, 1, 0
                            );
                        }

                        // Store result row (16B) from L1 to L2
                        if (r == rows_for_tid - 1) {
                            __store_cr(BANK_R, L2_RESULT + row_off, ROW_BYTES, (uint16_t)ROW_BYTES, 1, (uint16_t)ROW_BYTES, 1, tid_mask);
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

#endif
