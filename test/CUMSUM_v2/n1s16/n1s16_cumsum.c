//==================================================================
// Copyright   : (C) Supergate - All Rights Reserved
// Project     : GSF / VTS
// Test        : n1s16_cumsum (1 NEST x 16 SPUs = 16 SPUs)
// Description : Cumulative sum per row using shared/thread DMA pattern.
//
//               src: [256][64] FP16 at 0x1000000 (32768B)
//               dst: [256][64] FP16 at 0xf000000 (32768B)
//
//               For each row r in 0..255:
//                 dst[r][0] = src[r][0]
//                 dst[r][c] = dst[r][c-1] + src[r][c]  (c = 1..63)
//
//               Prefix sum is computed by a Hillis-Steele scan in L1.
//               Uses shared/thread DMA pattern:
//                 Shared: DDR -> L2 bulk load
//                 Thread: Each SPU loads its rows L2 -> L1,
//                         performs prefix sum in L1 via __add_vv,
//                         stores L1 -> L2 result
//                 Shared: L2 -> DDR bulk store
//
// Author      : sw.lee
//==================================================================

#ifndef N1S16_CUMSUM_C
#define N1S16_CUMSUM_C

#include <stdint.h>

#include "intrin.h"
#include "gtx/address.h"
#include "gtx_csr.h"

#define NEST_NUM            1
#define SPU_NUM_PER_NEST    16
#define DTYPE               4       // FP16

#define ROWS                256
#define COLS                64
#define FP16_B              2

#define BASE_DDR_INPUT      0x1000000   // src[256][64] FP16 = 32768B
#define BASE_DDR_RESULT     0xf000000   // dst[256][64] FP16 = 32768B

#define L2_A                0x000000
#define L2_RESULT           0x008000

#define BANK_A              0x00000     // L1 bank for running row values
#define BANK_B              0x20000     // L1 bank for shifted row values
#define BANK_R              0x50000     // L1 bank for dst row
#define SVR_ADDR            0x800
#define SVR_ZERO            0
#define SVR_BYTES           32

#define ROW_BYTES           (COLS * DTYPE)          // 128 bytes per row
#define TOTAL_BYTES         (ROWS * ROW_BYTES)      // 32768 bytes total
#define ROWS_PER_NEST       (ROWS / NEST_NUM)       // 256
#define ROWS_PER_SPU        (ROWS_PER_NEST / SPU_NUM_PER_NEST)  // 16

int main(void) {

    __split();

    {
        uint8_t nest_id = 0;

        __start_plan(nest_id);

            __start_shared();
                // DDR -> L2: load entire src tensor
                __load_cr(GTX_MAIN_ADDR(BASE_DDR_INPUT), L2_A,
                    (uint32_t)TOTAL_BYTES,
                    (uint16_t)(TOTAL_BYTES & 0xFFFF),
                    1, (uint16_t)(TOTAL_BYTES & 0xFFFF),
                    1, 0xFFFF, 0xBEEF);

                // Wait for all 16 SPUs to finish
                __credit_chk(0xFFFF);

                // L2 -> DDR: store result
                __store_cr(L2_RESULT, GTX_MAIN_ADDR(BASE_DDR_RESULT),
                    (uint32_t)TOTAL_BYTES,
                    (uint16_t)(TOTAL_BYTES & 0xFFFF),
                    1, (uint16_t)(TOTAL_BYTES & 0xFFFF),
                    1, 0xFFFF);
            __end_shared();

            for (uint8_t tid = 0; tid < SPU_NUM_PER_NEST; tid++) {
                __start_thread(tid);
                    __credit_chk(0xBEEF);

                    __wrspr(SVR_ADDR + SVR_ZERO*4 + 0, 0, 0, 0);
                    __wrspr(SVR_ADDR + SVR_ZERO*4 + 1, 0, 0, 0);
                    __wrspr(SVR_ADDR + SVR_ZERO*4 + 2, 0, 0, 0);
                    __wrspr(SVR_ADDR + SVR_ZERO*4 + 3, 0, 0, 0);

                    for (uint8_t r = 0; r < ROWS_PER_SPU; r++) {
                        uint32_t row_off = (uint32_t)(tid * ROWS_PER_SPU + r) * ROW_BYTES;
                        uint16_t tid_mask = (uint16_t)(0x1 << tid);

                        // L2 -> L1: load src row to Bank A
                        if (r == ROWS_PER_SPU - 1) {
                            __load_cr(L2_A + row_off, BANK_A,
                                ROW_BYTES, (uint16_t)ROW_BYTES, 1, (uint16_t)ROW_BYTES,
                                1, tid_mask, nest_id);
                        } else {
                            __load(L2_A + row_off, BANK_A,
                                ROW_BYTES, (uint16_t)ROW_BYTES, 1, (uint16_t)ROW_BYTES);
                        }

                        // Hillis-Steele inclusive scan along the 64-column row.
                        // B needs zeros only in the prefix exposed by each shift;
                        // A/R are ping-ponged so the full-row copy-back is avoided.
                        __set_spm_addr(BANK_R, 0x30000, BANK_B, BANK_A);
                        __store_svr(BANK_B, SVR_ZERO);
                        __copy(BANK_A, BANK_B + 1 * FP16_B,
                            ROW_BYTES - 1 * FP16_B, (uint16_t)(ROW_BYTES - 1 * FP16_B),
                            1, (uint16_t)(ROW_BYTES - 1 * FP16_B));
                        __add_vv(COLS);

                        __set_spm_addr(BANK_A, 0x30000, BANK_B, BANK_R);
                        __store_svr(BANK_B, SVR_ZERO);
                        __copy(BANK_R, BANK_B + 2 * FP16_B,
                            ROW_BYTES - 2 * FP16_B, (uint16_t)(ROW_BYTES - 2 * FP16_B),
                            1, (uint16_t)(ROW_BYTES - 2 * FP16_B));
                        __add_vv(COLS);

                        __set_spm_addr(BANK_R, 0x30000, BANK_B, BANK_A);
                        __store_svr(BANK_B, SVR_ZERO);
                        __copy(BANK_A, BANK_B + 4 * FP16_B,
                            ROW_BYTES - 4 * FP16_B, (uint16_t)(ROW_BYTES - 4 * FP16_B),
                            1, (uint16_t)(ROW_BYTES - 4 * FP16_B));
                        __add_vv(COLS);

                        __set_spm_addr(BANK_A, 0x30000, BANK_B, BANK_R);
                        __store_svr(BANK_B, SVR_ZERO);
                        __copy(BANK_R, BANK_B + 8 * FP16_B,
                            ROW_BYTES - 8 * FP16_B, (uint16_t)(ROW_BYTES - 8 * FP16_B),
                            1, (uint16_t)(ROW_BYTES - 8 * FP16_B));
                        __add_vv(COLS);

                        __set_spm_addr(BANK_R, 0x30000, BANK_B, BANK_A);
                        __store_svr(BANK_B, SVR_ZERO);
                        __copy(BANK_A, BANK_B + 16 * FP16_B,
                            ROW_BYTES - 16 * FP16_B, (uint16_t)(ROW_BYTES - 16 * FP16_B),
                            1, (uint16_t)(ROW_BYTES - 16 * FP16_B));
                        __add_vv(COLS);

                        __set_spm_addr(BANK_A, 0x30000, BANK_B, BANK_R);
                        __store_svr(BANK_B, SVR_ZERO);
                        __store_svr(BANK_B + SVR_BYTES, SVR_ZERO);
                        __copy(BANK_R, BANK_B + 32 * FP16_B,
                            ROW_BYTES - 32 * FP16_B, (uint16_t)(ROW_BYTES - 32 * FP16_B),
                            1, (uint16_t)(ROW_BYTES - 32 * FP16_B));
                        __add_vv(COLS);

                        // L1 -> L2: store final ping-pong result row from Bank A
                        if (r == ROWS_PER_SPU - 1) {
                            __store_cr(BANK_A, L2_RESULT + row_off,
                                ROW_BYTES, (uint16_t)ROW_BYTES, 1, (uint16_t)ROW_BYTES,
                                1, tid_mask);
                        } else {
                            __store(BANK_A, L2_RESULT + row_off,
                                ROW_BYTES, (uint16_t)ROW_BYTES, 1, (uint16_t)ROW_BYTES);
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
