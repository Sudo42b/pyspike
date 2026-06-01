//==================================================================
// Copyright   : (C) Supergate - All Rights Reserved
// Project     : GSF / VTS
// Test        : n1s16_cumsum (1 NEST x 16 SPUs = 16 SPUs)
// Description : Cumulative sum per row using shared/thread DMA pattern.
//
//               src: [112][64] FP16 at 0x1000000 (14336B)
//               dst: [112][64] FP16 at 0xf000000 (14336B)
//
//               For each row r in 0..111:
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

#include "intrin.h"
#include "gtx/address.h"
#include "gtx_csr.h"

#define NEST_NUM            1
#define SPU_NUM_PER_NEST    16
#define DTYPE               4       // FP16

#define ROWS                112
#define COLS                64
#define FP16_B              2

#define BASE_DDR_INPUT      0x1000000   // src[112][64] FP16 = 14336B
#define BASE_DDR_RESULT     0xf000000   // dst[112][64] FP16 = 14336B

#define L2_A                0x000000
#define L2_RESULT           0x008000

#define BANK_A              0x00000     // L1 bank for running row values
#define BANK_B              0x20000     // L1 bank for shifted row values
#define BANK_R              0x50000     // L1 bank for dst row
#define SVR_ADDR            0x800
#define SVR_BYTES           32

#define ROW_BYTES           (COLS * DTYPE)          // 128 bytes per row
#define TOTAL_BYTES         (ROWS * ROW_BYTES)      // 14336 bytes total
#define ROWS_PER_NEST       (ROWS / NEST_NUM)       // 112
#define ROWS_PER_SPU        (ROWS_PER_NEST / SPU_NUM_PER_NEST)  // 7

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
                    __set_spm_addr(BANK_R, 0x30000, BANK_B, BANK_A);
                    __credit_chk(0xBEEF);

                    for (uint8_t r = 0; r < ROWS_PER_SPU; r++) {
                        uint32_t row_off = (uint32_t)(tid * ROWS_PER_SPU + r) * ROW_BYTES;

                        // L2 -> L1: load src row to Bank A
                        __load(L2_A + row_off, BANK_A,
                            ROW_BYTES, (uint16_t)ROW_BYTES, 1, (uint16_t)ROW_BYTES);
                        if (r == ROWS_PER_SPU - 1) __credit_ld(tid, nest_id);

                        // Hillis-Steele inclusive scan along the 64-column row:
                        // on each pass, B is zero-filled then receives A shifted
                        // right by the pass offset, and __add_vv writes A+B to R.
                        for (uint32_t off = 0; off < ROW_BYTES; off += SVR_BYTES) {
                            __wrspr(SVR_ADDR + 0*4 + 0, 0, 0, 0);
                            __wrspr(SVR_ADDR + 0*4 + 1, 0, 0, 0);
                            __wrspr(SVR_ADDR + 0*4 + 2, 0, 0, 0);
                            __wrspr(SVR_ADDR + 0*4 + 3, 0, 0, 0);
                            __store_svr(BANK_B + off, 0);
                        }
                        __copy(BANK_A, BANK_B + 1 * FP16_B,
                            ROW_BYTES - 1 * FP16_B, (uint16_t)(ROW_BYTES - 1 * FP16_B),
                            1, (uint16_t)(ROW_BYTES - 1 * FP16_B));
                        __add_vv(COLS);
                        __copy(BANK_R, BANK_A, ROW_BYTES, (uint16_t)ROW_BYTES, 1, (uint16_t)ROW_BYTES);

                        for (uint32_t off = 0; off < ROW_BYTES; off += SVR_BYTES) {
                            __wrspr(SVR_ADDR + 0*4 + 0, 0, 0, 0);
                            __wrspr(SVR_ADDR + 0*4 + 1, 0, 0, 0);
                            __wrspr(SVR_ADDR + 0*4 + 2, 0, 0, 0);
                            __wrspr(SVR_ADDR + 0*4 + 3, 0, 0, 0);
                            __store_svr(BANK_B + off, 0);
                        }
                        __copy(BANK_A, BANK_B + 2 * FP16_B,
                            ROW_BYTES - 2 * FP16_B, (uint16_t)(ROW_BYTES - 2 * FP16_B),
                            1, (uint16_t)(ROW_BYTES - 2 * FP16_B));
                        __add_vv(COLS);
                        __copy(BANK_R, BANK_A, ROW_BYTES, (uint16_t)ROW_BYTES, 1, (uint16_t)ROW_BYTES);

                        for (uint32_t off = 0; off < ROW_BYTES; off += SVR_BYTES) {
                            __wrspr(SVR_ADDR + 0*4 + 0, 0, 0, 0);
                            __wrspr(SVR_ADDR + 0*4 + 1, 0, 0, 0);
                            __wrspr(SVR_ADDR + 0*4 + 2, 0, 0, 0);
                            __wrspr(SVR_ADDR + 0*4 + 3, 0, 0, 0);
                            __store_svr(BANK_B + off, 0);
                        }
                        __copy(BANK_A, BANK_B + 4 * FP16_B,
                            ROW_BYTES - 4 * FP16_B, (uint16_t)(ROW_BYTES - 4 * FP16_B),
                            1, (uint16_t)(ROW_BYTES - 4 * FP16_B));
                        __add_vv(COLS);
                        __copy(BANK_R, BANK_A, ROW_BYTES, (uint16_t)ROW_BYTES, 1, (uint16_t)ROW_BYTES);

                        for (uint32_t off = 0; off < ROW_BYTES; off += SVR_BYTES) {
                            __wrspr(SVR_ADDR + 0*4 + 0, 0, 0, 0);
                            __wrspr(SVR_ADDR + 0*4 + 1, 0, 0, 0);
                            __wrspr(SVR_ADDR + 0*4 + 2, 0, 0, 0);
                            __wrspr(SVR_ADDR + 0*4 + 3, 0, 0, 0);
                            __store_svr(BANK_B + off, 0);
                        }
                        __copy(BANK_A, BANK_B + 8 * FP16_B,
                            ROW_BYTES - 8 * FP16_B, (uint16_t)(ROW_BYTES - 8 * FP16_B),
                            1, (uint16_t)(ROW_BYTES - 8 * FP16_B));
                        __add_vv(COLS);
                        __copy(BANK_R, BANK_A, ROW_BYTES, (uint16_t)ROW_BYTES, 1, (uint16_t)ROW_BYTES);

                        for (uint32_t off = 0; off < ROW_BYTES; off += SVR_BYTES) {
                            __wrspr(SVR_ADDR + 0*4 + 0, 0, 0, 0);
                            __wrspr(SVR_ADDR + 0*4 + 1, 0, 0, 0);
                            __wrspr(SVR_ADDR + 0*4 + 2, 0, 0, 0);
                            __wrspr(SVR_ADDR + 0*4 + 3, 0, 0, 0);
                            __store_svr(BANK_B + off, 0);
                        }
                        __copy(BANK_A, BANK_B + 16 * FP16_B,
                            ROW_BYTES - 16 * FP16_B, (uint16_t)(ROW_BYTES - 16 * FP16_B),
                            1, (uint16_t)(ROW_BYTES - 16 * FP16_B));
                        __add_vv(COLS);
                        __copy(BANK_R, BANK_A, ROW_BYTES, (uint16_t)ROW_BYTES, 1, (uint16_t)ROW_BYTES);

                        for (uint32_t off = 0; off < ROW_BYTES; off += SVR_BYTES) {
                            __wrspr(SVR_ADDR + 0*4 + 0, 0, 0, 0);
                            __wrspr(SVR_ADDR + 0*4 + 1, 0, 0, 0);
                            __wrspr(SVR_ADDR + 0*4 + 2, 0, 0, 0);
                            __wrspr(SVR_ADDR + 0*4 + 3, 0, 0, 0);
                            __store_svr(BANK_B + off, 0);
                        }
                        __copy(BANK_A, BANK_B + 32 * FP16_B,
                            ROW_BYTES - 32 * FP16_B, (uint16_t)(ROW_BYTES - 32 * FP16_B),
                            1, (uint16_t)(ROW_BYTES - 32 * FP16_B));
                        __add_vv(COLS);

                        // L1 -> L2: store result row from Bank R
                        if (r == ROWS_PER_SPU - 1) {
                            __store_cr(BANK_R, L2_RESULT + row_off,
                                ROW_BYTES, (uint16_t)ROW_BYTES, 1, (uint16_t)ROW_BYTES,
                                1, 0x1 << tid);
                        } else {
                            __store(BANK_R, L2_RESULT + row_off,
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
