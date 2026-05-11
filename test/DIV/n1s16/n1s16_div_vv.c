//==================================================================
// n1s16_div_vv — element-wise FP16 vector-vector division
// GGML DIV semantic subset used here: dst[row] = src0[row] / src1[row]
// 1 NEST x 16 SPUs, HEIGHT rows x WIDTH FP16 elements
//==================================================================

#include <stdint.h>

#include "intrin.h"
#include "gtx/address.h"
#include "gtx_csr.h"

#define NEST_NUM            1
#define SPU_NUM_PER_NEST    16
#define DTYPE               2

#define WIDTH               32
#define HEIGHT              65

#define BASE_DDR_A          0x1000000
#define BASE_DDR_B          0x2000000
#define BASE_DDR_RESULT     0xf000000

#define BANK_A              0x00000
#define BANK_B              0x20000
#define BANK_C              0x30000
#define BANK_R              0x50000

#define ROW_BYTES           (WIDTH * DTYPE)
#define ROWS_PER_NEST       (HEIGHT / NEST_NUM)
#define TOTAL_BYTES         (ROWS_PER_NEST * ROW_BYTES)
#define L2_A                0x000000
#define L2_B                (L2_A + TOTAL_BYTES)
#define L2_RESULT           (L2_B + TOTAL_BYTES)
#define BASE_ROWS_PER_SPU   (ROWS_PER_NEST / SPU_NUM_PER_NEST)
#define EXTRA_ROWS_PER_SPU  (ROWS_PER_NEST % SPU_NUM_PER_NEST)

static inline uint16_t tid_mask(uint8_t tid) {
    return (uint16_t)(1u << tid);
}

static inline uint16_t active_tid_mask(void) {
    uint16_t mask = 0;
    uint32_t active_spus = (ROWS_PER_NEST < SPU_NUM_PER_NEST) ? ROWS_PER_NEST : SPU_NUM_PER_NEST;

    for (uint32_t tid = 0; tid < active_spus; tid++) {
        mask |= (uint16_t)(1u << tid);
    }

    return mask;
}

static inline uint32_t rows_for_tid(uint8_t tid) {
    return BASE_ROWS_PER_SPU + (tid < EXTRA_ROWS_PER_SPU ? 1u : 0u);
}

static inline uint32_t row_start_for_tid(uint8_t tid) {
    uint32_t extra_before_tid = (tid < EXTRA_ROWS_PER_SPU) ? tid : EXTRA_ROWS_PER_SPU;
    return (uint32_t)tid * BASE_ROWS_PER_SPU + extra_before_tid;
}

int main(void) {
    __split();

    {
        uint8_t nest_id = 0;
        uint16_t active_mask = active_tid_mask();
        uint32_t nest_off = (uint32_t)nest_id * ROWS_PER_NEST * ROW_BYTES;

        __start_plan(nest_id);

            __start_shared();
                __load(
                    GTX_MAIN_ADDR(BASE_DDR_A) + nest_off, L2_A,
                    (uint32_t)(ROWS_PER_NEST * ROW_BYTES),
                    (uint16_t)(ROWS_PER_NEST * ROW_BYTES),
                    1, (uint16_t)(ROWS_PER_NEST * ROW_BYTES)
                );

                __load_cr(
                    GTX_MAIN_ADDR(BASE_DDR_B) + nest_off, L2_B,
                    (uint32_t)(ROWS_PER_NEST * ROW_BYTES),
                    (uint16_t)(ROWS_PER_NEST * ROW_BYTES),
                    1, (uint16_t)(ROWS_PER_NEST * ROW_BYTES),
                    1, active_mask, 0xBEEF
                );

                __credit_chk(active_mask);

                __store_cr(
                    L2_RESULT, GTX_MAIN_ADDR(BASE_DDR_RESULT) + nest_off,
                    (uint32_t)(ROWS_PER_NEST * ROW_BYTES),
                    (uint16_t)(ROWS_PER_NEST * ROW_BYTES),
                    1, (uint16_t)(ROWS_PER_NEST * ROW_BYTES),
                    1, active_mask
                );
            __end_shared();

            for (uint8_t tid = 0; tid < SPU_NUM_PER_NEST; tid++) {
                __start_thread(tid);
                    __set_spm_addr(BANK_R, BANK_C, BANK_B, BANK_A);

                    uint32_t thread_rows = rows_for_tid(tid);
                    uint32_t thread_row_start = row_start_for_tid(tid);

                    if (thread_rows != 0) {
                        __credit_chk(0xBEEF);
                    }

                    for (uint32_t r = 0; r < thread_rows; r++) {
                        uint32_t row_off = (thread_row_start + r) * ROW_BYTES;

                        __load(
                            L2_A + row_off, BANK_A,
                            ROW_BYTES, (uint16_t)ROW_BYTES, 1, (uint16_t)ROW_BYTES
                        );

                        if (r == thread_rows - 1) {
                            __load_cr(
                                L2_B + row_off, BANK_B,
                                ROW_BYTES, (uint16_t)ROW_BYTES, 1, (uint16_t)ROW_BYTES,
                                1, tid_mask(tid), nest_id
                            );
                        } else {
                            __load(
                                L2_B + row_off, BANK_B,
                                ROW_BYTES, (uint16_t)ROW_BYTES, 1, (uint16_t)ROW_BYTES
                            );
                        }

                        __div_vv(WIDTH);

                        if (r == thread_rows - 1) {
                            __store_cr(
                                BANK_R, L2_RESULT + row_off,
                                ROW_BYTES, (uint16_t)ROW_BYTES, 1, (uint16_t)ROW_BYTES,
                                1, tid_mask(tid)
                            );
                        } else {
                            __store(
                                BANK_R, L2_RESULT + row_off,
                                ROW_BYTES, (uint16_t)ROW_BYTES, 1, (uint16_t)ROW_BYTES
                            );
                        }
                    }
                __end_thread(tid);
            }

        __end_plan(nest_id);
    }

    __join();
    return 0;
}
