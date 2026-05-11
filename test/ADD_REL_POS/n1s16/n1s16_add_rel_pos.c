//==================================================================
// n1s16_add_rel_pos
//
// Static verifier-aligned retry shape:
//   shape = [8, 95]
//   side  = 8
//   heads = 95
//   src0  = [64, 95] (patch_count x heads, flattened as 760 rows x 8 FP16)
//   pw    = [8, 95]  (per-head column bias vectors)
//   ph    = [8, 95]  (per-head row bias vectors)
//   result bytes = 760 * 8 * 2 = 12160 (0x2f80)
//   bias bytes   = 95 * 8 * 2  = 1520  (0x05f0) per pw/ph tensor
//
// GGML semantic subset implemented here:
//   dst[head][row][col] = src0[head][row][col] + pw[head][col] + ph[head][row]
//
// GTX implementation strategy:
//   - shared: DDR->L2 preload for src0/pw/ph, then final L2->DDR store
//   - thread: load one src0 row + matching pw vector, run __add_vv,
//             then broadcast the selected ph scalar across BANK_B and add again
//==================================================================

#include <stdint.h>

#include "gtx/intrinsics/intrin.h"
#include "gtx/address.h"
#include "gtx/intrinsics/gtx_csr.h"

#define NEST_NUM            1
#define SPU_NUM_PER_NEST    16
#define DTYPE               2

#define SIDE                8
#define HEADS               95
#define PATCH_COUNT         (SIDE * SIDE)

#define ALIGN_UP_32(x)      (((x) + 31u) & ~31u)

#define WIDTH               SIDE
#define HEIGHT              (HEADS * SIDE)

#define BASE_DDR_A          0x1000000
#define BASE_DDR_B          0x2000000
#define BASE_DDR_C          0x3000000
#define BASE_DDR_RESULT     0xf000000

#define L2_A                0x000000

#define BANK_A              0x00000
#define BANK_B              0x20000
#define BANK_C              0x30000
#define BANK_R              0x50000

#define ROW_BYTES           (WIDTH * DTYPE)
#define PW_BYTES            (HEADS * SIDE * DTYPE)
#define NEST_DATA_BYTES     (HEIGHT * ROW_BYTES)
#define L2_B                ALIGN_UP_32(L2_A + NEST_DATA_BYTES)
#define L2_C                ALIGN_UP_32(L2_B + PW_BYTES)
#define L2_RESULT           ALIGN_UP_32(L2_C + PW_BYTES)

int main(void) {

    const uint8_t nest_id = 0;
    uint16_t active_tid_mask = 0;

    for (uint8_t tid = 0; tid < SPU_NUM_PER_NEST; tid++) {
        const uint32_t row_begin = ((uint32_t) tid * HEIGHT) / SPU_NUM_PER_NEST;
        const uint32_t row_end = ((uint32_t) (tid + 1) * HEIGHT) / SPU_NUM_PER_NEST;

        if (row_begin < row_end) {
            active_tid_mask |= (uint16_t) (0x1u << tid);
        }
    }

    __split();

    {
        __start_plan(nest_id);

            __start_shared();
                __load(
                    GTX_MAIN_ADDR(BASE_DDR_A), L2_A,
                    NEST_DATA_BYTES,
                    (uint16_t) NEST_DATA_BYTES,
                    1, (uint16_t) NEST_DATA_BYTES
                );

                __load(
                    GTX_MAIN_ADDR(BASE_DDR_B), L2_B,
                    PW_BYTES,
                    (uint16_t) PW_BYTES,
                    1, (uint16_t) PW_BYTES
                );

                __load_cr(
                    GTX_MAIN_ADDR(BASE_DDR_C), L2_C,
                    PW_BYTES,
                    (uint16_t) PW_BYTES,
                    1, (uint16_t) PW_BYTES,
                    1, active_tid_mask, 0xBEEF
                );

                __credit_chk(active_tid_mask);

                __store_cr(
                    L2_RESULT, GTX_MAIN_ADDR(BASE_DDR_RESULT),
                    NEST_DATA_BYTES,
                    (uint16_t) NEST_DATA_BYTES,
                    1, (uint16_t) NEST_DATA_BYTES,
                    1, active_tid_mask
                );
            __end_shared();

            for (uint8_t tid = 0; tid < SPU_NUM_PER_NEST; tid++) {
                __start_thread(tid);
                    const uint32_t row_begin = ((uint32_t) tid * HEIGHT) / SPU_NUM_PER_NEST;
                    const uint32_t row_end = ((uint32_t) (tid + 1) * HEIGHT) / SPU_NUM_PER_NEST;
                    const uint32_t rows_for_tid = row_end - row_begin;
                    const uint16_t tid_mask = (uint16_t) (0x1u << tid);

                    __set_spm_addr(BANK_R, BANK_C, BANK_B, BANK_A);
                    if (rows_for_tid > 0) {
                        __credit_chk(0xBEEF);

                        for (uint32_t r = 0; r < rows_for_tid; r++) {
                            const uint32_t global_row = row_begin + r;
                            const uint32_t row_off = global_row * ROW_BYTES;
                            const uint32_t head = global_row / SIDE;
                            const uint32_t row_in_head = global_row % SIDE;
                            const uint32_t pw_off = head * ROW_BYTES;
                            const uint32_t ph_off = head * ROW_BYTES;

                            __load(
                                L2_A + row_off, BANK_A,
                                ROW_BYTES, (uint16_t) ROW_BYTES, 1, (uint16_t) ROW_BYTES
                            );

                            __load(
                                L2_B + pw_off, BANK_B,
                                ROW_BYTES, (uint16_t) ROW_BYTES, 1, (uint16_t) ROW_BYTES
                            );

                            if (r + 1 == rows_for_tid) {
                                __load_cr(
                                    L2_C + ph_off, BANK_C,
                                    ROW_BYTES, (uint16_t) ROW_BYTES, 1, (uint16_t) ROW_BYTES,
                                    1, tid_mask, nest_id
                                );
                            } else {
                                __load(
                                    L2_C + ph_off, BANK_C,
                                    ROW_BYTES, (uint16_t) ROW_BYTES, 1, (uint16_t) ROW_BYTES
                                );
                            }

                            __add_vv(WIDTH);

                            __copy(BANK_R, BANK_A,
                                ROW_BYTES, (uint16_t) ROW_BYTES, 1, (uint16_t) ROW_BYTES
                            );

                            __load_svr(BANK_C + row_in_head * DTYPE, 0);
                            __store_svr(BANK_C, 0);
                            for (uint8_t k = 0; k < WIDTH; k++) {
                                __copy(BANK_C, BANK_B + k * DTYPE, 0, DTYPE, 1, 0);
                            }

                            __add_vv(WIDTH);

                            if (r + 1 == rows_for_tid) {
                                __store_cr(
                                    BANK_R, L2_RESULT + row_off,
                                    ROW_BYTES, (uint16_t) ROW_BYTES, 1, (uint16_t) ROW_BYTES,
                                    1, tid_mask
                                );
                            } else {
                                __store(
                                    BANK_R, L2_RESULT + row_off,
                                    ROW_BYTES, (uint16_t) ROW_BYTES, 1, (uint16_t) ROW_BYTES
                                );
                            }
                        }
                    }
                __end_thread(tid);
            }

        __end_plan(nest_id);
    }

    __join();
    return 0;
}
