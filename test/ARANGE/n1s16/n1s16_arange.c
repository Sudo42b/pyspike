//==================================================================
// n1s16_arange — fixed 8x8 FP16 ARANGE tile on 1 nest x 16 SPUs
// dst[i] = i, i = 0..63
//
// Binding used by the current harness/data path:
// - no runtime input tensors
// - fixed ggml_arange(0.0f, 64.0f, 1.0f)
// - 64 FP16 outputs at BASE_DDR_RESULT (128 bytes)
//
// Each active SPU generates its assigned row(s) with the GTX
// __arange intrinsic, stores L1->L2, and the shared section performs
// the final L2->DDR store after all active threads complete.
//==================================================================

#include "intrin.h"
#include "gtx/address.h"
#include "gtx_csr.h"
#include <stdint.h>

#ifndef NEST_ID
#define NEST_ID             0
#endif

#define NEST_NUM            1
#define SPU_NUM_PER_NEST    16
#define DTYPE               2       // FP16

#define BASE_DDR_RESULT     0xf000000

#define L2_RESULT           0x000000

#define BANK_A              0x00000
#define BANK_B              0x20000
#define BANK_C              0x30000
#define BANK_R              0x50000
#define ROWS                8
#define COLS                8
#define ROW_BYTES           (COLS * DTYPE)
#define ROWS_PER_NEST       (ROWS / NEST_NUM)
#define STEP_FP16           0x3C00

static uint16_t float_to_fp16(float val) {
    uint32_t f;
    __builtin_memcpy(&f, &val, 4);
    uint32_t sign = (f >> 16) & 0x8000;
    int32_t exp = ((f >> 23) & 0xFF) - 127 + 15;
    uint32_t mant = f & 0x7FFFFF;
    if (exp <= 0) {
        if (exp < -10) {
            return (uint16_t) sign;
        }
        mant = (mant | 0x800000) >> (1 - exp);
        return (uint16_t) (sign | (mant >> 13));
    }
    if (exp >= 31) {
        return (uint16_t) (sign | 0x7C00);
    }
    return (uint16_t) (sign | ((uint32_t) exp << 10) | (mant >> 13));
}

static inline uint32_t min_u32(uint32_t a, uint32_t b) {
    return a < b ? a : b;
}

static inline uint32_t rows_for_tid(uint32_t total_rows, uint8_t tid) {
    uint32_t rows_per_spu_quot = total_rows / SPU_NUM_PER_NEST;
    uint32_t rows_per_spu_rem = total_rows % SPU_NUM_PER_NEST;
    return (uint32_t)(tid < rows_per_spu_rem ? (rows_per_spu_quot + 1) : rows_per_spu_quot);
}

static inline uint32_t row_start_for_tid(uint32_t total_rows, uint8_t tid) {
    uint32_t rows_per_spu_quot = total_rows / SPU_NUM_PER_NEST;
    uint32_t rows_per_spu_rem = total_rows % SPU_NUM_PER_NEST;

    return (uint32_t)(tid < rows_per_spu_rem
        ? tid * (rows_per_spu_quot + 1)
        : rows_per_spu_rem * (rows_per_spu_quot + 1) + (tid - rows_per_spu_rem) * rows_per_spu_quot);
}

static inline uint16_t active_tid_mask_for_rows(uint32_t total_rows) {
    uint32_t active_tid_count = min_u32(total_rows, SPU_NUM_PER_NEST);
    return (uint16_t)(active_tid_count >= SPU_NUM_PER_NEST ? 0xFFFFu : ((1u << active_tid_count) - 1u));
}

int main(void) {
    const uint16_t active_tid_mask = active_tid_mask_for_rows(ROWS_PER_NEST);
    const uint8_t nest_id = NEST_ID;

    __split();

    {
        __start_plan(nest_id);

            // ARANGE has no DDR input payload. The shared section only waits for
            // the participating SPUs to finish their final L1->L2 stores, then
            // performs the single L2->DDR commit for the completed output tile.
            __start_shared();
                __credit_chk(active_tid_mask);

                __store_cr(
                    L2_RESULT,
                    GTX_MAIN_ADDR(BASE_DDR_RESULT),
                    ROW_BYTES,
                    (uint16_t) ROW_BYTES,
                    ROWS_PER_NEST,
                    (uint16_t) ROW_BYTES,
                    1,
                    active_tid_mask
                );
            __end_shared();

            for (uint8_t tid = 0; tid < SPU_NUM_PER_NEST; tid++) {
                __start_thread(tid);
                    __set_spm_addr(BANK_R, BANK_C, BANK_B, BANK_A);

                    uint32_t rows_for_this_tid = rows_for_tid(ROWS_PER_NEST, tid);
                    uint32_t row_start = row_start_for_tid(ROWS_PER_NEST, tid);

                    for (uint32_t i = 0; i < rows_for_this_tid; i++) {
                        uint32_t global_row = row_start + i;
                        uint32_t start_index = global_row * COLS;

                        __arange((uint32_t) COLS, float_to_fp16((float) start_index), STEP_FP16);

                        if (i == rows_for_this_tid - 1) {
                            __store_cr(
                                BANK_R,
                                L2_RESULT + global_row * ROW_BYTES,
                                ROW_BYTES,
                                (uint16_t) ROW_BYTES,
                                1,
                                (uint16_t) ROW_BYTES,
                                1,
                                (uint16_t) (1u << tid)
                            );
                        } else {
                            __store(
                                BANK_R,
                                L2_RESULT + global_row * ROW_BYTES,
                                ROW_BYTES,
                                (uint16_t) ROW_BYTES,
                                1,
                                (uint16_t) ROW_BYTES
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
