//==================================================================
// {{OP_NAME}} (generated) — unary element-wise kernel via GTX_LAUNCH
// HEIGHT rows × 8 FP16 elements; intrinsic {{INTRIN_CALL}} substituted.
// Equivalent to the legacy split/plan/shared/thread version but written
// with the CUDA-style gtx_kernel.h abstraction (<<<1, 16>>>).
//==================================================================

#include "gtx_kernel.h"
#include "gtx_csr.h"

#define NESTS               1
#define SPUS_PER_NEST       16
#define DTYPE               2       // FP16
#define WIDTH               8
#define HEIGHT              {{HEIGHT}}

#define BASE_DDR_A          0x1000000
#define BASE_DDR_RESULT     0xf000000

#define L2_A                0x000000

#define BANK_A              0x00000
#define BANK_B              0x20000
#define BANK_C              0x30000
#define BANK_R              0x50000

#define ROW_BYTES           (WIDTH * DTYPE)
#define ROWS_PER_NEST       (HEIGHT / NESTS)
#define NEST_DATA_BYTES     (ROWS_PER_NEST * ROW_BYTES)
#define MAX_SHARED_DMA_BYTES 65535u
#define SHARED_TILE_MAX_ROWS ((uint32_t)(MAX_SHARED_DMA_BYTES / ROW_BYTES))
#define SHARED_TILE_MAX_BYTES (SHARED_TILE_MAX_ROWS * ROW_BYTES)
#define L2_RESULT           (L2_A + SHARED_TILE_MAX_BYTES)

GTX_KERNEL_BODY(
    /* SHARED_BODY */ {
        uint32_t nest_off = (uint32_t)nest_id * NEST_DATA_BYTES;
        /* Single-tile launch — caller must ensure NEST_DATA_BYTES ≤ DMA cap. */
        __load_cr(GTX_MAIN_ADDR(BASE_DDR_A) + nest_off, L2_A,
            (uint32_t)NEST_DATA_BYTES,
            (uint16_t)NEST_DATA_BYTES,
            1, (uint16_t)NEST_DATA_BYTES,
            1, active_mask, 0xBEEF);

        __credit_chk(active_mask);

        __store_cr(L2_RESULT, GTX_MAIN_ADDR(BASE_DDR_RESULT) + nest_off,
            (uint32_t)NEST_DATA_BYTES,
            (uint16_t)NEST_DATA_BYTES,
            1, (uint16_t)NEST_DATA_BYTES,
            1, active_mask);
    },
    /* THREAD_BODY */ {
        uint32_t rows_this = gtx_items_for_tid(ROWS_PER_NEST, tid, n_threads);
        uint32_t row_start = gtx_start_for_tid(ROWS_PER_NEST, tid, n_threads);

        __set_spm_addr(BANK_R, BANK_C, BANK_B, BANK_A);
        __credit_chk(0xBEEF);

        for (uint32_t r = 0; r < rows_this; r++) {
            uint32_t row_off = (row_start + r) * ROW_BYTES;
            int last = (r == rows_this - 1);

            if (last) {
                __load_cr(L2_A + row_off, {{LOAD_BANK}},
                    ROW_BYTES, (uint16_t)ROW_BYTES, 1, (uint16_t)ROW_BYTES,
                    1, tid_mask, nest_id);
            } else {
                __load(L2_A + row_off, {{LOAD_BANK}},
                    ROW_BYTES, (uint16_t)ROW_BYTES, 1, (uint16_t)ROW_BYTES);
            }

            {{INTRIN_CALL}}

            if (last) {
                __store_cr({{STORE_BANK}}, L2_RESULT + row_off,
                    ROW_BYTES, (uint16_t)ROW_BYTES, 1, (uint16_t)ROW_BYTES,
                    1, tid_mask);
            } else {
                __store({{STORE_BANK}}, L2_RESULT + row_off,
                    ROW_BYTES, (uint16_t)ROW_BYTES, 1, (uint16_t)ROW_BYTES);
            }
        }
    }
)

int main(void) {
    GTX_LAUNCH(NESTS, SPUS_PER_NEST);
    return 0;
}
