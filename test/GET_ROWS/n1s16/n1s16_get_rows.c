// =================================================================
// GGML_OP_GET_ROWS — gather rows by I32 index (n1s16)
//
// Self-check shape: [128,4]
//   src0: 4 rows x 128 FP16 values at 0x1000000
//   src1: 4 I32 row indices at 0x2000000
//   dst:  4 rows x 128 FP16 values at 0xf000000
//
// The row-index stream is bounded metadata: the RISC-V side reads only the
// four I32 row selectors used by the generated GET_ROWS case.  The tensor data
// movement is still performed by GTX DMA inside one split/plan/shared region,
// using a contiguous L2 row buffer for each gathered row.
// =================================================================

#include <stdint.h>

#include "intrin.h"
#include "gtx/address.h"
#include "gtx_csr.h"

#ifndef NEST_ID
#define NEST_ID             0
#endif

#define BASE_DDR_SRC        0x1000000
#define BASE_DDR_INDICES    0x2000000
#define BASE_DDR_RESULT     0xf000000

#define SRC_ROWS            4
#define DST_ROWS            4
#define COLS                32767
#define FP16_B              2
#define I32_B               4
#define ROW_BYTES           (COLS * FP16_B)
#define OUT_BYTES           (DST_ROWS * ROW_BYTES)

#define L2_ROW_BUF          0x000000
#define SHARED_ONLY_SPU_MASK 0x0000
#define SHARED_LOAD_TOKEN    0xBEEF

int main(void)
{
    uint8_t nest_id = NEST_ID;

    volatile int32_t *idx = (volatile int32_t *)GTX_MAIN_ADDR(BASE_DDR_INDICES);
    int32_t row_index[DST_ROWS];

    for (uint32_t row = 0; row < DST_ROWS; ++row) {
        int32_t selected = idx[row];
        if (selected < 0) {
            selected = 0;
        }
        if (selected >= SRC_ROWS) {
            selected = SRC_ROWS - 1;
        }
        row_index[row] = selected;
    }

    __split();
    __start_plan(nest_id);
    __start_shared();

    for (uint32_t row = 0; row < DST_ROWS; ++row) {
        const uint32_t src_offset = ((uint32_t)row_index[row]) * ROW_BYTES;
        const uint32_t dst_offset = row * ROW_BYTES;

        __load(
            GTX_MAIN_ADDR(BASE_DDR_SRC) + src_offset,
            L2_ROW_BUF,
            ROW_BYTES,
            (uint16_t)ROW_BYTES,
            1,
            ROW_BYTES
        );
        __credit_ld(SHARED_ONLY_SPU_MASK, SHARED_LOAD_TOKEN);
        __credit_chk(SHARED_ONLY_SPU_MASK);

        __store(
            L2_ROW_BUF,
            GTX_MAIN_ADDR(BASE_DDR_RESULT) + dst_offset,
            ROW_BYTES,
            (uint16_t)ROW_BYTES,
            1,
            ROW_BYTES
        );
        __credit_st(SHARED_ONLY_SPU_MASK);
    }

    __end_shared();
    __end_plan(nest_id);
    __join();

    (void)I32_B;
    (void)OUT_BYTES;
    return 0;
}
