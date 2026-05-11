// =================================================================
// GGML_OP_CONCAT — Concatenate two tensors (n1s16)
//
// Description:
//   Concatenates two [32, 8] FP16 tensors along ggml dim 0:
//   - src0: 32 cols × 8 rows = 512 bytes at 0x1000000
//   - src1: 32 cols × 8 rows = 512 bytes at 0x2000000
//   - dst:  64 cols × 8 rows = 1024 bytes at 0xf000000
//
// Implementation:
//   - Uses a shared-section L2 row buffer with row-wise DDR->L2->DDR DMA
//   - Copies each src0 row to the front half of the destination row
//   - Copies each src1 row to the back half of the destination row
//   - Executes the copy schedule inside one GTX split/plan/shared/join scope
//   - No thread section needed (pure data movement operation)
//
// Memory Layout:
//   - BASE_DDR_SRC0: 0x1000000 (first tensor)
//   - BASE_DDR_SRC1: 0x2000000 (second tensor)
//   - BASE_DDR_RESULT: 0xf000000 (destination)
//   - SRC_ROW_BYTES: 64 bytes, DST_ROW_BYTES: 128 bytes
// =================================================================

#include <stdint.h>

#include "intrin.h"
#include "gtx/address.h"
#include "gtx_csr.h"

#ifndef NEST_ID
#define NEST_ID             0
#endif

#define BASE_DDR_SRC0       0x1000000
#define BASE_DDR_SRC1       0x2000000
#define BASE_DDR_RESULT     0xf000000

#define SRC_COLS            2049
#define ROWS                257
#define FP16_B              2
#define SRC_ROW_BYTES       (SRC_COLS * FP16_B)          // 64
#define DST_ROW_BYTES       (SRC_ROW_BYTES * 2)          // 128

#define L2_ROW_BUF          0x000000
#define SHARED_ONLY_SPU_MASK 0x0000
#define SHARED_LOAD_TOKEN    0xBEEF

int main(void)
{
    uint8_t nest_id = NEST_ID;

    __split();
    __start_plan(nest_id);
    __start_shared();

    for (uint32_t row = 0; row < ROWS; ++row) {
        const uint32_t src_row_offset = row * SRC_ROW_BYTES;
        const uint32_t dst_row_offset = row * DST_ROW_BYTES;

        __load(
            GTX_MAIN_ADDR(BASE_DDR_SRC0) + src_row_offset,
            L2_ROW_BUF,
            SRC_ROW_BYTES,
            (uint16_t) SRC_ROW_BYTES,
            1,
            SRC_ROW_BYTES
        );

        __credit_ld(SHARED_ONLY_SPU_MASK, SHARED_LOAD_TOKEN);
        __credit_chk(SHARED_ONLY_SPU_MASK);

        __store(
            L2_ROW_BUF,
            GTX_MAIN_ADDR(BASE_DDR_RESULT) + dst_row_offset,
            SRC_ROW_BYTES,
            (uint16_t) SRC_ROW_BYTES,
            1,
            SRC_ROW_BYTES
        );

        __credit_st(SHARED_ONLY_SPU_MASK);

        __load(
            GTX_MAIN_ADDR(BASE_DDR_SRC1) + src_row_offset,
            L2_ROW_BUF,
            SRC_ROW_BYTES,
            (uint16_t) SRC_ROW_BYTES,
            1,
            SRC_ROW_BYTES
        );

        __credit_ld(SHARED_ONLY_SPU_MASK, SHARED_LOAD_TOKEN);
        __credit_chk(SHARED_ONLY_SPU_MASK);

        __store(
            L2_ROW_BUF,
            GTX_MAIN_ADDR(BASE_DDR_RESULT) + dst_row_offset + SRC_ROW_BYTES,
            SRC_ROW_BYTES,
            (uint16_t) SRC_ROW_BYTES,
            1,
            SRC_ROW_BYTES
        );

        __credit_st(SHARED_ONLY_SPU_MASK);
    }

    __end_shared();
    __end_plan(nest_id);

    __join();

    return 0;
}
