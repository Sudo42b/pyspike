// =================================================================
// GGML_OP_PAD_REFLECT_1D — 1D reflect padding (n1s16, 8190x3 FP16 -> 8192x3 FP16)
//
// generate_data.cpp builds ggml_pad_reflect_1d(ctx, src0, 1, 1). For each row:
//   dst[0]            = src[1]
//   dst[1..SRC_LEN]   = src[0..SRC_LEN-1]
//   dst[SRC_LEN + 1]  = src[SRC_LEN-2]
//
// This is a pure data movement kernel. It composes shared-mode DDR->L2 loads for
// the left reflected element, contiguous center, and right reflected element,
// then performs one contiguous L2->DDR store of the padded output.
// =================================================================

#include "intrin.h"
#include "gtx/address.h"
#include "gtx_csr.h"
#include <stdint.h>

#ifndef NEST_ID
#define NEST_ID             0
#endif

#define BASE_DDR_A          0x1000000
#define BASE_DDR_R          0xf000000

#define L2_R                0x000000

#define ROWS                3
#define SRC_LEN             8190
#define PAD_LEFT            1
#define PAD_RIGHT           1
#define FP16_B              2
#define DST_LEN             (SRC_LEN + PAD_LEFT + PAD_RIGHT)
#define SRC_ROW_BYTES       (SRC_LEN * FP16_B)             // 16380
#define DST_ROW_BYTES       (DST_LEN * FP16_B)             // 16384
#define TOTAL_DST_BYTES     (ROWS * DST_ROW_BYTES)         // 49152

int main(void)
{
    uint8_t nest_id = NEST_ID;

    __split();
    {
        __start_plan(nest_id);
            __start_shared();
                for (int row = 0; row < ROWS; ++row) {
                    const uint64_t src_row = GTX_MAIN_ADDR(BASE_DDR_A) + (uint64_t)row * SRC_ROW_BYTES;
                    const uint32_t l2_row  = L2_R + (uint32_t)row * DST_ROW_BYTES;

                    // Left reflect: dst[0] = src[1].
                    __load(src_row + FP16_B,
                        l2_row,
                        (uint32_t)FP16_B,
                        (uint16_t)FP16_B,
                        1,
                        (uint16_t)FP16_B);

                    // Center copy: dst[1..SRC_LEN] = src[0..SRC_LEN-1].
                    __load(src_row,
                        l2_row + PAD_LEFT * FP16_B,
                        (uint32_t)SRC_ROW_BYTES,
                        (uint16_t)SRC_ROW_BYTES,
                        1,
                        (uint16_t)SRC_ROW_BYTES);

                    // Right reflect: dst[SRC_LEN + 1] = src[SRC_LEN - 2].
                    __load(src_row + (uint64_t)(SRC_LEN - 2) * FP16_B,
                        l2_row + (PAD_LEFT + SRC_LEN) * FP16_B,
                        (uint32_t)FP16_B,
                        (uint16_t)FP16_B,
                        1,
                        (uint16_t)FP16_B);
                }

                __store(L2_R, GTX_MAIN_ADDR(BASE_DDR_R),
                    (uint32_t)TOTAL_DST_BYTES,
                    (uint16_t)TOTAL_DST_BYTES,
                    1,
                    (uint16_t)TOTAL_DST_BYTES);
            __end_shared();
        __end_plan(nest_id);
    }
    __join();

    return 0;
}
