//==================================================================
// Copyright   : (C) Supergate - All Rights Reserved
// Project     : GSF / VTS
// Test        : yaml_get_rel_pos (YAML-scale)
// Description : GTX DMA extraction of relative position bias (SAM).
//               Extracts a 2D slice from the relative position table
//               using GTX DDR->L2->DDR row movement.
//
//               a:   [H=5, W_FEAT=3] FP16 at 0x1000000 (30B)
//               dst: [QH=3, KH=3, W_FEAT=3] FP16 at 0xf000000 (54B)
//
//               H = 2 * max(QH, KH) - 1 = 2*3 - 1 = 5
//
//               For each (q, k) in [0..QH-1] x [0..KH-1]:
//                 pos = (KH - 1 - k) + q
//                 dst[q][k][:] = a[pos][:]
//
//               Each inner transfer is W_FEAT*2 bytes (contiguous feature row).
//               Per-row GTX DMA transfers are required because GET_REL_POS
//               gathers rows in a non-contiguous order, so a single bulk linear
//               DMA cannot express the layout. The gather is issued inside the
//               canonical single split/plan/shared region through an L2 row
//               buffer rather than DDR-to-DDR copy.mem.
//
// Author      : sw.lee
//==================================================================

#ifndef YAML_GET_REL_POS_C
#define YAML_GET_REL_POS_C

#include <stdint.h>

#include "intrin.h"
#include "gtx/address.h"
#include "gtx_csr.h"

// Dimensions
#define REL_H           7       // a.ne[1] = 2 * max(QH, KH) - 1
#define W_FEAT          5       // feature dimension
#define QH              4       // query spatial size
#define KH              4       // key spatial size
#define FP16_B          2

#define ROW_BYTES       (W_FEAT * FP16_B)  // 6 bytes per feature row

#define OUTPUT_BYTES    160     // QH * KH * W_FEAT * 2 = 4*4*5*2

// L2 addresses and shared-only credit tokens
#define L2_ROW_BUF      0x000000
#define SHARED_ONLY_SPU_MASK 0x0000u
#define SHARED_LOAD_TOKEN    0xBEEF

// DDR addresses
#define BASE_DDR_A      0x1000000       // a[H=5][W_FEAT=3] FP16 (30B)
#define BASE_DDR_RESULT 0xf000000       // dst[QH=3][KH=3][W_FEAT=3] FP16 (54B)

int main(void) {
    const uint8_t nest_id = 0;

    // Extract relative position bias using DMA:
    // pos = (KH - 1 - k) + q; dst[q][k][:] = a[pos][:].
    // The q/k address computation is bounded metadata/control logic; all data
    // movement is performed by GTX DDR->L2->DDR DMA, not CPU scalar element copies.
    __split();
    __start_plan(nest_id);
    __start_shared();
    for (int q = 0; q < QH; ++q) {
        for (int k = 0; k < KH; ++k) {
            const int pos = (KH - 1 - k) + q;
            const uint32_t src_off = (uint32_t)(pos * W_FEAT) * FP16_B;
            const uint32_t dst_off = (uint32_t)(q * KH * W_FEAT + k * W_FEAT) * FP16_B;

            __load(
                GTX_MAIN_ADDR(BASE_DDR_A)      + src_off,
                L2_ROW_BUF,
                ROW_BYTES,
                (uint16_t) ROW_BYTES,
                1,
                (uint16_t) ROW_BYTES
            );

            // Shared-only credit boundary keeps L2_ROW_BUF reuse ordered while
            // avoiding an SPU thread block for this pure data-movement kernel.
            __credit_ld(SHARED_ONLY_SPU_MASK, SHARED_LOAD_TOKEN);
            __credit_chk(SHARED_ONLY_SPU_MASK);

            __store(
                L2_ROW_BUF,
                GTX_MAIN_ADDR(BASE_DDR_RESULT) + dst_off,
                ROW_BYTES,
                (uint16_t) ROW_BYTES,
                1,
                (uint16_t) ROW_BYTES
            );

            __credit_st(SHARED_ONLY_SPU_MASK);
        }
    }
    __end_shared();
    __end_plan(nest_id);
    __join();

    return 0;
}

#endif
