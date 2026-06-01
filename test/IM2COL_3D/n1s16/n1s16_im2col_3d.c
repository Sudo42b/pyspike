//==================================================================
// Copyright   : (C) Supergate - All Rights Reserved
// Project     : GSF / VTS
// Test        : n1s16_im2col_3d (1 NEST x 16 SPUs = 16 SPUs)
// Description : 3D im2col using shared/thread DMA pattern.
//
//               input:  [ID=4][IH=25][IW=40] FP16 at 0x2000000 (8000B)
//               output: [PATCHES=2622][PATCH_SIZE=18] FP16
//                       at 0xf000000 (94392B)
//
//               Fixed generator parameters for IM2COL_3D: depth 4,
//               kernel 2x3x3, stride 1, no padding, dilation 1,
//               channels 1, batch 1.  The public shape [W,H] binds only
//               IW/IH; this file is aligned to [40,25] so the current
//               harness output size 0x170B8 is self-consistent.
//
//               OD = ID - KD + 1 = 3
//               OH = IH - KH + 1 = 23
//               OW = IW - KW + 1 = 38
//               PATCHES = OD * OH * OW = 2622
//               PATCH_SIZE = KD * KH * KW = 18
//
//               Shared: DDR -> L2 (input volume)
//               Thread: Each SPU extracts contiguous KW rows directly from
//                       L2 into their final patch positions in L1, then
//                       stores patch rows to L2 output region.
//               Shared: L2 -> DDR (output matrix)
//
// Author      : sw.lee
//==================================================================

#ifndef N1S16_IM2COL_3D_C
#define N1S16_IM2COL_3D_C

#include "intrin.h"
#include "gtx/address.h"
#include "gtx_csr.h"

#include <stdint.h>

#define NEST_ID             0
#define NEST_NUM            1
#define SPU_NUM_PER_NEST    16
#define DTYPE               4       // FP16

// Input dimensions
#define ID                  4
#define IH                  25
#define IW                  40

// Kernel dimensions
#define KD                  2
#define KH                  3
#define KW                  3
#define STRIDE              1

// Output dimensions
#define OD                  (ID - KD + 1)   // 3
#define OH                  (IH - KH + 1)   // 23
#define OW                  (IW - KW + 1)   // 38
#define PATCHES             (OD * OH * OW)  // 2622
#define PATCH_SIZE          (KD * KH * KW)  // 18
#define FP16_B              2

#define INPUT_BYTES         (ID * IH * IW * FP16_B)      // 8000B
#define OUTPUT_ROW_BYTES    (PATCH_SIZE * FP16_B)         // 36B
#define OUTPUT_BYTES        (PATCHES * OUTPUT_ROW_BYTES)  // 94392B

// DDR addresses
// IM2COL_3D generate_data writes the convolution kernel at SRC0 and the
// image volume at SRC1; this kernel consumes only the image volume.
#define BASE_DDR_INPUT      0x2000000
#define BASE_DDR_RESULT     0xf000000

// L2 layout
#define L2_A                0x000000    // input volume (8000B)
#define L2_RESULT           0x002000    // output matrix (94392B)

// L1 banks
#define BANK_A              0x00000
#define BANK_B              0x20000
#define BANK_C              0x30000
#define BANK_R              0x50000     // output patch temp

#define ACTIVE_SPU_COUNT    SPU_NUM_PER_NEST
#define ACTIVE_TID_MASK     0xFFFFu
#define PATCHES_PER_SPU     (PATCHES / SPU_NUM_PER_NEST)  // 163 for [40,25]
#define PATCHES_REMAINDER   (PATCHES % SPU_NUM_PER_NEST)

int main(void) {

    const uint8_t nest_id = NEST_ID;

    __split();

    {
        __start_plan(nest_id);

            __start_shared();
                // DDR -> L2: load input volume
                __load_cr(GTX_MAIN_ADDR(BASE_DDR_INPUT), L2_A,
                    (uint32_t)INPUT_BYTES,
                    (uint16_t)(INPUT_BYTES & 0xFFFF),
                    1, (uint16_t)(INPUT_BYTES & 0xFFFF),
                    1, ACTIVE_TID_MASK, 0xBEEF);

                // Wait for all SPUs to finish
                __credit_chk(ACTIVE_TID_MASK);

                // L2 -> DDR: store output matrix as one bounded 2D transfer.
                // OUTPUT_BYTES is larger than uint16, so do not use it as the
                // 16-bit line length; each patch row is only OUTPUT_ROW_BYTES.
                __store_cr(L2_RESULT, GTX_MAIN_ADDR(BASE_DDR_RESULT),
                    (uint32_t)OUTPUT_ROW_BYTES,
                    (uint16_t)OUTPUT_ROW_BYTES,
                    (uint16_t)PATCHES, (uint16_t)OUTPUT_ROW_BYTES,
                    (uint32_t)OUTPUT_ROW_BYTES, ACTIVE_TID_MASK);
            __end_shared();

            for (uint8_t tid = 0; tid < ACTIVE_SPU_COUNT; tid++) {
                const uint16_t tid_mask = (uint16_t)(0x1u << tid);
                const uint32_t extra_patch = tid < PATCHES_REMAINDER ? 1u : 0u;
                const uint32_t base_patch_idx = (uint32_t)tid * PATCHES_PER_SPU
                    + (tid < PATCHES_REMAINDER ? tid : PATCHES_REMAINDER);
                const uint32_t thread_patches = PATCHES_PER_SPU + extra_patch;

                __start_thread(tid);
                    __set_spm_addr(BANK_R, BANK_C, BANK_B, BANK_A);
                    __credit_chk(0xBEEF);

                    // Each SPU processes its contiguous patch range.
                    for (uint32_t p = 0; p < thread_patches; p++) {
                        const uint32_t patch_idx = base_patch_idx + p;

                        // Decompose patch index into (od, oh, ow)
                        const uint32_t od = patch_idx / (OH * OW);
                        const uint32_t oh = (patch_idx / OW) % OH;
                        const uint32_t ow = patch_idx % OW;

                        // Extract KD*KH contiguous KW rows directly into the
                        // patch row locations in L1.  The row loads are small
                        // because im2col gathers non-contiguous source rows;
                        // avoiding L1 scalar pointer copies keeps movement in
                        // GTX DMA/intrinsic form.
                        for (uint32_t kd = 0; kd < KD; kd++) {
                            for (uint32_t kh = 0; kh < KH; kh++) {
                                const uint32_t id_idx = od * STRIDE + kd;
                                const uint32_t ih_idx = oh * STRIDE + kh;
                                const uint32_t iw_idx = ow * STRIDE;
                                const uint32_t src_off =
                                    (id_idx * IH * IW + ih_idx * IW + iw_idx) * FP16_B;
                                const uint32_t dst_off = (kd * KH + kh) * KW * FP16_B;

                                if (p == thread_patches - 1 && kd == KD - 1 && kh == KH - 1) {
                                    __load_cr(L2_A + src_off, BANK_R + dst_off,
                                        (uint32_t)(KW * FP16_B), (uint16_t)(KW * FP16_B),
                                        1, (uint16_t)(KW * FP16_B),
                                        1, tid_mask, nest_id);
                                } else {
                                    __load(L2_A + src_off, BANK_R + dst_off,
                                        (uint32_t)(KW * FP16_B), (uint16_t)(KW * FP16_B),
                                        1, (uint16_t)(KW * FP16_B));
                                }
                            }
                        }

                        // L1 -> L2: store output patch from Bank R
                        const uint32_t out_off = patch_idx * OUTPUT_ROW_BYTES;
                        if (p == thread_patches - 1) {
                            __store_cr(BANK_R, L2_RESULT + out_off,
                                OUTPUT_ROW_BYTES, (uint16_t)OUTPUT_ROW_BYTES,
                                1, (uint16_t)OUTPUT_ROW_BYTES,
                                1, tid_mask);
                        } else {
                            __store(BANK_R, L2_RESULT + out_off,
                                OUTPUT_ROW_BYTES, (uint16_t)OUTPUT_ROW_BYTES,
                                1, (uint16_t)OUTPUT_ROW_BYTES);
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
