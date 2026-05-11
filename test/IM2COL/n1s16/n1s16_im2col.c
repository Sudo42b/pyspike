//==================================================================
// Copyright   : (C) Supergate - All Rights Reserved
// Project     : GSF / VTS
// Test        : n1s16_im2col (1 NEST x 16 SPUs = 16 SPUs)
// Description : 2D im2col for fixed generator shape [10,10].
//
//               input image:  [10][10] FP16 at 0x2000000
//               output cols:  [64][9]  FP16 at 0xf000000
//
//               Fixed generator parameters: 3x3 kernel, stride 1,
//               pad 0, dilation 1, channels 1, batch 1.
//==================================================================

#ifndef N1S16_IM2COL_C
#define N1S16_IM2COL_C

#include "intrin.h"
#include "gtx/address.h"
#include "gtx_csr.h"

#include <stdint.h>

#define NEST_ID             0
#define SPU_NUM_PER_NEST    16

#define IN_W                17
#define IN_H                19
#define K_W                 3
#define K_H                 3
#define STRIDE              1

#define OUT_W               (IN_W - K_W + 1)
#define OUT_H               (IN_H - K_H + 1)
#define NUM_PATCHES         (OUT_W * OUT_H)
#define PATCH_ELEMS         (K_W * K_H)

#define FP16_B              2
#define INPUT_BYTES         (IN_W * IN_H * FP16_B)
#define PATCH_ROW_BYTES     (K_W * FP16_B)
#define PATCH_BYTES         (PATCH_ELEMS * FP16_B)
#define OUTPUT_BYTES        (NUM_PATCHES * PATCH_BYTES)

#define ACTIVE_SPU_COUNT    SPU_NUM_PER_NEST
#define ACTIVE_TID_MASK     0xFFFFu
#define PATCHES_PER_SPU     (NUM_PATCHES / ACTIVE_SPU_COUNT)
#define PATCHES_REMAINDER   (NUM_PATCHES % ACTIVE_SPU_COUNT)

// IM2COL generate_data writes the convolution kernel at SRC0 and the image at SRC1.
#define BASE_DDR_INPUT      0x2000000
#define BASE_DDR_RESULT     0xf000000

#define L2_INPUT            0x000000
#define L2_RESULT           0x002000

#define BANK_A              0x00000
#define BANK_B              0x20000
#define BANK_C              0x30000
#define BANK_R              0x50000

int main(void) {
    const uint8_t nest_id = NEST_ID;

    __split();

    {
        __start_plan(nest_id);

            __start_shared();
                __load_cr(
                    GTX_MAIN_ADDR(BASE_DDR_INPUT), L2_INPUT,
                    INPUT_BYTES, (uint16_t) INPUT_BYTES,
                    1, (uint16_t) INPUT_BYTES,
                    1, ACTIVE_TID_MASK, 0xBEEF
                );

                __credit_chk(ACTIVE_TID_MASK);

                __store_cr(
                    L2_RESULT, GTX_MAIN_ADDR(BASE_DDR_RESULT),
                    OUTPUT_BYTES, (uint16_t) OUTPUT_BYTES,
                    1, (uint16_t) OUTPUT_BYTES,
                    1, ACTIVE_TID_MASK
                );
            __end_shared();

            for (uint8_t tid = 0; tid < ACTIVE_SPU_COUNT; tid++) {
                const uint16_t tid_mask = (uint16_t) (0x1u << tid);
                const uint32_t extra_patch = tid < PATCHES_REMAINDER ? 1u : 0u;
                const uint32_t base_patch_idx = (uint32_t) tid * PATCHES_PER_SPU
                    + (tid < PATCHES_REMAINDER ? tid : PATCHES_REMAINDER);
                const uint32_t thread_patches = PATCHES_PER_SPU + extra_patch;

                __start_thread(tid);
                    __set_spm_addr(BANK_R, BANK_C, BANK_B, BANK_A);
                    __credit_chk(0xBEEF);

                    for (uint32_t p = 0; p < thread_patches; p++) {
                        const uint32_t patch_idx = base_patch_idx + p;
                        const uint32_t out_r = patch_idx / OUT_W;
                        const uint32_t out_c = patch_idx % OUT_W;

                        for (uint32_t kh = 0; kh < K_H; kh++) {
                            const uint32_t src_off = ((out_r * STRIDE + kh) * IN_W
                                + out_c * STRIDE) * FP16_B;
                            const uint32_t dst_off = kh * PATCH_ROW_BYTES;

                            if (p == thread_patches - 1 && kh == K_H - 1) {
                                __load_cr(
                                    L2_INPUT + src_off, BANK_R + dst_off,
                                    PATCH_ROW_BYTES, (uint16_t) PATCH_ROW_BYTES,
                                    1, (uint16_t) PATCH_ROW_BYTES,
                                    1, tid_mask, nest_id
                                );
                            } else {
                                __load(
                                    L2_INPUT + src_off, BANK_R + dst_off,
                                    PATCH_ROW_BYTES, (uint16_t) PATCH_ROW_BYTES,
                                    1, (uint16_t) PATCH_ROW_BYTES
                                );
                            }
                        }

                        if (p == thread_patches - 1) {
                            __store_cr(
                                BANK_R, L2_RESULT + patch_idx * PATCH_BYTES,
                                PATCH_BYTES, (uint16_t) PATCH_BYTES,
                                1, (uint16_t) PATCH_BYTES,
                                1, tid_mask
                            );
                        } else {
                            __store(
                                BANK_R, L2_RESULT + patch_idx * PATCH_BYTES,
                                PATCH_BYTES, (uint16_t) PATCH_BYTES,
                                1, (uint16_t) PATCH_BYTES
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

#endif
