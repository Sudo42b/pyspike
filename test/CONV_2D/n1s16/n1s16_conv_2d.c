/* n1s16_conv_2d.c — single-channel ggml_conv_2d (s=1, p=0, d=1), 1 NEST,
 * 16-way SPU split.
 *
 * Standalone port of libkernels/src/gtx/op_conv_2d.c (op_conv_2d_f16).
 * tensor_view_t parameters are replaced by compile-time constants matching
 * the test/CONV_2D/n1s16 fixture (IW=IH=35, KW=KH=3, single channel/batch,
 * F16). Logic is otherwise byte-identical to the library kernel:
 *
 *   • 1089 outputs partitioned across 16 SPUs (SPU0 gets 69, SPU1..15 get 68).
 *   • Per SPU: load kernel once into BANK_B, then for each owned out_idx
 *     gather K_H patch rows into BANK_C @ PATCH_ADDR.
 *   • The LAST patch row of the LAST owned out_idx is __load_cr — drains the
 *     shared kernel __load_cr (broadcast +1 across all SPUs) once per SPU.
 *   • __dot_product → __store_svr(SLOT_ADDR, 0) → __store of lane-0 FP16 to
 *     L2_RESULT[out_idx]; the LAST owned out_idx uses __store_cr to settle
 *     the shared L2→DDR __store_cr deferred S-loop credit.
 */

#ifndef N1S16_CONV_2D_C
#define N1S16_CONV_2D_C

#include "intrin.h"
#include "gtx/address.h"
#include "gtx_csr.h"

#include <stdint.h>

#define NEST_ID             0
#define SPU_NUM_PER_NEST    16
#define IN_W                35
#define IN_H                35
#define K_W                 3
#define K_H                 3
#define K_ELEMS             (K_W * K_H)
#define OUT_W               (IN_W - K_W + 1)        /* 33 */
#define OUT_H               (IN_H - K_H + 1)        /* 33 */
#define NUM_OUTPUTS         (OUT_W * OUT_H)         /* 1089 */

#define FP16_B              2
#define INPUT_BYTES         (IN_W * IN_H * FP16_B)
#define KERNEL_BYTES        (K_ELEMS * FP16_B)
#define OUTPUT_BYTES        (NUM_OUTPUTS * FP16_B)
#define PATCH_ROW_BYTES     (K_W * FP16_B)

#define ACTIVE_SPU_COUNT    SPU_NUM_PER_NEST
#define ACTIVE_TID_MASK     0xFFFFu

/* 1089 / 16 = 68 r 1 → SPU0 owns 69, SPU1..15 own 68 each. */
#define OUTPUTS_PER_SPU     (NUM_OUTPUTS / ACTIVE_SPU_COUNT)   /* 68 */
#define OUTPUTS_REM         (NUM_OUTPUTS % ACTIVE_SPU_COUNT)   /* 1  */

#define BASE_DDR_KERNEL     0x1000000
#define BASE_DDR_INPUT      0x2000000
#define BASE_DDR_RESULT     0xf000000

#define L2_KERNEL           0x000000
#define L2_INPUT            0x001000
#define L2_RESULT           0x002000

#define BANK_A              0x00000
#define BANK_B              0x20000
#define BANK_C              0x30000
#define BANK_R              0x50000

#define PATCH_ADDR          (BANK_C + 0x000)
#define SLOT_ADDR           (BANK_C + 0x100)

int main(void) {
    const uint8_t nest_id = NEST_ID;

    __split();
    {
        __start_plan(nest_id);

            __start_shared();
                /* Plain load of the data plane — no credit needed. */
                __load(
                    GTX_MAIN_ADDR(BASE_DDR_INPUT), L2_INPUT,
                    INPUT_BYTES, (uint16_t) INPUT_BYTES,
                    1, (uint16_t) INPUT_BYTES
                );

                /* Load kernel weights + raise ld credit (broadcast to all 16
                 * SPUs; each SPU drains via the last-patch-row __load_cr). */
                __load_cr(
                    GTX_MAIN_ADDR(BASE_DDR_KERNEL), L2_KERNEL,
                    KERNEL_BYTES, (uint16_t) KERNEL_BYTES,
                    1, (uint16_t) KERNEL_BYTES,
                    1, ACTIVE_TID_MASK, 0xBEEF
                );

                __credit_chk(ACTIVE_TID_MASK);

                /* Deferred L2 → DDR copy of the entire output plane. Each SPU
                 * settles its st-credit via the per-thread last-output
                 * __store_cr below; pyspike defers the S-loop dec to end.p.   */
                __store_cr(
                    L2_RESULT, GTX_MAIN_ADDR(BASE_DDR_RESULT),
                    OUTPUT_BYTES, (uint16_t) OUTPUT_BYTES,
                    1, (uint16_t) OUTPUT_BYTES,
                    1, ACTIVE_TID_MASK
                );
            __end_shared();

            for (uint8_t tid = 0; tid < (uint8_t) ACTIVE_SPU_COUNT; tid++) {
                const uint16_t tid_mask        = (uint16_t)(0x1u << tid);
                const uint32_t extra_output    = (tid < OUTPUTS_REM) ? 1u : 0u;
                const uint32_t base_output_idx = (uint32_t) tid * OUTPUTS_PER_SPU
                                               + (tid < OUTPUTS_REM ? tid : OUTPUTS_REM);
                const uint32_t thread_outputs  = OUTPUTS_PER_SPU + extra_output;

                __start_thread(tid);
                    __set_spm_addr(BANK_R, BANK_C, BANK_B, BANK_A);
                    __credit_chk(0xBEEF);

                    /* L2_KERNEL → BANK_B (once per SPU). */
                    __load(
                        L2_KERNEL, BANK_B,
                        KERNEL_BYTES, (uint16_t) KERNEL_BYTES,
                        1, (uint16_t) KERNEL_BYTES
                    );

                    for (uint32_t out_local = 0; out_local < thread_outputs; out_local++) {
                        const uint32_t out_idx = base_output_idx + out_local;
                        const uint32_t out_r   = out_idx / OUT_W;
                        const uint32_t out_c   = out_idx % OUT_W;

                        /* Online gather: K_H rows of K_W FP16 into PATCH_ADDR.
                         * The LAST kh of the LAST owned output uses load_cr to
                         * drain the shared kernel broadcast credit. */
                        for (uint32_t kh = 0; kh < K_H; kh++) {
                            const uint32_t src_off   = ((out_r + kh) * IN_W + out_c) * FP16_B;
                            const uint32_t patch_off = PATCH_ADDR + kh * PATCH_ROW_BYTES;

                            if (out_local == thread_outputs - 1 && kh == K_H - 1) {
                                __load_cr(
                                    L2_INPUT + src_off, patch_off,
                                    PATCH_ROW_BYTES, (uint16_t) PATCH_ROW_BYTES,
                                    1, (uint16_t) PATCH_ROW_BYTES,
                                    1, tid_mask, nest_id
                                );
                            } else {
                                __load(
                                    L2_INPUT + src_off, patch_off,
                                    PATCH_ROW_BYTES, (uint16_t) PATCH_ROW_BYTES,
                                    1, (uint16_t) PATCH_ROW_BYTES
                                );
                            }
                        }

                        __set_spm_addr_A(PATCH_ADDR);
                        __dot_product(K_ELEMS, 0);
                        __set_spm_addr_A(BANK_A);

                        __store_svr(SLOT_ADDR, 0);

                        if (out_local == thread_outputs - 1) {
                            __store_cr(
                                SLOT_ADDR,
                                L2_RESULT + out_idx * FP16_B,
                                FP16_B, (uint16_t) FP16_B,
                                1, (uint16_t) FP16_B,
                                1, tid_mask
                            );
                        } else {
                            __store(
                                SLOT_ADDR,
                                L2_RESULT + out_idx * FP16_B,
                                FP16_B, (uint16_t) FP16_B,
                                1, (uint16_t) FP16_B
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
