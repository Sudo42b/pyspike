//==================================================================
// {{OP_NAME}} (generated) — multi-IC IM2COL.
// Generalises the single-channel reference (user supplied) to arbitrary IC.
// Layout:
//   input:  (IC, IN_H, IN_W) row-major in DDR_B (= BASE_DDR_INPUT)
//   output: (NUM_PATCHES, IC*K_H*K_W) row-major in DDR_R, with innermost
//           order KW, KH, IC (matching ggml's im2col).
// Constraints: stride is uniform (STRIDE), pad=0, dilation=1, batch=1.
//==================================================================

#include "gtx/intrin.h"
#include "gtx/address.h"
#include "gtx/csr.h"
#include <stdint.h>

#define NEST_ID             0
#define SPU_NUM_PER_NEST    16
#define ACTIVE_TID_MASK     0xFFFFu

#define FP16_B              2

#define IC                  {{IC}}
#define IN_W                {{IN_W}}
#define IN_H                {{IN_H}}
#define K_W                 {{K_W}}
#define K_H                 {{K_H}}
#define STRIDE              {{STRIDE}}
#define STRIDE_W            STRIDE
#define STRIDE_H            STRIDE
#define DILATION_W          1
#define DILATION_H          1

#define OUT_W               ((IN_W - DILATION_W * (K_W - 1) - 1) / STRIDE_W + 1)
#define OUT_H               ((IN_H - DILATION_H * (K_H - 1) - 1) / STRIDE_H + 1)
#define NUM_PATCHES         (OUT_W * OUT_H)

#define INPUT_ROW_BYTES     (IN_W * FP16_B)
#define INPUT_PLANE_BYTES   (IN_H * INPUT_ROW_BYTES)
#define INPUT_BYTES         (IC * INPUT_PLANE_BYTES)

#define PATCH_ROW_BYTES     (K_W * FP16_B)
#define PATCH_KH_BYTES      (K_H * PATCH_ROW_BYTES)
#define PATCH_BYTES         (IC * PATCH_KH_BYTES)         // IC*K_H*K_W*2
#define OUTPUT_BYTES        (NUM_PATCHES * PATCH_BYTES)

#if PATCH_BYTES > 65535
#error "im2col_multich PATCH_BYTES exceeds 16-bit DMA length cap"
#endif
#if NUM_PATCHES > 65535
#error "im2col_multich NUM_PATCHES exceeds 16-bit DMA height cap"
#endif

#define BASE_DDR_INPUT      0x2000000
#define BASE_DDR_RESULT     0xf000000

#define ALIGN_UP_32(x)      (((x) + 31u) & ~31u)
#define L2_INPUT            0x000000
#define L2_RESULT           ALIGN_UP_32(L2_INPUT + INPUT_BYTES)

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
                // Stage the whole (IC*IN_H*IN_W) FP16 cube into L2_INPUT.
                __load_cr(
                    GTX_MAIN_ADDR(BASE_DDR_INPUT), L2_INPUT,
                    INPUT_ROW_BYTES, (uint16_t) INPUT_ROW_BYTES,
                    (uint16_t)(IC * IN_H), INPUT_ROW_BYTES,
                    1, ACTIVE_TID_MASK, 0xBEEF
                );

                __credit_chk(ACTIVE_TID_MASK);

                // length=PATCH_BYTES (per-patch row), height=NUM_PATCHES.
                // OUTPUT_BYTES > 65535 in YOLO shapes, so we split the store
                // along patches instead of one big contiguous transfer.
                __store_cr(
                    L2_RESULT, GTX_MAIN_ADDR(BASE_DDR_RESULT),
                    PATCH_BYTES, (uint16_t) PATCH_BYTES,
                    (uint16_t) NUM_PATCHES, PATCH_BYTES,
                    1, ACTIVE_TID_MASK
                );
            __end_shared();

            for (uint8_t tid = 0; tid < SPU_NUM_PER_NEST; tid++) {
                const uint16_t tid_mask = (uint16_t)(0x1u << tid);
                const uint32_t patch_begin = ((uint32_t)NUM_PATCHES * tid) / SPU_NUM_PER_NEST;
                const uint32_t patch_end = ((uint32_t)NUM_PATCHES * (tid + 1u)) / SPU_NUM_PER_NEST;
                const uint32_t patches_this_spu = patch_end - patch_begin;

                __start_thread(tid);
                    __set_spm_addr(BANK_R, BANK_C, BANK_B, BANK_A);

                    if (patches_this_spu > 0u) {
                        __credit_chk(0xBEEF);

                        for (uint32_t local_patch = 0; local_patch < patches_this_spu; local_patch++) {
                            const uint32_t patch_idx = patch_begin + local_patch;
                            const uint32_t out_y = patch_idx / OUT_W;
                            const uint32_t out_x = patch_idx - out_y * OUT_W;
                            const uint32_t spatial_off =
                                (out_y * STRIDE_H) * INPUT_ROW_BYTES +
                                (out_x * STRIDE_W) * FP16_B;
                            const uint32_t dst_base = patch_idx * PATCH_BYTES;
                            const uint8_t is_last_patch = (local_patch + 1u == patches_this_spu);

                            // Load IC × K_H rows into BANK_A; KW innermost.
                            for (uint32_t ic_i = 0; ic_i < IC; ic_i++) {
                                uint32_t ic_off = ic_i * INPUT_PLANE_BYTES + spatial_off;
                                for (uint32_t kh_i = 0; kh_i < K_H; kh_i++) {
                                    uint32_t src = L2_INPUT + ic_off + (kh_i * DILATION_H) * INPUT_ROW_BYTES;
                                    uint32_t dst_bank = BANK_A + (ic_i * K_H + kh_i) * PATCH_ROW_BYTES;
                                    int last_load = (is_last_patch
                                                     && ic_i == IC - 1u
                                                     && kh_i == K_H - 1u);
                                    if (last_load) {
                                        __load_cr(src, dst_bank,
                                            PATCH_ROW_BYTES, (uint16_t)PATCH_ROW_BYTES,
                                            1, (uint16_t)PATCH_ROW_BYTES,
                                            1, tid_mask, nest_id);
                                    } else {
                                        __load(src, dst_bank,
                                            PATCH_ROW_BYTES, (uint16_t)PATCH_ROW_BYTES,
                                            1, (uint16_t)PATCH_ROW_BYTES);
                                    }
                                }
                            }

                            if (is_last_patch) {
                                __store_cr(BANK_A, L2_RESULT + dst_base,
                                    PATCH_BYTES, (uint16_t)PATCH_BYTES,
                                    1, (uint16_t)PATCH_BYTES,
                                    1, tid_mask);
                            } else {
                                __store(BANK_A, L2_RESULT + dst_base,
                                    PATCH_BYTES, (uint16_t)PATCH_BYTES,
                                    1, (uint16_t)PATCH_BYTES);
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
