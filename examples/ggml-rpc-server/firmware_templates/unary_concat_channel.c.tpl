//==================================================================
// {{OP_NAME}} (generated) — channel concat for ggml CONCAT axis=2.
// src0 shape (W, H, A_CH, BATCH), src1 shape (W, H, B_CH, BATCH).
// dst shape (W, H, A_CH+B_CH, BATCH), row-major.
//
// Per batch: dst[batch] = a[batch] (A_CH*H*W) ‖ b[batch] (B_CH*H*W) along
// the channel dim. Pure DDR→DDR memcpy via __copy_mem; no L2/L1 staging,
// no SPU thread body.
//==================================================================

#include "gtx_kernel.h"

#define NESTS               1
#define DTYPE               2

#define W                   {{W}}
#define H                   {{H}}
#define A_CH                {{A_CH}}
#define B_CH                {{B_CH}}
#define BATCH               {{BATCH}}

#define BASE_DDR_A          0x1000000
#define BASE_DDR_B          0x2000000
#define BASE_DDR_RESULT     0xf000000

#define HW_BYTES_RAW        (H * W * DTYPE)
#define A_BATCH_BYTES_RAW   (A_CH * HW_BYTES_RAW)
#define B_BATCH_BYTES_RAW   (B_CH * HW_BYTES_RAW)
#define DST_BATCH_BYTES_RAW (A_BATCH_BYTES_RAW + B_BATCH_BYTES_RAW)

#if HW_BYTES_RAW == 0
#error "concat_channel needs non-zero H*W"
#endif
#if A_BATCH_BYTES_RAW > 16777215 || B_BATCH_BYTES_RAW > 16777215
#error "concat_channel per-batch chunk exceeds 24-bit DMA length cap"
#endif

#define HW_BYTES            ((uint32_t)HW_BYTES_RAW)
#define A_BATCH_BYTES       ((uint32_t)A_BATCH_BYTES_RAW)
#define B_BATCH_BYTES       ((uint32_t)B_BATCH_BYTES_RAW)
#define DST_BATCH_BYTES     ((uint32_t)DST_BATCH_BYTES_RAW)

GTX_KERNEL_BODY(
    /* SHARED_BODY */ {
        for (uint32_t batch = 0; batch < BATCH; batch++) {
            uint32_t src_a_off = batch * A_BATCH_BYTES;
            uint32_t src_b_off = batch * B_BATCH_BYTES;
            uint32_t dst_off   = batch * DST_BATCH_BYTES;

            // a[batch] → dst[batch][:A_CH]
            __copy_mem(
                GTX_MAIN_ADDR(BASE_DDR_A) + src_a_off,
                GTX_MAIN_ADDR(BASE_DDR_RESULT) + dst_off,
                A_BATCH_BYTES,
                (uint16_t)A_BATCH_BYTES,
                1,
                (uint16_t)A_BATCH_BYTES,
                (uint16_t)(A_BATCH_BYTES >> 16));

            // b[batch] → dst[batch][A_CH:A_CH+B_CH]
            __copy_mem(
                GTX_MAIN_ADDR(BASE_DDR_B) + src_b_off,
                GTX_MAIN_ADDR(BASE_DDR_RESULT) + dst_off + A_BATCH_BYTES,
                B_BATCH_BYTES,
                (uint16_t)B_BATCH_BYTES,
                1,
                (uint16_t)B_BATCH_BYTES,
                (uint16_t)(B_BATCH_BYTES >> 16));
        }
    },
    /* THREAD_BODY */ { /* shared-only */ }
)

int main(void) {
    GTX_LAUNCH_SHARED(NESTS);
    return 0;
}
