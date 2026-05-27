//==================================================================
// {{OP_NAME}} (generated) — concat axis=0 via GTX_LAUNCH<<<1, 0>>>.
// Row-wise DDR→L2→DDR copy: src0 row to front half, src1 row to back half.
// Shared-only (no per-SPU body).
//==================================================================

#include "gtx_kernel.h"
#include "gtx_csr.h"

#define NESTS               1

#define BASE_DDR_SRC0       0x1000000
#define BASE_DDR_SRC1       0x2000000
#define BASE_DDR_RESULT     0xf000000

#define SRC_COLS            {{SRC_COLS}}
#define ROWS                {{ROWS}}
#define FP16_B              2
#define SRC_ROW_BYTES       (SRC_COLS * FP16_B)
#define DST_ROW_BYTES       (SRC_ROW_BYTES * 2)

#define L2_ROW_BUF          0x000000
#define SHARED_ONLY_SPU_MASK 0x0000
#define SHARED_LOAD_TOKEN    0xBEEF

GTX_KERNEL_BODY(
    /* SHARED_BODY */ {
        for (uint32_t row = 0; row < ROWS; ++row) {
            uint32_t src_off = row * SRC_ROW_BYTES;
            uint32_t dst_off = row * DST_ROW_BYTES;

            __load(GTX_MAIN_ADDR(BASE_DDR_SRC0) + src_off, L2_ROW_BUF,
                SRC_ROW_BYTES, (uint16_t)SRC_ROW_BYTES, 1, SRC_ROW_BYTES);
            __credit_ld(SHARED_ONLY_SPU_MASK, SHARED_LOAD_TOKEN);
            __credit_chk(SHARED_ONLY_SPU_MASK);
            __store(L2_ROW_BUF, GTX_MAIN_ADDR(BASE_DDR_RESULT) + dst_off,
                SRC_ROW_BYTES, (uint16_t)SRC_ROW_BYTES, 1, SRC_ROW_BYTES);
            __credit_st(SHARED_ONLY_SPU_MASK);

            __load(GTX_MAIN_ADDR(BASE_DDR_SRC1) + src_off, L2_ROW_BUF,
                SRC_ROW_BYTES, (uint16_t)SRC_ROW_BYTES, 1, SRC_ROW_BYTES);
            __credit_ld(SHARED_ONLY_SPU_MASK, SHARED_LOAD_TOKEN);
            __credit_chk(SHARED_ONLY_SPU_MASK);
            __store(L2_ROW_BUF,
                GTX_MAIN_ADDR(BASE_DDR_RESULT) + dst_off + SRC_ROW_BYTES,
                SRC_ROW_BYTES, (uint16_t)SRC_ROW_BYTES, 1, SRC_ROW_BYTES);
            __credit_st(SHARED_ONLY_SPU_MASK);
        }
    },
    /* THREAD_BODY */ { /* shared-only */ }
)

int main(void) {
    GTX_LAUNCH_SHARED(NESTS);
    return 0;
}
