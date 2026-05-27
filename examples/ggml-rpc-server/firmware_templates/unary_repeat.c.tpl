//==================================================================
// {{OP_NAME}} (generated) — 4D broadcast tile via GTX_LAUNCH<<<1, 0>>>.
// dst[i3, i2, i1, i0] = src[i3 % S3, i2 % S2, i1 % S1, i0 % S0].
// Shared-only kernel: pure DMA, no per-SPU work. Each DST_NEx must be an
// integer multiple of SRC_NEx; both src and dst must fit a 512KB L2 pool.
//==================================================================

#include "gtx_kernel.h"

#define NESTS               1
#define DTYPE               2

#define SRC_NE0             {{SRC_NE0}}
#define SRC_NE1             {{SRC_NE1}}
#define SRC_NE2             {{SRC_NE2}}
#define SRC_NE3             {{SRC_NE3}}
#define DST_NE0             {{DST_NE0}}
#define DST_NE1             {{DST_NE1}}
#define DST_NE2             {{DST_NE2}}
#define DST_NE3             {{DST_NE3}}

#define REP0                (DST_NE0 / SRC_NE0)

#define BASE_DDR_A          0x1000000
#define BASE_DDR_R          0xf000000

#define SRC_ROW_BYTES       (SRC_NE0 * DTYPE)
#define DST_ROW_BYTES       (DST_NE0 * DTYPE)
#define SRC_BYTES           (SRC_NE0 * SRC_NE1 * SRC_NE2 * SRC_NE3 * DTYPE)
#define DST_BYTES           (DST_NE0 * DST_NE1 * DST_NE2 * DST_NE3 * DTYPE)
#define SRC_TOTAL_ROWS      (SRC_NE1 * SRC_NE2 * SRC_NE3)
#define DST_TOTAL_ROWS      (DST_NE1 * DST_NE2 * DST_NE3)

#define L2_A                0x000000
#define L2_RESULT           0x080000

#if SRC_BYTES > 0x80000
#error "repeat src exceeds L2_A budget"
#endif
#if DST_BYTES > 0x80000
#error "repeat dst exceeds L2_RESULT budget"
#endif
#if (DST_NE0 % SRC_NE0) != 0 || (DST_NE1 % SRC_NE1) != 0 \
    || (DST_NE2 % SRC_NE2) != 0 || (DST_NE3 % SRC_NE3) != 0
#error "repeat requires each DST_NEx % SRC_NEx == 0"
#endif

GTX_KERNEL_BODY(
    /* SHARED_BODY */ {
        // Stage src into L2 as a flat stream of SRC_TOTAL_ROWS rows.
        __load(GTX_MAIN_ADDR(BASE_DDR_A), L2_A,
            (uint32_t)SRC_ROW_BYTES,
            (uint16_t)SRC_ROW_BYTES,
            (uint16_t)SRC_TOTAL_ROWS,
            (uint16_t)SRC_ROW_BYTES);

        // Walk the dst index space; per (i3, i2, i1) tile, copy the
        // matching src row REP0 times along the inner (col) dim.
        for (uint32_t i3 = 0; i3 < DST_NE3; i3++) {
            uint32_t s3 = i3 % SRC_NE3;
            for (uint32_t i2 = 0; i2 < DST_NE2; i2++) {
                uint32_t s2 = i2 % SRC_NE2;
                for (uint32_t i1 = 0; i1 < DST_NE1; i1++) {
                    uint32_t s1 = i1 % SRC_NE1;
                    uint32_t src_off = ((s3 * SRC_NE2 + s2) * SRC_NE1 + s1)
                                        * SRC_ROW_BYTES;
                    uint32_t dst_off = ((i3 * DST_NE2 + i2) * DST_NE1 + i1)
                                        * DST_ROW_BYTES;
                    for (uint32_t r0 = 0; r0 < REP0; r0++) {
                        __copy(L2_A + src_off,
                            L2_RESULT + dst_off + r0 * SRC_ROW_BYTES,
                            (uint32_t)SRC_ROW_BYTES,
                            (uint16_t)SRC_ROW_BYTES, 1,
                            (uint16_t)SRC_ROW_BYTES);
                    }
                }
            }
        }

        __store(L2_RESULT, GTX_MAIN_ADDR(BASE_DDR_R),
            (uint32_t)DST_ROW_BYTES,
            (uint16_t)DST_ROW_BYTES,
            (uint16_t)DST_TOTAL_ROWS,
            (uint16_t)DST_ROW_BYTES);
    },
    /* THREAD_BODY */ { /* shared-only */ }
)

int main(void) {
    GTX_LAUNCH_SHARED(NESTS);
    return 0;
}
