//==================================================================
// n1s16_fill — fill tensor with ggml_fill constant, 1 NEST x 16 SPUs
//
// ggml_fill(ctx, input, 0.25f) ignores input values and writes the op
// parameter to every output element.  The current generator exports FP16
// references, so this kernel fills L2 with the FP16 bit pattern for 0.25
// and stores the contiguous result to DDR.
//==================================================================

#include <stdint.h>

#include "intrin.h"
#include "gtx/address.h"

#define NEST_ID             0
#define DTYPE               4

#define WIDTH               511
#define HEIGHT              64

#define BASE_DDR_RESULT     0xf000000

#define L2_RESULT           0x002000

#define OUTPUT_BYTES        (WIDTH * HEIGHT * DTYPE)

// FP16 0.25f is 0x3400.  Replicate across the 64-bit fill pattern.
#define FP16_FILL_VALUE     0x3400ull
#define FILL_PATTERN        ((FP16_FILL_VALUE << 48) | \
                             (FP16_FILL_VALUE << 32) | \
                             (FP16_FILL_VALUE << 16) | \
                              FP16_FILL_VALUE)

int main(void) {
    __split();

    {
        __start_plan(NEST_ID);

            __start_shared();
                __fill(L2_RESULT,
                    (uint32_t)OUTPUT_BYTES,
                    (uint16_t)OUTPUT_BYTES,
                    1,
                    FILL_PATTERN,
                    0);

                __store(L2_RESULT,
                    GTX_MAIN_ADDR(BASE_DDR_RESULT),
                    (uint32_t)OUTPUT_BYTES,
                    (uint16_t)OUTPUT_BYTES,
                    1,
                    (uint16_t)OUTPUT_BYTES);
            __end_shared();

        __end_plan(NEST_ID);
    }

    __join();
    return 0;
}
