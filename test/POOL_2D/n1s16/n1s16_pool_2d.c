//==================================================================
// n1s16_pool_2d - ggml POOL_2D average pooling
//
// generate_data.cpp builds:
//   ggml_pool_2d(src0, GGML_OP_POOL_AVG, k0=2, k1=2, s0=2, s1=2, p0=0, p1=0)
//
// Shape [66,50]: input 66x50 FP16 values, output 33x25 FP16 values.
// The target output size is 825 * sizeof(FP16) = 0x672 bytes.
//==================================================================

#ifndef N1S16_POOL_2D_C
#define N1S16_POOL_2D_C

#include "intrin.h"
#include "gtx/address.h"
#include "gtx_csr.h"
#include <stdint.h>

#define NEST_NUM            1
#define SPU_NUM_PER_NEST    16
#define DTYPE               2

#define IN_H                50
#define IN_W                66
#define OUT_H               25
#define OUT_W               33

#define K_H                 2
#define K_W                 2
#define S_H                 2
#define S_W                 2

#define INPUT_ELEMS         (IN_H * IN_W)
#define OUTPUT_ELEMS        (OUT_H * OUT_W)
#define INPUT_BYTES         (INPUT_ELEMS * DTYPE)
#define OUTPUT_BYTES        (OUTPUT_ELEMS * DTYPE)

#define BASE_DDR_A          0x1000000
#define BASE_DDR_RESULT     0xf000000

#define L2_A                0x000000
#define L2_RESULT           0x002000

#define BANK_A              0x00000
#define BANK_B              0x20000
#define BANK_C              0x30000
#define BANK_R              0x50000

// FP16 0.25, the reciprocal of K_H * K_W for average pooling.
#define FP16_QUARTER        0x3400

int main(void) {
    uint8_t nest_id = 0;
    uint16_t active_tid_mask = 0x0001;

    __split();

    {
        __start_plan(nest_id);

            __start_shared();
                __load_cr(GTX_MAIN_ADDR(BASE_DDR_A), L2_A,
                    INPUT_BYTES,
                    (uint16_t)INPUT_BYTES,
                    1, (uint16_t)INPUT_BYTES,
                    1, active_tid_mask, 0xBEEF);

                __credit_chk(active_tid_mask);

                __store_cr(L2_RESULT, GTX_MAIN_ADDR(BASE_DDR_RESULT),
                    OUTPUT_BYTES,
                    (uint16_t)OUTPUT_BYTES,
                    1, (uint16_t)OUTPUT_BYTES,
                    1, active_tid_mask);
            __end_shared();

            for (uint8_t tid = 0; tid < SPU_NUM_PER_NEST; tid++) {
                if (tid == 0) {
                    __start_thread(tid);
                        __set_spm_addr(BANK_R, BANK_C, BANK_B, BANK_A);
                        __credit_chk(0xBEEF);

                        __load_cr(L2_A, BANK_A,
                            INPUT_BYTES,
                            (uint16_t)INPUT_BYTES,
                            1, (uint16_t)INPUT_BYTES,
                            1, active_tid_mask, nest_id);

                        __pool_a(IN_H, IN_W, OUT_H, OUT_W, K_H, K_W, S_H, S_W, FP16_QUARTER);

                        __store_cr(BANK_R, L2_RESULT,
                            OUTPUT_BYTES,
                            (uint16_t)OUTPUT_BYTES,
                            1, (uint16_t)OUTPUT_BYTES,
                            1, active_tid_mask);
                    __end_thread(tid);
                }
            }

        __end_plan(nest_id);
    }

    __join();

    return 0;
}

#endif
