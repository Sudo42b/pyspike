#ifndef N1S16_CONV_TR1D_C
#define N1S16_CONV_TR1D_C

#include "intrin.h"
#include "gtx/address.h"
#include "gtx_csr.h"

#include <stdint.h>

#define NEST_ID             0
#define SPU_NUM_PER_NEST    16

#define FP16_B              2

#define K_SIZE              3
#define STRIDE              1
#define CIN                 1
#define COUT                1

#define L_IN                32765
#define L_OUT               ((L_IN - 1) * STRIDE + K_SIZE)

#define OUTPUT_BYTES        (L_OUT * FP16_B)
#define INPUT_BYTES         (L_IN * FP16_B)
#define KERNEL_BYTES        (K_SIZE * FP16_B)

#define ACTIVE_TID_MASK     0xFFFFu

#define BASE_DDR_KERNEL     0x1000000
#define BASE_DDR_INPUT      0x2000000
#define BASE_DDR_RESULT     0xf000000

#define L2_ALIGN_UP(x, a)   (((x) + ((a) - 1u)) & ~((a) - 1u))
#define L2_KERNEL           0x000000
#define L2_INPUT            0x001000
#define L2_RESULT           L2_ALIGN_UP(L2_INPUT + INPUT_BYTES, 0x1000u)

#define BANK_A              0x00000
#define BANK_B              0x20000
#define BANK_C              0x30000
#define BANK_R              0x50000

#define SLOT_ADDR           (BANK_C + 0x100)

int main(void) {
    const uint8_t nest_id = NEST_ID;

    __split();

    {
        __start_plan(nest_id);

            __start_shared();
                __load(
                    GTX_MAIN_ADDR(BASE_DDR_INPUT), L2_INPUT,
                    INPUT_BYTES, (uint16_t) INPUT_BYTES,
                    1, (uint16_t) INPUT_BYTES
                );

                __load_cr(
                    GTX_MAIN_ADDR(BASE_DDR_KERNEL), L2_KERNEL,
                    KERNEL_BYTES, (uint16_t) KERNEL_BYTES,
                    1, (uint16_t) KERNEL_BYTES,
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

            for (uint8_t tid = 0; tid < SPU_NUM_PER_NEST; tid++) {
                const uint16_t tid_mask = (uint16_t) (0x1u << tid);
                const uint32_t output_start = ((uint32_t) L_OUT * tid) / SPU_NUM_PER_NEST;
                const uint32_t output_end = ((uint32_t) L_OUT * (tid + 1u)) / SPU_NUM_PER_NEST;
                const uint32_t outputs_this_spu = output_end - output_start;

                __start_thread(tid);
                    __set_spm_addr(BANK_R, BANK_C, BANK_B, BANK_A);
                    __credit_chk(0xBEEF);

                    __load(
                        L2_KERNEL + 2 * FP16_B, BANK_B + 0 * FP16_B,
                        FP16_B, (uint16_t) FP16_B,
                        1, (uint16_t) FP16_B
                    );
                    __load(
                        L2_KERNEL + 1 * FP16_B, BANK_B + 1 * FP16_B,
                        FP16_B, (uint16_t) FP16_B,
                        1, (uint16_t) FP16_B
                    );
                    __load(
                        L2_KERNEL + 0 * FP16_B, BANK_B + 2 * FP16_B,
                        FP16_B, (uint16_t) FP16_B,
                        1, (uint16_t) FP16_B
                    );

                    for (uint32_t out_local = 0; out_local < outputs_this_spu; out_local++) {
                        const uint32_t out_idx = output_start + out_local;
                        uint32_t input_addr = L2_INPUT;
                        uint32_t input_bytes = KERNEL_BYTES;
                        uint32_t kernel_addr = BANK_B;
                        const uint8_t is_last_output = (out_local + 1u == outputs_this_spu);

                        if (out_idx == 0) {
                            input_addr = L2_INPUT + 0 * FP16_B;
                            input_bytes = 1 * FP16_B;
                            kernel_addr = BANK_B + 2 * FP16_B;
                        } else if (out_idx == 1) {
                            input_addr = L2_INPUT + 0 * FP16_B;
                            input_bytes = 2 * FP16_B;
                            kernel_addr = BANK_B + 1 * FP16_B;
                        } else if (out_idx < (uint32_t) L_IN) {
                            input_addr = L2_INPUT + (out_idx - 2u) * FP16_B;
                            input_bytes = KERNEL_BYTES;
                            kernel_addr = BANK_B;
                        } else if (out_idx == (uint32_t) L_IN) {
                            input_addr = L2_INPUT + (L_IN - 2) * FP16_B;
                            input_bytes = 2 * FP16_B;
                            kernel_addr = BANK_B + 0 * FP16_B;
                        } else {
                            input_addr = L2_INPUT + (L_IN - 1) * FP16_B;
                            input_bytes = 1 * FP16_B;
                            kernel_addr = BANK_B + 0 * FP16_B;
                        }

                        if (is_last_output) {
                            __load_cr(
                                input_addr, BANK_A,
                                input_bytes, (uint16_t) input_bytes,
                                1, (uint16_t) input_bytes,
                                1, tid_mask, nest_id
                            );
                        } else {
                            __load(
                                input_addr, BANK_A,
                                input_bytes, (uint16_t) input_bytes,
                                1, (uint16_t) input_bytes
                            );
                        }

                        __set_spm_addr_B(kernel_addr);
                        __dot_product(input_bytes / FP16_B, 0);
                        __set_spm_addr_B(BANK_B);

                        __store_svr(SLOT_ADDR, 0);

                        if (is_last_output) {
                            __store_cr(
                                SLOT_ADDR, L2_RESULT + out_idx * FP16_B,
                                FP16_B, (uint16_t) FP16_B,
                                1, (uint16_t) FP16_B,
                                1, tid_mask
                            );
                        } else {
                            __store(
                                SLOT_ADDR, L2_RESULT + out_idx * FP16_B,
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
