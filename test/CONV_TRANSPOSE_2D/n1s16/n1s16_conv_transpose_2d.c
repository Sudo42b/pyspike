// =================================================================
// GGML_OP_CONV_TRANSPOSE_2D — 2D transposed convolution (n1s16)
// Input: [IH=4][IW=6] FP16, Kernel: [KH=3][KW=3] FP16, Stride 1
// Output: [OH=6][OW=8] = 48 FP16 values
//
// 16-SPU parallel: output rows distributed across SPUs.
// OH=6 -> SPU 0-5 active (one output row each), SPU 6-15 idle.
// Each SPU accumulates only contributions to its assigned output row,
// eliminating scatter conflicts between SPUs.
//
// Scatter-accumulate: for each (ih,iw) × (kh,kw):
//   oh_out = ih*S + kh;  ow_out = iw*S + kw;
//   output[oh_out][ow_out] += input[ih][iw] * kernel[kh][kw]
// =================================================================

#include "intrin.h"
#include "gtx/address.h"
#include "gtx_csr.h"

#ifndef NEST_ID
#define NEST_ID             0
#endif
#ifndef SPU_ID
#define SPU_ID              0
#endif

#define SPU_NUM_PER_NEST    16

#define BASE_DDR_KERNEL     0x1000000
#define BASE_DDR_INPUT      0x2000000
#define BASE_DDR_RESULT     0xf000000

#define BASE_L2_INPUT       0x000000
#define BASE_L2_KERNEL      0x100000
#define BASE_L2_RESULT      0x200000

#define BANK_A              0x00000
#define BANK_B              0x20000
#define BANK_C              0x30000
#define BANK_R              0x50000

#define IH                  14
#define IW                  30
#define KH                  3
#define KW                  3
#define S                   1
#define OH                  (IH + KH - 1)   // 6
#define OW                  (IW + KW - 1)   // 8
#define FP16_B              2
#define SVR_BYTES           32

#define IN_BYTES            (IH * IW * FP16_B)          // 48B
#define KERNEL_BYTES        (KH * KW * FP16_B)          // 18B
#define OUT_ROW_ELEMENTS    OW                          // 8
#define OUT_ROW_BYTES       (OUT_ROW_ELEMENTS * FP16_B) // 16B
#define OUT_ELEMENTS        (OH * OW)                   // 48
#define OUT_BYTES           (OUT_ELEMENTS * FP16_B)     // 96B

#define ACTIVE_TID_MASK     0xFFFFu

// L1 temp areas per SPU: OW=8 accumulator slots x 32B = 256B
#define ACC_BASE            (BANK_C + 0x000)
#define COMPACT_BASE        (BANK_C + 0x400)

int main(void)
{
    uint8_t nest_id = NEST_ID;

    __split();

    __start_plan(nest_id);

        __start_shared();
            // Load input (48B) DDR -> L2
            __load(
                GTX_MAIN_ADDR(BASE_DDR_INPUT), BASE_L2_INPUT,
                IN_BYTES, (uint16_t)IN_BYTES, 1, (uint16_t)IN_BYTES
            );
            // Load kernel (18B) DDR -> L2 with credit
            __load_cr(
                GTX_MAIN_ADDR(BASE_DDR_KERNEL), BASE_L2_KERNEL,
                KERNEL_BYTES, (uint16_t)KERNEL_BYTES, 1, (uint16_t)KERNEL_BYTES,
                1, ACTIVE_TID_MASK, 0xBEEF
            );

            // Wait for all SPUs to complete
            __credit_chk(ACTIVE_TID_MASK);

            // Store result (96B) L2 -> DDR with credit from active SPUs
            __store_cr(
                BASE_L2_RESULT, GTX_MAIN_ADDR(BASE_DDR_RESULT),
                OUT_BYTES, (uint16_t)OUT_BYTES, 1, (uint16_t)OUT_BYTES,
                1, ACTIVE_TID_MASK
            );
        __end_shared();

        //=====================================================
        // Threads: Each SPU processes one output row
        // SPU tid handles output row oh = tid (if tid < OH)
        //=====================================================
        for (uint8_t tid = 0; tid < SPU_NUM_PER_NEST; tid++) {

            __start_thread(tid);

                if (tid < OH) {
                    const uint16_t tid_mask = (uint16_t)(0x1u << tid);

                    __credit_chk(0xBEEF);   // wait for DDR->L2 load
                    __set_spm_addr(BANK_R, BANK_C, BANK_B, BANK_A);

                    uint16_t my_oh = tid;   // this SPU's output row

                    // L2 -> L1: load input (48B) and kernel (18B)
                    __load(
                        BASE_L2_INPUT, BANK_A,
                        IN_BYTES, (uint16_t)IN_BYTES, 1, (uint16_t)IN_BYTES
                    );
                    __load_cr(
                        BASE_L2_KERNEL, BANK_B,
                        KERNEL_BYTES, (uint16_t)KERNEL_BYTES, 1, (uint16_t)KERNEL_BYTES,
                        1, tid_mask, nest_id
                    );

                    // Zero accumulator slots for this output row (OW=8 slots x 32B)
                    __fill(ACC_BASE, OW * SVR_BYTES, OW * SVR_BYTES, 1, 0, 0);

                    uint32_t in_temp  = BANK_R + 0x100;
                    uint32_t ker_temp = BANK_R + 0x200;

                    // Scatter-accumulate: only process contributions to my_oh
                    // oh_out = ih*S + kh → ih = (my_oh - kh)/S
                    // For stride S=1: ih = my_oh - kh (valid when 0 <= ih < IH)
                    for (uint16_t kh = 0; kh < KH; kh++) {
                        int16_t ih = (int16_t)my_oh - (int16_t)kh;
                        if (ih < 0 || ih >= IH) continue;

                        for (uint16_t iw = 0; iw < IW; iw++) {
                            // Load input[ih][iw]
                            uint32_t in_off = ((uint32_t)ih * IW + (uint32_t)iw) * FP16_B;
                            __copy(BANK_A + in_off, in_temp, 0, FP16_B, 1, 0);

                            for (uint16_t kw = 0; kw < KW; kw++) {
                                uint16_t ow_out = iw * S + kw;

                                // Load kernel[kh][kw]
                                uint32_t k_off = ((uint32_t)kh * KW + (uint32_t)kw) * FP16_B;
                                __copy(BANK_B + k_off, ker_temp, 0, FP16_B, 1, 0);

                                // Multiply: input[ih][iw] * kernel[kh][kw]
                                __set_spm_addr(BANK_R, BANK_C, ker_temp, in_temp);
                                __dot_product(1, 0);

                                // Accumulate into slot[ow_out]
                                uint32_t slot_addr = ACC_BASE + (uint32_t)ow_out * SVR_BYTES;
                                __load_svr(slot_addr, 1);
                                __add_ii(0, 1, 2);
                                __store_svr(slot_addr, 2);
                            }
                        }
                    }

                    __set_spm_addr(BANK_R, BANK_C, BANK_B, BANK_A);

                    // Compact OW=6 accumulator slots into contiguous FP16 output
                    for (uint16_t i = 0; i < OW; i++) {
                        __copy(
                            ACC_BASE + (uint32_t)i * SVR_BYTES,
                            COMPACT_BASE + (uint32_t)i * FP16_B,
                            0, FP16_B, 1, 0
                        );
                    }

                    // L1 -> L2: store this row's output at correct offset
                    __store_cr(
                        COMPACT_BASE,
                        BASE_L2_RESULT + (uint32_t)my_oh * OUT_ROW_BYTES,
                        OUT_ROW_BYTES, (uint16_t)OUT_ROW_BYTES,
                        1, (uint16_t)OUT_ROW_BYTES,
                        1, tid_mask
                    );
                }

            __end_thread(tid);
        }

    __end_plan(nest_id);

    __join();

    return 0;
}
