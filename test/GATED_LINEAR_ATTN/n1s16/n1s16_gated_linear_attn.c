//==================================================================
// GGML_OP_GATED_LINEAR_ATTN — Gated Linear Attention (n1s16)
//
// Fixed generator binding used by this kernel:
//   shape [S,T] = [4,5], H = 1, n_seqs = 1
//
// Implementation path: composed GTX path.  The recurrence is ordered on
// SPU0, but the state update, q/state dot product, and scale operations are
// expressed with GTX vector/dot intrinsics over the S-wide head vector.
//==================================================================

#ifndef N1S16_GATED_LINEAR_ATTN_C
#define N1S16_GATED_LINEAR_ATTN_C

#include <stdint.h>

#include "intrin.h"
#include "gtx/address.h"
#include "gtx_csr.h"

#define NEST_ID             0
#define SPU_NUM_PER_NEST    16
#define ACTIVE_SPU_MASK     0x0001
#define NOT_USE             0xBEEF

#define S                   4
#define H                   1
#define T                   5
#define N_SEQS              1
#define C                   (S * H)
#define FP16_B              2
#define F32_B               4

#define FP16_SCALE          0x3F000000  // 1 / sqrt(4)

#define VEC_ELEMS           S
#define VEC_BYTES           (VEC_ELEMS * FP16_B)
#define INPUT_ELEMS         (C * T)
#define INPUT_BYTES         (INPUT_ELEMS * FP16_B)
#define STATE_ELEMS         (S * S * H)
#define STATE_BYTES         (STATE_ELEMS * FP16_B)
#define HALF_RESULT_ELEMS   (INPUT_ELEMS + STATE_ELEMS)
#define HALF_RESULT_BYTES   (HALF_RESULT_ELEMS * FP16_B)
#define RESULT_BYTES        (HALF_RESULT_ELEMS * F32_B)

#define BASE_DDR_K          0x1000000
#define BASE_DDR_V          0x2000000
#define BASE_DDR_Q          0x3000000
#define BASE_DDR_G          0x4000000
#define BASE_DDR_STATE      0x5000000
#define BASE_DDR_RESULT     0xf000000

#define L2_K                0x000000
#define L2_V                0x000100
#define L2_Q                0x000200
#define L2_G                0x000300
#define L2_STATE            0x000400
#define L2_RESULT           0x000500

#define BANK_A              0x00000
#define BANK_B              0x20000
#define BANK_C              0x30000
#define BANK_R              0x50000

#define K_VEC               (BANK_A + 0x0000)
#define V_VEC               (BANK_A + 0x0020)
#define Q_VEC               (BANK_A + 0x0040)
#define G_VEC               (BANK_A + 0x0060)
#define STATE_MAT           (BANK_A + 0x0080)
#define STATE_VEC           (BANK_A + 0x00A0)
#define TMP1_VEC            (BANK_A + 0x00C0)
#define TMP2_VEC            (BANK_A + 0x00E0)
#define Q_SCALE_VEC         (BANK_A + 0x0100)
#define HALF_RESULT         (BANK_A + 0x0120)
#define F32_RESULT          (BANK_R + 0x0000)
#define SVR_TMP             0
#define SVR_PACK            1
#define SVR_ADDR            0x800
#define SVR_WORD_ADDR(svr, word) (SVR_ADDR + ((svr) * 4) + (word))

int main(void) {
    __split();
    {
        __start_plan(NEST_ID);

            __start_shared();
                __fill(L2_RESULT,
                    (uint32_t)RESULT_BYTES, (uint16_t)RESULT_BYTES, 1, 0, 0);

                __load(GTX_MAIN_ADDR(BASE_DDR_K), L2_K,
                    INPUT_BYTES, (uint16_t)INPUT_BYTES, 1, (uint16_t)INPUT_BYTES);
                __load(GTX_MAIN_ADDR(BASE_DDR_V), L2_V,
                    INPUT_BYTES, (uint16_t)INPUT_BYTES, 1, (uint16_t)INPUT_BYTES);
                __load(GTX_MAIN_ADDR(BASE_DDR_Q), L2_Q,
                    INPUT_BYTES, (uint16_t)INPUT_BYTES, 1, (uint16_t)INPUT_BYTES);
                __load(GTX_MAIN_ADDR(BASE_DDR_G), L2_G,
                    INPUT_BYTES, (uint16_t)INPUT_BYTES, 1, (uint16_t)INPUT_BYTES);
                __load_cr(GTX_MAIN_ADDR(BASE_DDR_STATE), L2_STATE,
                    STATE_BYTES, (uint16_t)STATE_BYTES, 1, (uint16_t)STATE_BYTES,
                    1, ACTIVE_SPU_MASK, NOT_USE);

                __credit_chk(ACTIVE_SPU_MASK);

                __store_cr(L2_RESULT, GTX_MAIN_ADDR(BASE_DDR_RESULT),
                    RESULT_BYTES, (uint16_t)RESULT_BYTES, 1, (uint16_t)RESULT_BYTES,
                    1, ACTIVE_SPU_MASK);
            __end_shared();

            for (uint8_t tid = 0; tid < SPU_NUM_PER_NEST; ++tid) {
                __start_thread(tid);
                    if (tid == 0) {
                        __set_spm_addr(BANK_R, BANK_C, BANK_B, BANK_A);
                        __credit_chk(NOT_USE);

                        __load(L2_STATE, STATE_MAT,
                            STATE_BYTES, (uint16_t)STATE_BYTES, 1, (uint16_t)STATE_BYTES);
                        __credit_ld(ACTIVE_SPU_MASK, NEST_ID);

                        for (uint8_t t = 0; t < T; ++t) {
                            const uint32_t token_off = (uint32_t)t * C * FP16_B;

                            __load(L2_K + token_off, K_VEC,
                                VEC_BYTES, (uint16_t)VEC_BYTES, 1, (uint16_t)VEC_BYTES);
                            __load(L2_Q + token_off, Q_VEC,
                                VEC_BYTES, (uint16_t)VEC_BYTES, 1, (uint16_t)VEC_BYTES);
                            __load(L2_G + token_off, G_VEC,
                                VEC_BYTES, (uint16_t)VEC_BYTES, 1, (uint16_t)VEC_BYTES);

                            __set_spm_addr(Q_SCALE_VEC, 0, BANK_B, Q_VEC);
                            __mul_vs(VEC_ELEMS, FP16_SCALE, 0);

                            for (uint8_t j = 0; j < S; ++j) {
                                // Gather state[:, j] and broadcast v[t, j].
                                for (uint8_t i = 0; i < S; ++i) {
                                    __copy(STATE_MAT + ((uint32_t)i * S + j) * FP16_B,
                                        STATE_VEC + (uint32_t)i * FP16_B,
                                        0, FP16_B, 1, 0);
                                }
                                __load(L2_V + token_off + (uint32_t)j * FP16_B, V_VEC,
                                    0, FP16_B, S, FP16_B);

                                // state = state * g + k * v_j
                                __set_spm_addr(TMP1_VEC, 0, G_VEC, STATE_VEC);
                                __mul_vv(VEC_ELEMS);

                                __set_spm_addr(TMP2_VEC, 0, V_VEC, K_VEC);
                                __mul_vv(VEC_ELEMS);

                                __set_spm_addr(STATE_VEC, 0, TMP2_VEC, TMP1_VEC);
                                __add_vv(VEC_ELEMS);

                                for (uint8_t i = 0; i < S; ++i) {
                                    __copy(STATE_VEC + (uint32_t)i * FP16_B,
                                        STATE_MAT + ((uint32_t)i * S + j) * FP16_B,
                                        0, FP16_B, 1, 0);
                                }

                                // y[t, j] = dot(q * scale, state[:, j])
                                __set_spm_addr(BANK_R, 0, STATE_VEC, Q_SCALE_VEC);
                                __dot_product(VEC_ELEMS, SVR_TMP);
                                __store_svr(TMP1_VEC, SVR_TMP);
                                __copy(TMP1_VEC, HALF_RESULT + token_off + (uint32_t)j * FP16_B,
                                    0, FP16_B, 1, 0);
                            }
                        }

                        // Materialize final state in ggml output order after y.
                        for (uint8_t i = 0; i < S; ++i) {
                            for (uint8_t j = 0; j < S; ++j) {
                                __copy(STATE_MAT + ((uint32_t)i * S + j) * FP16_B,
                                    HALF_RESULT + INPUT_BYTES + ((uint32_t)i * S + j) * FP16_B,
                                    0, FP16_B, 1, 0);
                            }
                        }

                        // Materialize the FP16 ggml result prefix.  Shared mode
                        // zero-initialized the full 32-byte L2_RESULT slot before
                        // this 16-byte prefix store, so the trailing 16 bytes are
                        // explicitly initialized before the full DDR store.
                        __store_cr(HALF_RESULT, L2_RESULT,
                            HALF_RESULT_BYTES, (uint16_t)HALF_RESULT_BYTES, 1, (uint16_t)HALF_RESULT_BYTES,
                            1, ACTIVE_SPU_MASK);
                    }
                __end_thread(tid);
            }

        __end_plan(NEST_ID);
    }
    __join();

    return 0;
}

#endif
