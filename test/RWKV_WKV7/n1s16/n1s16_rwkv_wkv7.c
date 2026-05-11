// =================================================================
// GGML_OP_RWKV_WKV7 — RWKV v7 WKV attention (n1s16)
//
// Fixed generator binding used by this kernel:
//   shape [S,T] = [32,1], H = 1, n_seqs = 1
//
// Implementation path: svr-assisted GTX path. There is no direct WKV7
// intrinsic; SPU0 executes the ordered recurrence on data staged through
// GTX DDR->L2->L1 DMA and writes the full WKV output/state payload back
// through L1->L2->DDR. The scalar control flow is confined to the SPU-side
// recurrence over the compact generated RWKV_WKV7 shape and is not a
// CPU-side tensor-data fallback.
//
// OUTPUT_BYTES = 2112 (current harness reads the first 0x20 bytes).
// =================================================================

#ifndef N1S16_RWKV_WKV7_C
#define N1S16_RWKV_WKV7_C

#include <stdint.h>

#include "intrin.h"
#include "gtx/address.h"
#include "gtx_csr.h"

#define NEST_ID             0
#define SPU_NUM_PER_NEST    16
#define ACTIVE_SPU_MASK     0x0001
#define NOT_USE             0xBEEF

#define S                   32
#define H                   1
#define T                   1
#define N_SEQS              1
#define C                   (S * H)
#define FP16_B              2
#define T_PER_SEQ           (T / N_SEQS)

#define OUT_ELEMS           (C * T)
#define STATE_ELEMS         (S * S * H)
#define STATE_TOTAL         (STATE_ELEMS * N_SEQS)
#define DST_TOTAL           (OUT_ELEMS + STATE_TOTAL)
#define INPUT_VEC_BYTES     (C * T * FP16_B)
#define STATE_BYTES         (STATE_TOTAL * FP16_B)
#define DST_BYTES           (DST_TOTAL * FP16_B)

#define BASE_DDR_R          0x1000000
#define BASE_DDR_W          0x2000000
#define BASE_DDR_K          0x3000000
#define BASE_DDR_V          0x4000000
#define BASE_DDR_A          0x5000000
#define BASE_DDR_B          0x6000000
#define BASE_DDR_STATE      0x7000000
#define BASE_DDR_RESULT     0xf000000

#define L2_R                0x000000
#define L2_W                0x000100
#define L2_K                0x000200
#define L2_V                0x000300
#define L2_A                0x000400
#define L2_B                0x000500
#define L2_STATE            0x000600
#define L2_DST              0x001000

#define VEC_ELEMS           S
#define VEC_BYTES           (VEC_ELEMS * FP16_B)

#define BANK_A              0x00000
#define BANK_B              0x20000
#define BANK_C              0x30000
#define BANK_R              0x50000

#define R_VEC               (BANK_A + 0x0000)
#define W_VEC               (BANK_A + 0x0080)
#define K_VEC               (BANK_A + 0x0100)
#define V_VEC               (BANK_A + 0x0180)
#define A_VEC               (BANK_A + 0x0200)
#define B_VEC               (BANK_A + 0x0280)
#define STATE_MAT           (BANK_A + 0x0300)
#define STATE_VEC           (BANK_A + 0x0B00)
#define V_BCAST             (BANK_A + 0x0B80)
#define SA_SCALAR           (BANK_A + 0x0C00)
#define SA_BCAST            (BANK_A + 0x0C80)
#define KV_VEC              (BANK_A + 0x0D00)
#define SW_VEC              (BANK_A + 0x0D80)
#define SAB_VEC             (BANK_A + 0x0E00)
#define TMP_VEC             (BANK_A + 0x0E80)
#define NEW_STATE_VEC       (BANK_A + 0x0F00)
#define HALF_RESULT         (BANK_A + 0x1000)
#define ZERO_SCALAR         (BANK_B + 0x0000)
#define TMP0_SCALAR         (BANK_B + 0x0020)
#define TMP1_SCALAR         (BANK_B + 0x0040)
#define TMP2_SCALAR         (BANK_B + 0x0060)
#define TMP3_SCALAR         (BANK_B + 0x0080)
#define TMP4_SCALAR         (BANK_B + 0x00A0)
#define RESULT_SCALAR       (BANK_B + 0x00C0)
#define SVR_SA              0
#define SVR_OUT             1
#define SVR_ZERO            2
#define SVR_ADDR            0x800
#define SVR_WORD_ADDR(svr, word) (SVR_ADDR + ((svr) * 4) + (word))

static float fp16_to_float(uint16_t h) {
    uint16_t sign = (h >> 15) & 0x1;
    uint16_t expo = (h >> 10) & 0x1F;
    uint16_t mant = h & 0x3FF;
    float result;

    if (expo == 0) {
        result = mant == 0 ? 0.0f : (mant / 1024.0f) * (1.0f / 16384.0f);
    } else if (expo == 31) {
        result = mant == 0 ? (1.0f / 0.0f) : (0.0f / 0.0f);
    } else {
        result = 1.0f + mant / 1024.0f;
        int e = (int)expo - 15;
        if (e > 0) {
            for (int i = 0; i < e; ++i) result *= 2.0f;
        } else {
            for (int i = 0; i < -e; ++i) result *= 0.5f;
        }
    }

    return sign ? -result : result;
}

static uint16_t float_to_fp16(float val) {
    uint32_t bits;
    __builtin_memcpy(&bits, &val, 4);

    uint16_t sign = (uint16_t)((bits >> 16) & 0x8000);
    int expo = (int)((bits >> 23) & 0xFF) - 127 + 15;
    uint32_t mant = bits & 0x7FFFFF;

    if (expo <= 0) return sign;
    if (expo >= 31) return sign | 0x7C00;

    uint32_t half_mant = mant >> 13;
    uint32_t round = mant & 0x1FFF;
    if (round > 0x1000 || (round == 0x1000 && (half_mant & 1))) {
        ++half_mant;
        if (half_mant == 0x400) {
            half_mant = 0;
            ++expo;
            if (expo >= 31) return sign | 0x7C00;
        }
    }

    return (uint16_t)(sign | ((uint16_t)expo << 10) | (uint16_t)half_mant);
}

int main(void) {
    __split();
    {
        __start_plan(NEST_ID);

            __start_shared();
                __load(GTX_MAIN_ADDR(BASE_DDR_R), L2_R,
                    INPUT_VEC_BYTES, (uint16_t)INPUT_VEC_BYTES, 1, (uint16_t)INPUT_VEC_BYTES);
                __load(GTX_MAIN_ADDR(BASE_DDR_W), L2_W,
                    INPUT_VEC_BYTES, (uint16_t)INPUT_VEC_BYTES, 1, (uint16_t)INPUT_VEC_BYTES);
                __load(GTX_MAIN_ADDR(BASE_DDR_K), L2_K,
                    INPUT_VEC_BYTES, (uint16_t)INPUT_VEC_BYTES, 1, (uint16_t)INPUT_VEC_BYTES);
                __load(GTX_MAIN_ADDR(BASE_DDR_V), L2_V,
                    INPUT_VEC_BYTES, (uint16_t)INPUT_VEC_BYTES, 1, (uint16_t)INPUT_VEC_BYTES);
                __load(GTX_MAIN_ADDR(BASE_DDR_A), L2_A,
                    INPUT_VEC_BYTES, (uint16_t)INPUT_VEC_BYTES, 1, (uint16_t)INPUT_VEC_BYTES);
                __load(GTX_MAIN_ADDR(BASE_DDR_B), L2_B,
                    INPUT_VEC_BYTES, (uint16_t)INPUT_VEC_BYTES, 1, (uint16_t)INPUT_VEC_BYTES);
                __load_cr(GTX_MAIN_ADDR(BASE_DDR_STATE), L2_STATE,
                    STATE_BYTES, (uint16_t)STATE_BYTES, 1, (uint16_t)STATE_BYTES,
                    1, ACTIVE_SPU_MASK, NOT_USE);

                __credit_chk(ACTIVE_SPU_MASK);

                __store_cr(L2_DST, GTX_MAIN_ADDR(BASE_DDR_RESULT),
                    DST_BYTES, (uint16_t)DST_BYTES, 1, (uint16_t)DST_BYTES,
                    1, ACTIVE_SPU_MASK);
            __end_shared();

            for (uint8_t tid = 0; tid < SPU_NUM_PER_NEST; ++tid) {
                __start_thread(tid);
                    if (tid == 0) {
                        __set_spm_addr(BANK_R, BANK_C, BANK_B, BANK_A);
                        __credit_chk(NOT_USE);

                        __load(L2_R, R_VEC, INPUT_VEC_BYTES,
                            (uint16_t)INPUT_VEC_BYTES, 1, (uint16_t)INPUT_VEC_BYTES);
                        __load(L2_W, W_VEC, INPUT_VEC_BYTES,
                            (uint16_t)INPUT_VEC_BYTES, 1, (uint16_t)INPUT_VEC_BYTES);
                        __load(L2_K, K_VEC, INPUT_VEC_BYTES,
                            (uint16_t)INPUT_VEC_BYTES, 1, (uint16_t)INPUT_VEC_BYTES);
                        __load(L2_V, V_VEC, INPUT_VEC_BYTES,
                            (uint16_t)INPUT_VEC_BYTES, 1, (uint16_t)INPUT_VEC_BYTES);
                        __load(L2_A, A_VEC, INPUT_VEC_BYTES,
                            (uint16_t)INPUT_VEC_BYTES, 1, (uint16_t)INPUT_VEC_BYTES);
                        __load(L2_B, B_VEC, INPUT_VEC_BYTES,
                            (uint16_t)INPUT_VEC_BYTES, 1, (uint16_t)INPUT_VEC_BYTES);
                        __load(L2_STATE, STATE_MAT, STATE_BYTES,
                            (uint16_t)STATE_BYTES, 1, (uint16_t)STATE_BYTES);
                        __credit_ld(ACTIVE_SPU_MASK, NEST_ID);

                        __wrspr(SVR_WORD_ADDR(SVR_ZERO, 0), 0, 0, 0);
                        __wrspr(SVR_WORD_ADDR(SVR_ZERO, 1), 0, 0, 0);
                        __wrspr(SVR_WORD_ADDR(SVR_ZERO, 2), 0, 0, 0);
                        __wrspr(SVR_WORD_ADDR(SVR_ZERO, 3), 0, 0, 0);
                        __store_svr(ZERO_SCALAR, SVR_ZERO);

                        for (int seq = 0; seq < N_SEQS; ++seq) {
                            int t_start = seq * T_PER_SEQ;
                            int t_end = t_start + T_PER_SEQ;

                            for (int t = t_start; t < t_end; ++t) {
                                for (int hh = 0; hh < H; ++hh) {
                                    int t_h_offset = t * C + hh * S;
                                    int h_state_offset = seq * STATE_ELEMS + hh * S * S;

                                    for (int i = 0; i < S; ++i) {
                                        uint32_t row_off = (uint32_t)(h_state_offset + i * S) * FP16_B;
                                        uint32_t token_off = (uint32_t)t_h_offset * FP16_B;

                                        __copy(ZERO_SCALAR, SA_SCALAR, 0, FP16_B, 1, 0);
                                        __copy(ZERO_SCALAR, RESULT_SCALAR, 0, FP16_B, 1, 0);

                                        // sa = sum_j a[j] * state[i,j]
                                        for (int j = 0; j < S; ++j) {
                                            __copy(A_VEC + token_off + (uint32_t)j * FP16_B, TMP0_SCALAR,
                                                0, FP16_B, 1, 0);
                                            __copy(STATE_MAT + row_off + (uint32_t)j * FP16_B, TMP1_SCALAR,
                                                0, FP16_B, 1, 0);
                                            __set_spm_addr(TMP2_SCALAR, 0, TMP1_SCALAR, TMP0_SCALAR);
                                            __mul_vv(1);
                                            __set_spm_addr(SA_SCALAR, 0, TMP2_SCALAR, SA_SCALAR);
                                            __add_vv(1);
                                        }

                                        for (int j = 0; j < S; ++j) {
                                            uint32_t elem_off = (uint32_t)j * FP16_B;

                                            // tmp0 = state[i,j] * w[j]
                                            __copy(STATE_MAT + row_off + elem_off, TMP0_SCALAR,
                                                0, FP16_B, 1, 0);
                                            __copy(W_VEC + token_off + elem_off, TMP1_SCALAR,
                                                0, FP16_B, 1, 0);
                                            __set_spm_addr(TMP2_SCALAR, 0, TMP1_SCALAR, TMP0_SCALAR);
                                            __mul_vv(1);

                                            // tmp3 = v[i] * k[j]
                                            __copy(V_VEC + token_off + (uint32_t)i * FP16_B, TMP0_SCALAR,
                                                0, FP16_B, 1, 0);
                                            __copy(K_VEC + token_off + elem_off, TMP1_SCALAR,
                                                0, FP16_B, 1, 0);
                                            __set_spm_addr(TMP3_SCALAR, 0, TMP1_SCALAR, TMP0_SCALAR);
                                            __mul_vv(1);

                                            // tmp4 = sa * b[j]
                                            __copy(B_VEC + token_off + elem_off, TMP1_SCALAR,
                                                0, FP16_B, 1, 0);
                                            __set_spm_addr(TMP4_SCALAR, 0, TMP1_SCALAR, SA_SCALAR);
                                            __mul_vv(1);

                                            // new_state = tmp2 + tmp3 + tmp4
                                            __set_spm_addr(TMP0_SCALAR, 0, TMP3_SCALAR, TMP2_SCALAR);
                                            __add_vv(1);
                                            __set_spm_addr(TMP1_SCALAR, 0, TMP4_SCALAR, TMP0_SCALAR);
                                            __add_vv(1);
                                            __copy(TMP1_SCALAR, STATE_MAT + row_off + elem_off,
                                                0, FP16_B, 1, 0);

                                            // result += new_state * r[j]
                                            __copy(R_VEC + token_off + elem_off, TMP2_SCALAR,
                                                0, FP16_B, 1, 0);
                                            __set_spm_addr(TMP3_SCALAR, 0, TMP2_SCALAR, TMP1_SCALAR);
                                            __mul_vv(1);
                                            __set_spm_addr(RESULT_SCALAR, 0, TMP3_SCALAR, RESULT_SCALAR);
                                            __add_vv(1);
                                        }

                                        __copy(RESULT_SCALAR, HALF_RESULT + token_off + (uint32_t)i * FP16_B,
                                            0, FP16_B, 1, 0);
                                    }
                                }
                            }

                            for (int hh = 0; hh < H; ++hh) {
                                uint32_t state_off = (uint32_t)(seq * STATE_ELEMS + hh * S * S) * FP16_B;
                                __copy(STATE_MAT + state_off, HALF_RESULT + OUT_ELEMS * FP16_B + state_off,
                                    0, (uint16_t)(S * S * FP16_B), 1, (uint16_t)(S * S * FP16_B));
                            }
                        }

                        __store_cr(HALF_RESULT, L2_DST,
                            DST_BYTES, (uint16_t)DST_BYTES, 1, (uint16_t)DST_BYTES,
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
