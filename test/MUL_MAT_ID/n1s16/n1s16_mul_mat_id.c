// =================================================================
// GGML_OP_MUL_MAT_ID — Indirect matrix multiply for MoE (n1s16)
// Implements generated-data MUL_MAT_ID shape [16,64]:
//   as  [K=16, rows=32, experts=4]     at BASE_DDR_AS    (FP16)
//   b   [K=16, used=2, tokens=64]      at BASE_DDR_B     (FP16)
//   ids [used=2, tokens=64]            at BASE_DDR_IDS   (I32)
//   dst [rows=32, used=2, tokens=64]   at BASE_DDR_RESULT(FP16)
//   dst[:, e, t] = as[:, :, ids[e,t]] @ b[:, e, t]
//
// Strategy:
//   1. RISC-V control plane reads the small I32 ids[] metadata and uses
//      GTX copy.mem to gather selected expert matrices into TEMP DDR.
//   2. GTX plan: 16 SPUs compute the 128 (used, token) matrix-vector
//      products, eight work items per SPU, using direct __mm.
// =================================================================

#include "intrin.h"
#include "gtx/address.h"
#include "gtx_csr.h"

// Hardware
#define NEST_NUM            1
#define SPU_NUM_PER_NEST    16
#define WORK_PER_NEST       (WORK_ITEMS / NEST_NUM)         // 32
#define WORK_PER_SPU        (WORK_PER_NEST / SPU_NUM_PER_NEST) // 2

// DDR addresses
#define BASE_DDR_AS         0x1000000   // experts (4 x 32 x 16 FP16 = 4096B)
#define BASE_DDR_B          0x2000000   // input tokens (16 x 2 x 64 FP16 = 4096B)
#define BASE_DDR_IDS        0x3000000   // indices (2 x 64 I32 = 512B)
#define BASE_DDR_TEMP       0x4000000   // gathered experts (128 x 1024B = 131072B)
#define BASE_DDR_RESULT     0xf000000

// L2 SPM addresses
#define L2_A                0x000000    // 128 selected experts × 1024B = 131072B
#define L2_B                0x021000
#define L2_RESULT           0x023000

// L1 SPM bank addresses
#define BANK_A              0x00000
#define BANK_B              0x20000
#define BANK_C              0x30000
#define BANK_R              0x50000

// Dimensions
#define N_EXPERTS           4
#define K_DIM               16
#define ROW_DIM             16
#define N_EXPERT_USED       2
#define N_TOKENS            32
#define WORK_ITEMS          (N_EXPERT_USED * N_TOKENS)
#define FP16_B              2

#define EXPERT_BYTES        (ROW_DIM * K_DIM * FP16_B)  // 1024B per expert
#define B_VEC_BYTES         (K_DIM * FP16_B)            // 32B per b vector
#define OUT_VEC_BYTES       (ROW_DIM * FP16_B)          // 64B per output vector

// Per-NEST sizes
#define A_BYTES_PER_NEST    (WORK_PER_NEST * EXPERT_BYTES) // 131072B
#define B_BYTES_PER_NEST    (WORK_PER_NEST * B_VEC_BYTES)  // 4096B
#define OUT_BYTES_PER_NEST  (WORK_PER_NEST * OUT_VEC_BYTES)// 8192B

// Stack

int main(void)
{
    // ---------------------------------------------------------------
    // Step 1: CPU reads ids, gathers expert matrices to TEMP DDR
    // ---------------------------------------------------------------
    volatile int32_t *ids = (volatile int32_t *)GTX_MAIN_ADDR(BASE_DDR_IDS);

    for (uint16_t work = 0; work < WORK_ITEMS; work++) {
        int32_t eidx = ids[work];
        if (eidx < 0) {
            eidx = 0;
        }
        if (eidx >= N_EXPERTS) {
            eidx = N_EXPERTS - 1;
        }
        uint32_t src_off = (uint32_t)eidx * EXPERT_BYTES;
        uint32_t dst_off = (uint32_t)work * EXPERT_BYTES;
        __copy_mem(
            GTX_MAIN_ADDR(BASE_DDR_AS) + src_off,
            GTX_MAIN_ADDR(BASE_DDR_TEMP) + dst_off,
            EXPERT_BYTES, EXPERT_BYTES, 1,
            (uint16_t)(EXPERT_BYTES & 0xFFFF),
            (uint16_t)(EXPERT_BYTES >> 16)
        );
    }

    // ---------------------------------------------------------------
    // Step 2: GTX plan — each SPU computes 1 output row
    // ---------------------------------------------------------------

    __split();

    {
        uint8_t nest_id = 0;

        __start_plan(nest_id);

            __start_shared();
                // Load gathered expert matrices for this NEST.  Use a 2-D DMA
                // shape so the 16-bit length field never wraps for [16,64].
                uint64_t temp_nest_ddr = GTX_MAIN_ADDR(BASE_DDR_TEMP) + (uint32_t)nest_id * A_BYTES_PER_NEST;
                __load(
                    temp_nest_ddr, L2_A,
                    EXPERT_BYTES, (uint16_t)EXPERT_BYTES,
                    WORK_PER_NEST, (uint16_t)EXPERT_BYTES
                );

                // Load B vectors for this NEST (128 × 32B = 4096B)
                uint64_t b_nest_ddr = GTX_MAIN_ADDR(BASE_DDR_B) + (uint32_t)nest_id * B_BYTES_PER_NEST;
                __load_cr(
                    b_nest_ddr, L2_B,
                    B_BYTES_PER_NEST, (uint16_t)B_BYTES_PER_NEST,
                    1, (uint16_t)B_BYTES_PER_NEST,
                    1, 0xFFFF, 0xBEEF
                );

                // Wait for all SPUs, store result
                __credit_chk(0xFFFF);
                uint64_t dst_nest_ddr = GTX_MAIN_ADDR(BASE_DDR_RESULT) + (uint32_t)nest_id * OUT_BYTES_PER_NEST;
                __store_cr(
                    L2_RESULT, dst_nest_ddr,
                    OUT_BYTES_PER_NEST, (uint16_t)OUT_BYTES_PER_NEST,
                    1, (uint16_t)OUT_BYTES_PER_NEST,
                    1, 0xFFFF
                );
            __end_shared();

            for (uint8_t tid = 0; tid < SPU_NUM_PER_NEST; tid++) {

                __start_thread(tid);
                    __set_spm_addr(BANK_R, BANK_C, BANK_B, BANK_A);
                    __credit_chk(0xBEEF);
                    for (uint8_t r = 0; r < WORK_PER_SPU; r++) {
                        uint32_t work_idx = (uint32_t)(tid * WORK_PER_SPU + r);

                        // Load selected expert matrix [ROW_DIM×K = EXPERT_BYTES] → Bank A
                        __load(
                            L2_A + work_idx * EXPERT_BYTES,
                            BANK_A,
                            EXPERT_BYTES, (uint16_t)EXPERT_BYTES,
                            1, (uint16_t)EXPERT_BYTES
                        );

                        // Load b[e,t] vector [1×K = B_VEC_BYTES] → Bank B.
                        // The last load consumes the shared load credit for this SPU.
                        if (r == WORK_PER_SPU - 1) {
                            __load_cr(
                                L2_B + work_idx * B_VEC_BYTES,
                                BANK_B,
                                B_VEC_BYTES, (uint16_t)B_VEC_BYTES,
                                1, (uint16_t)B_VEC_BYTES,
                                1, (uint64_t)(0x1u << tid), nest_id
                            );
                        } else {
                            __load(
                                L2_B + work_idx * B_VEC_BYTES,
                                BANK_B,
                                B_VEC_BYTES, (uint16_t)B_VEC_BYTES,
                                1, (uint16_t)B_VEC_BYTES
                            );
                        }

                        // Matrix multiply: A[ROW_DIM×K] × B[1×K]^T → R[ROW_DIM×1] FP16
                        __mm(ROW_DIM, K_DIM, 1);

                        // Store result [ROW_DIM×1 = 16B] from BANK_R → L2
                        if (r == WORK_PER_SPU - 1) {
                            __store_cr(BANK_R, L2_RESULT + work_idx * OUT_VEC_BYTES, OUT_VEC_BYTES, (uint16_t)OUT_VEC_BYTES, 1, (uint16_t)OUT_VEC_BYTES, 1, (uint64_t)(0x1u << tid));
                        } else {
                            __store(BANK_R, L2_RESULT + work_idx * OUT_VEC_BYTES, OUT_VEC_BYTES, (uint16_t)OUT_VEC_BYTES, 1, (uint16_t)OUT_VEC_BYTES);
                        }
                    }
                __end_thread(tid);
            }

        __end_plan(nest_id);
    }

    __join();

    return 0;
}
