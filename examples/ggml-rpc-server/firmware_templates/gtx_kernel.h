//==================================================================
// gtx_kernel.h — CUDA-style launch abstraction for GTX NPU kernels
//
// Usage in a .c.tpl:
//
//   #include "gtx_kernel.h"
//   GTX_KERNEL_BODY(SHARED_BODY, THREAD_BODY)
//   int main(void) {
//     GTX_LAUNCH(NESTS, SPUS_PER_NEST);
//     return 0;
//   }
//
// Where:
//   - NESTS = grid dim (CUDA blockIdx range). Hardware supports 1 NEST;
//     N > 1 is virtualised as a sequential nest_id loop.
//   - SPUS_PER_NEST = block dim (CUDA threadIdx range). Hardware caps at 16;
//     S < 16 limits the active-tid mask, S > 16 is rejected at compile time.
//   - SHARED_BODY  = code that runs once per NEST inside __start_shared() —
//     typically the DDR↔L2 DMA pair plus the matching __credit_chk.
//   - THREAD_BODY  = code that runs per SPU. The variables `tid` (uint8_t)
//     and `tid_mask` (uint16_t) are in scope.
//
// Mapping to CUDA semantics:
//   nest_id  ↔ blockIdx.x
//   tid      ↔ threadIdx.x
//   NESTS    ↔ gridDim.x
//   SPUS     ↔ blockDim.x
//
// HARDWARE LIMITS (per Supergate GTX NPU n1s16):
//   - 1 NEST physically; NESTS > 1 → host-driven nest_id loop (virtualised).
//   - 16 SPUs per NEST physically; SPUS in [1, 16].
//   - For SPUS < 16 the inactive lanes are masked out via active_tid_mask.
//==================================================================

#ifndef GTX_KERNEL_H
#define GTX_KERNEL_H

#include "intrin.h"
#include "gtx/address.h"
#include <stdint.h>

#ifndef GTX_SPU_HW_LIMIT
#define GTX_SPU_HW_LIMIT    16
#endif

// Helper: compile-time check that SPUS fits the hardware lane count.
// SPUS=0 is allowed and means "shared-only kernel — no per-SPU thread body".
#define GTX_STATIC_ASSERT_SPUS(SPUS) \
    _Static_assert((SPUS) >= 0 && (SPUS) <= GTX_SPU_HW_LIMIT, \
                   "SPUS_PER_NEST must be in [0, 16]")

// Mask of the first `n` lanes (n ≤ 16).
static inline uint16_t gtx_active_mask(uint8_t n) {
    return (uint16_t)((n >= GTX_SPU_HW_LIMIT) ? 0xFFFFu : ((1u << n) - 1u));
}

// Work-split helper: how many of `total` items thread `tid` owns when
// `n_threads` lanes split the work as evenly as possible (heavier lanes first).
static inline uint32_t gtx_items_for_tid(uint32_t total, uint8_t tid, uint8_t n_threads) {
    uint32_t q = total / n_threads;
    uint32_t r = total % n_threads;
    return tid < r ? q + 1u : q;
}

// First item index thread `tid` owns under the same even-split policy.
static inline uint32_t gtx_start_for_tid(uint32_t total, uint8_t tid, uint8_t n_threads) {
    uint32_t q = total / n_threads;
    uint32_t r = total % n_threads;
    return tid < r ? (uint32_t)tid * (q + 1u)
                   : r * (q + 1u) + ((uint32_t)tid - r) * q;
}

//-----------------------------------------------------------------------------
// GTX_KERNEL_BODY declares the per-launch shared & thread bodies.
// Use it once, *outside* main(), with two compound statements.
//-----------------------------------------------------------------------------
#define GTX_KERNEL_BODY(SHARED_BODY, THREAD_BODY)                              \
    static inline void __gtx_shared_body(uint8_t nest_id, uint16_t active_mask) {\
        (void)nest_id; (void)active_mask;                                      \
        SHARED_BODY                                                            \
    }                                                                          \
    static inline void __gtx_thread_body(uint8_t nest_id, uint8_t tid,         \
                                          uint16_t tid_mask, uint8_t n_threads) {\
        (void)nest_id; (void)tid; (void)tid_mask; (void)n_threads;             \
        THREAD_BODY                                                            \
    }

//-----------------------------------------------------------------------------
// GTX_LAUNCH(NESTS, SPUS) — emit the split→plan→shared→thread→join loop.
//   NESTS iterations over nest_id (virtualised when NESTS > 1).
//   SPUS active lanes per NEST.
//-----------------------------------------------------------------------------
#define GTX_LAUNCH(NESTS, SPUS)                                                \
    do {                                                                       \
        GTX_STATIC_ASSERT_SPUS(SPUS);                                          \
        const uint16_t __active_mask =                                         \
            ((SPUS) > 0) ? gtx_active_mask(SPUS) : (uint16_t)0;                \
        for (uint8_t __nest = 0; __nest < (NESTS); __nest++) {                 \
            __split();                                                          \
            {                                                                   \
                __start_plan(__nest);                                           \
                    __start_shared();                                           \
                        __gtx_shared_body(__nest, __active_mask);              \
                    __end_shared();                                             \
                    /* thread loop only when SPUS > 0; shared-only kernels    \
                     * skip the per-SPU body entirely. */                      \
                    for (uint8_t __tid = 0; __tid < (SPUS); __tid++) {         \
                        uint16_t __tid_mask = (uint16_t)(1u << __tid);          \
                        __start_thread(__tid);                                  \
                            __gtx_thread_body(__nest, __tid, __tid_mask, (SPUS));\
                        __end_thread(__tid);                                    \
                    }                                                           \
                __end_plan(__nest);                                             \
            }                                                                   \
            __join();                                                           \
        }                                                                       \
    } while (0)

// Sugar: shared-only launch (no per-SPU body). Equivalent to GTX_LAUNCH(N, 0)
// but the macro still requires GTX_KERNEL_BODY to declare a (possibly empty)
// THREAD_BODY for the linker.
#define GTX_LAUNCH_SHARED(NESTS)   GTX_LAUNCH((NESTS), 0)

#endif  // GTX_KERNEL_H
