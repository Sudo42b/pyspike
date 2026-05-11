/**
 * fp16_utils.h — Integer-only FP16 utilities for GTX kernels
 *
 * All operations use uint16_t/uint32_t/int32_t only.
 * No float/double types to avoid FP instruction EBREAK on ISS/RTL.
 *
 * FP16 format: 1 sign | 5 exponent | 10 mantissa
 * FP32 format: 1 sign | 8 exponent | 23 mantissa
 */
#ifndef FP16_UTILS_H
#define FP16_UTILS_H

#include <stdint.h>

/**
 * Compare two FP16 values using integer arithmetic only.
 * Returns: positive if a > b, 0 if equal, negative if a < b
 */
static inline int32_t fp16_compare(uint16_t a, uint16_t b) {
    int32_t sa = (a >> 15) & 1;
    int32_t sb = (b >> 15) & 1;
    /* Both zero (±0 are equal) */
    if ((a & 0x7FFF) == 0 && (b & 0x7FFF) == 0) return 0;
    /* Different signs: positive > negative */
    if (sa != sb) return sa ? -1 : 1;
    /* Both positive: larger uint16 = larger value */
    if (!sa) return (int32_t)a - (int32_t)b;
    /* Both negative: smaller uint16 = larger value */
    return (int32_t)b - (int32_t)a;
}

/**
 * FP16 to FP32 bit conversion (returns uint32_t with float32 bits).
 * No float type used.
 */
static inline uint32_t fp16_to_f32bits(uint16_t h) {
    uint32_t sign = (uint32_t)(h >> 15) << 31;
    uint32_t exp  = (h >> 10) & 0x1F;
    uint32_t mant = h & 0x3FF;
    if (exp == 0) {
        if (mant == 0) return sign;  /* ±zero */
        /* Denormalized: normalize */
        exp = 1;
        while (!(mant & 0x400)) { mant <<= 1; exp--; }
        mant &= 0x3FF;
        return sign | ((uint32_t)(127 - 15 + exp) << 23) | (mant << 13);
    }
    if (exp == 0x1F) return sign | 0x7F800000 | (mant << 13);  /* inf/nan */
    return sign | ((uint32_t)(exp - 15 + 127) << 23) | (mant << 13);
}

/**
 * FP32 bits (as uint32_t) to FP16 conversion. Integer-only.
 */
static inline uint16_t f32bits_to_fp16(uint32_t f) {
    uint16_t sign = (f >> 16) & 0x8000;
    int32_t exp = ((f >> 23) & 0xFF) - 127 + 15;
    uint32_t mant = f & 0x7FFFFF;
    if (exp <= 0) {
        if (exp < -10) return sign;
        mant = (mant | 0x800000) >> (1 - exp);
        return sign | (uint16_t)(mant >> 13);
    }
    if (exp >= 0x1F) return sign | 0x7C00;
    return sign | (uint16_t)(exp << 10) | (uint16_t)(mant >> 13);
}

/**
 * FP32 integer addition: add two FP32 values represented as uint32_t bits.
 * NOTE: This is a simplified version for accumulation use.
 * For complex arithmetic (GATED_LINEAR_ATTN etc), use the firmware's
 * FP hardware through ISA intrinsics instead.
 */

/* Access FP16 value at volatile L1 address and return as uint16_t */
#define FP16_READ(addr, idx) (((volatile uint16_t*)(uintptr_t)(addr))[idx])
#define FP16_WRITE(addr, idx, val) (((volatile uint16_t*)(uintptr_t)(addr))[idx] = (val))
#define FP32_WRITE(addr, idx, val) (((volatile uint32_t*)(uintptr_t)(addr))[idx] = (val))
#define INT32_READ(addr, idx) (((volatile int32_t*)(uintptr_t)(addr))[idx])
#define INT32_WRITE(addr, idx, val) (((volatile int32_t*)(uintptr_t)(addr))[idx] = (val))

#endif /* FP16_UTILS_H */
