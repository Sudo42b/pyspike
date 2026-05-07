# Phase 5: VEC/ACT/Pool - Discussion Log

> **Audit trail only.** Do not use as input to planning, research, or execution agents.
> Decisions are captured in `05-CONTEXT.md` — this log preserves the alternatives considered.

**Date:** 2026-05-07
**Phase:** 05-vec-act-pool
**Areas discussed:** Module split (VEC + ACT + VRF-02 + waves), Activation direction asymmetry, VSUM/DOT precision, format_cvt + FP8 codec

---

## Area Selection

User selected all 4 candidate gray areas (multi-select).

| Area | Selected |
|------|----------|
| Module split for VEC + ACT | ✓ |
| Activation direction asymmetry | ✓ |
| VSUM partial-sum recombination | ✓ |
| format_cvt + FP8 codec strategy | ✓ |

---

## Area 1: Module split for VEC + ACT

### Q1: VEC module split — same 3-way pattern as P4 MM?

| Option | Description | Selected |
|--------|-------------|----------|
| 3-way mirror P4 (Recommended) | ops/vec.py + vec_engine.py + vec_core.py. VSUM is P7-numba hot candidate; same JIT-boundary rationale as gemm_core. | ✓ |
| 2-way (skip engine) | ops/vec.py inline decode + dispatch + vec_core.py. Fewer files; trade-off: P4 split made test_op_mm.py boundaries cleaner. | |
| 1 file per VEC family | ops/vec.py + sasmd.py + dot.py + vsum.py + clamp.py. Clearer per-op ownership; risks more import surface. | |

**User's choice:** 3-way mirror P4 (Recommended)
**Notes:** Direct application of P4 D-01 lineage to VEC.

### Q2: ACT module split — bundle or split?

| Option | Description | Selected |
|--------|-------------|----------|
| Bundle act+pool+format (Recommended) | ops/act.py + act_engine.py + act_core.py. Mirrors P4 single-engine pattern. Larger files (~400 LOC). | ✓ |
| Split per concern | ops/act.py + act_engine.py + activation_core.py + pool_core.py + format_core.py. Cleaner separation; FP8 codec lives alone. | |
| Bundle activation; split format/pool | Hybrid: activation_core.py + pool_core.py + format_core.py. Compromise. | |

**User's choice:** Bundle act+pool+format (Recommended)
**Notes:** Single-engine variant of P4 pattern. Re-evaluation trigger documented (act_core.py > 600 LOC → split).

### Q3: VRF-02 (verify_ref.py oracle) port location?

| Option | Description | Selected |
|--------|-------------|----------|
| tests/gtx/_oracles.py (Recommended) | Test-only tier (same as _verify_minimal.py / _mocks.py). Stays out of wheel. | ✓ |
| tests/gtx/oracles/{vec,act,format,pool}.py | Split per family for navigability. More imports per test. | |
| src/main/python/riscv/gtx/_verify_ref.py | Production-importable. Bundles into wheel (~500 LOC). Violates dev-only tier. | |

**User's choice:** tests/gtx/_oracles.py (Recommended)
**Notes:** Single file initially; split threshold deferred to plan-stage (~200 LOC).

### Q4: Wave structure?

| Option | Description | Selected |
|--------|-------------|----------|
| Mirror P4 (Recommended) | Wave 1a scaffold + Wave 1b 4 plans (sequential within wave) + Wave 2 integration. ~6 plans. | ✓ |
| Smaller waves, more plans | Split per op family. ~8-9 plans. Slower but more focused. | |
| Defer to plan-phase | Let gsd-planner figure out wave structure. Risks missing P4 test-file-collision lessons. | |

**User's choice:** Mirror P4 (Recommended)
**Notes:** Sequential within Wave 1b is mandatory due to test_op_*.py shared edit surface (P4 lesson).

---

## Area 2: Activation direction asymmetry

### Q1: How to encode forward/reverse direction?

| Option | Description | Selected |
|--------|-------------|----------|
| is_reversed param at @handler (Recommended) | Mirrors P4 D-05 is_accumulate precedent. Each @handler explicitly passes True/False. | ✓ |
| Literal direction dict + 1 dispatcher | Single _exec_activation reads dict. Risk: silent fallthrough on missing entry. | |
| Two separate @handler families | _exec_act_relu / _exec_act_gelu_rev. 8 functions; disasm shows direction in name. | |

**User's choice:** is_reversed param at @handler (Recommended)
**Notes:** Direct application of P4 D-05.

### Q2: Direction source-of-truth location?

| Option | Description | Selected |
|--------|-------------|----------|
| Inline in @handler (Recommended) | Each @handler self-documents. Direction next to dispatch site. | ✓ |
| Module-level REVERSED_OPS frozenset | Indirection between @handler and direction. Single edit point if taxonomy changes. | |
| Pytest parametrize fixture | Production imports from tests. Anti-pattern. | |

**User's choice:** Inline in @handler (Recommended)
**Notes:** No indirection for an 8-element set.

### Q3: ACT-05 _imm variants — same asymmetry?

| Option | Description | Selected |
|--------|-------------|----------|
| Same asymmetry, separate @handlers (Recommended) | 16 total @handlers (8 ISS + 8 L0). Direction taxonomy preserved. | ✓ |
| Defer ACT-05 to research | Lock ISS asymmetry; flag _imm direction question for researcher. | |
| Bundle _imm into same @handlers | 8 total; @handler branches on funct7 internally. Less disasm clarity. | |

**User's choice:** Same asymmetry, separate @handlers (Recommended)
**Notes:** Mirrors P4 D-04 "every variant separate @handler" lesson.

### Q4: format_cvt + pooling direction (always forward) — separate concern?

| Option | Description | Selected |
|--------|-------------|----------|
| Yes — hard-code forward, no asymmetry param (Recommended) | format_cvt and pooling don't carry is_reversed. Engine surface stays clean. | ✓ |
| Pass is_reversed=False explicitly for symmetry | Uniform dispatch shape across all @handlers. Slightly more boilerplate. | |
| Separate engine functions per family | firmware_act vs firmware_format vs firmware_pool. Matches D-02 single-engine = 3 entry points. | |

**User's choice:** Yes — hard-code forward, no asymmetry param (Recommended)
**Notes:** D-08 — clean engine surface for irrelevant param.

---

## Area 3: VSUM partial-sum recombination

### Context discovered during discussion

Vendor `gtx/CLAUDE.md` "VSUM 정밀도" section explicitly states:
> FP32 내부 누적 후 1회 FP16 변환. 레퍼런스 매칭 필요 시 행별 분할 후 FP16 부분합 재합산.

This resolves the apparent ROADMAP success #1 vs VEC-02 spec contradiction — they describe two distinct modes (single-call kernel vs row-split firmware composition), not a contradiction.

C++ `gtx_npu_vec.cc:103-108` (VSUM) and `:251-262` (DOT) confirmed: single-call FP32 internal accumulate + single FP16 cast via `gtx_fp32_to_16`.

### Q1: Lock both VSUM modes?

| Option | Description | Selected |
|--------|-------------|----------|
| Both modes — kernel + firmware separation (Recommended) | vec_core.vsum is mode-agnostic (always FP32 internal). Mode B is firmware composition. | ✓ |
| Single mode (FP32 internal only) | Treat row-split as documentation. Risk: firmware regression .elf could diverge silently. | |
| Add explicit mode parameter to kernel | vec_core.vsum(mode=...). Couples kernel to firmware concern; violates P4 D-03. | |

**User's choice:** Both modes — kernel + firmware separation (Recommended)
**Notes:** D-09 — composition pattern, not kernel feature.

### Q2: How to test both modes?

| Option | Description | Selected |
|--------|-------------|----------|
| Two test families (Recommended) | test_vsum_fp32_internal_anti_pattern + test_vsum_row_split_matches_cpp. Two distinct claims. | ✓ |
| Single ROADMAP-success-#1 test only | FP32-internal anti-pattern only. Defer row-split coverage to .elf regression. | |
| Defer row-split test to research | Lock kernel; flag row-split firmware exercise question to researcher. | |

**User's choice:** Two test families (Recommended)
**Notes:** D-10 — explicit row-split test prevents silent firmware-mode divergence.

### Q3: DOT precision contract?

| Option | Description | Selected |
|--------|-------------|----------|
| Yes, same contract (Recommended) | vec_core.dot = FP32 internal accumulate + single FP16 cast. Anti-pattern test parallel to VSUM. | ✓ |
| Only verify against C++ golden, no anti-pattern test | Trust C++ source; bit-exact compare against verify_ref.py oracle only. | |

**User's choice:** Yes, same contract (Recommended)
**Notes:** D-11 — symmetric with VSUM mode A.

### Q4: VSUM/DOT FP32→FP16 cast on overflow?

| Option | Description | Selected |
|--------|-------------|----------|
| IEEE round-to-nearest, inf on overflow (Recommended) | np.float16(fp32). Mirrors C++ gtx_fp32_to_16. inf is correct, not a bug. | ✓ |
| Saturate to FP16 max (no inf) | Diverges from C++. Reject. | |
| Defer to research | Lock IEEE; have research confirm gtx_fp32_to_16 actually uses IEEE conversion. | |

**User's choice:** IEEE round-to-nearest, inf on overflow (Recommended)
**Notes:** D-12 + explicit overflow anti-pattern test.

---

## Area 4: format_cvt + FP8 codec strategy

### Context discovered during discussion

C++ `gtx_npu_act.cc:223-272` confirmed format_cvt structure: 3 funct7 cases (0x20 FP8, 0x21 INT8, 0x?? INT32), each with sub_op LSB selecting direction.

C++ `gtx_npu.h:154` (`gtx_fp8_to_32`) and `:182` (`gtx_fp16_to_8`) confirmed FP8 codec is **labeled** E4M3 but has two intentional divergences:
1. Subnormal handling: `(h_frac/8) * 2^(-6)` (not standard E4M3's 2^-9 base)
2. exp=0xF maps to FP32 inf (standard E4M3 has no inf, only NaN)

### Q1: format_cvt @handler granularity?

| Option | Description | Selected |
|--------|-------------|----------|
| 1 per direction (Recommended) | 6 @handlers (FP16↔FP8/INT8/INT32 × 2 directions). Mirrors P4 D-04. | ✓ |
| 1 per funct7 (mirror C++) | 3 @handlers; each branches on sub_op LSB. Closer to C++ source. | |
| 1 monolithic _exec_format_cvt | Single @handler does everything. Anti-pattern vs P4 D-04. | |

**User's choice:** 1 per direction (Recommended)
**Notes:** D-13 — P4 D-04 lesson directly applied. Sub_op LSB encoding trick stays in HW; Python boundary clean.

### Q2: FP8→FP16 codec implementation?

| Option | Description | Selected |
|--------|-------------|----------|
| 256-byte LUT precomputed at import (Recommended) | LUT[fp8_byte] vectorized lookup. Bit-twiddle source-of-truth in builder docstring. | ✓ |
| Per-call bit-decode (1:1 with C++) | Inline gtx_fp8_to_32 mirror. Slower at scale; harder vectorization. | |
| Defer to research — verify LUT bit-exactness first | Lock LUT pending researcher confirmation of bit-identical output for all 256 inputs. | |

**User's choice:** 256-byte LUT precomputed at import (Recommended)
**Notes:** D-14 — LUT is cache, builder is spec.

### Q3: FP16→FP8 codec implementation?

| Option | Description | Selected |
|--------|-------------|----------|
| 64KB LUT precomputed at import (Recommended) | LUT[fp16.view(uint16)] vectorized. 64KB negligible. | ✓ |
| Per-call bit-decode (1:1 with C++) | Inline gtx_fp16_to_8 with RNE rounding. Smaller import footprint. | |
| Mixed — LUT for FP8→FP16 only, bit-decode for FP16→FP8 | Asymmetric. Two implementation styles. | |

**User's choice:** 64KB LUT precomputed at import (Recommended)
**Notes:** D-15 — both directions vectorizable.

### Q4: FP8 codec test coverage?

| Option | Description | Selected |
|--------|-------------|----------|
| Explicit subnormal + exp=0xF tests (Recommended) | Document custom (non-standard E4M3) semantics. 256-input round-trip identity. | ✓ |
| Just verify_ref oracle parity | Trust verify_ref.py covers it. Risk: may not exercise subnormals or exp=0xF. | |
| Defer to research | Lock explicit tests; researcher enumerates verify_ref.py coverage gaps. | |

**User's choice:** Explicit subnormal + exp=0xF tests (Recommended)
**Notes:** D-16 — divergence from NVIDIA E4M3 is documentation-worthy.

---

## More Gray Areas Check

After 4 selected areas completed, presented user with 4 additional areas they could explore:
- SASMD IS-vs-VS variant routing
- CLAMP L0/L1 path branching
- activation_relu_gelu.elf fixture sourcing strategy
- _imm L0 variants funct7 LUT

| Option | Description | Selected |
|--------|-------------|----------|
| I'm ready for context (Recommended) | Architectural shape + precision contract + codec strategy locked. Tactical decoding details to research. | ✓ |
| Explore SASMD/CLAMP routing | Concretize sub-encoding before research. | |
| Explore .elf fixture strategy | Vendor borrow vs hand-written .S. Affects Wave 1a scaffold. | |
| Explore _imm L0 variants | Routing matrix shape; immediate operand decoding. | |

**User's choice:** I'm ready for context (Recommended)
**Notes:** Same pattern as P4 — research locks ~9-12 deferred items.

---

## Claude's Discretion

The 12 items listed in CONTEXT.md `<decisions>` § "Claude's Discretion" — primarily research/plan-stage tactical decoding details (funct7 sub-encoding, packed-rs1 layouts, immediate operand positions, exact normalization formulas, oracle file split thresholds).

## Deferred Ideas

12 items captured in CONTEXT.md `<deferred>` section, split between explicit out-of-P5-scope deferrals (Phase 6 / Phase 7 / v2) and within-domain ideas surfaced but not selected for further discussion (CLAMP arange use case, IS variant L0 SVR[0] writeback, gtx_npu_pool.cc disambiguation, vec_engine granularity).
