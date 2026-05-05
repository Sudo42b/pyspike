# Phase 3: DMA & DDR I/O - Context

**Gathered:** 2026-05-05
**Status:** Ready for planning

<domain>
## Phase Boundary

Phase 3은 GTX NPU의 **메모리 이동(DMA + DDR I/O) 레이어**를 구축한다. 구체적으로:

1. **DMA op 풀세트** — `firmware_dma` (funct7=0x40 LOAD/STORE/COPY) + `firmware_dma_svr` (funct7=0x41 load_svr/store_svr/load_3d/store_3d) + `exec_dma_2d` / `exec_load_svr` / `exec_store_svr` / `exec_transpose` / `exec_transpose_ddr` / `exec_fill` 핸들러 (DMA-01, DMA-02)
2. **`firmware_dma_op` 패킹 인코딩 디코딩** — rs1 = `addr_hi[63:27] | addr_lo[26:0]`, rs2 = `height[63:48] | length[47:32] | stride[31:0]`, rs3 from `gspr[GSPR_GTX_OPERAND3]`, HW convention 0=65536(length) / 0=1(height) (DMA-02)
3. **S-loop deferred-store 큐** — `DeferredDdrStore` 데이터클래스 + push at S-loop store, flush API (트리거 위치는 research 결과로 잠금 — `end_p`/`end_s`/`credit_st_chk` 후보) (DMA-03)
4. **DDR hex I/O 양 모드** — `ddr_init_from_file` / `ddr_dump_to_file` + 표준 LTR(default) + `GTX_DDR_REVERSED=1` (32-byte 버스 워드 역순) 모두 round-trip (DMA-04)
5. **DDR 라운드트립 단위 회귀** — Python에서 L1 패턴 → DMA store → DDR → re-load → bit-exact 일치 (DMA-05)
6. **Mode 1 (no loop, broadcast NEST×SPU) + Mode 3 (P+S, single NEST DDR↔L2) dispatch 라우팅** — `riscv/gtx/dispatch.py`에 4-mode router 함수 추가 (DISP-03)
7. **Phase 1 `ensure_ddr` upgrade** — stub→C++ doubling-grow strategy 교체

다음 모두는 **Phase 3 비범위(out-of-scope)** — 다른 페이즈가 다룬다:

- `gemm_core` / MM op 변형 + `mxe_accum` write — Phase 4
- 첫 `.elf` 펌웨어 strict-mode 회귀 — Phase 4 success #4 (P3는 programmatic 회귀만)
- VEC/ACT/Pool op 핸들러 — Phase 5
- DMA-3D / IM2COL-N / IM2COL-D / MCAST 본격 구현 — v2 (`gtx_npu_disasm.inc`의 `mcast_*` 0x42/0x44, `load_3d`/`store_3d` 0x41 funct3=4/5는 disasm만 등록, 본체는 stub or 미포함)
- `mexec` full microcode 페치-디코드 루프 — v1 펌웨어 미요구 시 stub
- `verify.py` 포팅 + `riscv.gtx._verify` import path — Phase 6
- `[tool.setuptools.package-data]`에 ELF/golden hex 등록 — Phase 5/6
- `pyspike-verify` console script + spike 종료 시 자동 DDR dump 환경변수 훅 — Phase 6 또는 별도 follow-up

</domain>

<decisions>
## Implementation Decisions

### DMA 모듈 구성 (D-01 ~ D-03)

- **D-01:** **DMA = `ops/dma.py`(@handler 진입점) + `riscv/gtx/dma_engine.py`(순수 함수) 분리.**
  - `ops/dma.py`: `@handler(funct7=..., funct3=..., mnemonic=...)` 데코레이터로 진입점만 등록. spike 의존(`proc`, `insn`).
  - `riscv/gtx/dma_engine.py`: 실제 메모리 카피 로직 (NumPy + GtxMemory 인스턴스만 의존). spike 의존성 0 → 단위 테스트가 mock 없이 가능.
  - **이유:** P2 D-14 패턴을 약간 확장. C++ `gtx_npu_dma.cc` 단일 파일을 두 파일로 분리하지만 진입점 직역 일관성 유지. dma_engine.py를 통한 **회귀 디버깅 용이성** 확보.
- **D-02:** **`firmware_dma`(funct7=0x40)와 `firmware_dma_svr`(funct7=0x41) 별도 `@handler` 등록.**
  - 두 개의 각각 다른 dispatch entry. `_registry.py`의 funct7-keyed dispatch dict에 자연 들어맞음.
  - **이유:** dispatch 테이블 일관성, mnemonic 분리 (load/store/copy vs load_svr/store_svr/load_3d/store_3d).
- **D-03:** **funct3 sub-variant까지 데코레이터 분해 + 2-level dispatch.**
  - `@handler(funct7=0x40, funct3=0, mnemonic='load')`, `funct3=1 'store'`, `funct3=2 'copy'`로 함수 3개 분리.
  - `_registry.py`의 `mask_funct3=True` 경로가 disasm `add_rf3` 등록 트리거.
  - **dispatch 자료 구조: `dict[funct7, dict[funct3, Callable]]`** (Claude's discretion — `dict[(funct7, funct3), Callable]` 옵션도 plan에서 비교).
  - **`npu.custom0` 수정 필요:** 현재 `funct7 → handler` 단일 lookup → `funct7 → funct3-table → handler`. funct3 sub-decomposition 없는 funct7(예: 0x49 wrspr)은 sentinel `funct3=None` 키로 처리 또는 별도 dict 유지 (plan에서 정확화).
  - **이유:** mnemonic이 funct3마다 다르므로 disasm 분리가 자연. spike trace에 `load`/`store`/`copy`로 정확히 표시.

### Deferred-store 큐 (D-04 ~ D-06)

- **D-04:** **`@dataclass DeferredDdrStore` 7-field 정의** — `nest: int, l2_off: int, ddr_off: int, length: int, height: int, l2_stride: int, ddr_stride: int`. C++ `deferred_ddr_store_t` (gtx_npu.h) 직역.
  - 위치: `riscv/gtx/dma_engine.py` 또는 신규 `riscv/gtx/dma_state.py` (Claude's discretion — plan 단계에서 결정).
- **D-05:** **`self.deferred_ddr_stores: list[DeferredDdrStore]`를 `GtxNpu` 인스턴스에 보유.**
  - `npu.py:GtxNpu.__init__`에 `self.deferred_ddr_stores: list = []` 추가.
  - `reset()` 마지막에 `self.deferred_ddr_stores.clear()` 추가 (P2 reset 패턴 연장).
  - **이유:** P2 D-04 `WarpState` 패턴 일치. flush 트리거(end_p/end_s)도 GtxNpu가 받으므로 일관.
- **D-06:** **`npu.flush_deferred_ddr_stores() -> None` API** — C++ `flush_deferred_ddr_stores()` 직역.
  - 시그니처는 잠그되, **호출 트리거 위치(end_p / end_s / credit_st_chk)는 research가 확정.** ROADMAP P3 success #4는 "end_p에서 flush"이지만 C++ 실제 호출은 3곳에서 발견됨 (`gtx_npu_loop.cc:53`, `gtx_npu_dispatch.cc:902` `if (is_sloop)`, `gtx_npu_dispatch.cc:784`).
  - flush는 `for req in self.deferred_ddr_stores: ... copy L2→DDR` 후 `self.deferred_ddr_stores.clear()`.

### DDR Hex I/O (D-07 ~ D-09)

- **D-07:** **DDR I/O는 `riscv/gtx/ddr.py`에 채움** (Phase 1 D-13 의도 유지).
  - 시그니처: `def ddr_init_from_file(mem: GtxMemory, filename: str) -> None`, `def ddr_dump_to_file(mem: GtxMemory, filename: str, addr: int, size: int) -> None`.
  - 두 함수 모두 **순수 함수** — `mem` 인스턴스만 받음, spike 의존 0, op 로직과 I/O 분리.
  - **이유:** Phase 1 D-13 모듈 layout 준수, Layered API 분리 원칙(D-10) 강화, 단위 테스트가 spike 없이 가능.
- **D-08:** **`GTX_DDR_REVERSED` 환경변수는 매 I/O 호출 시 read** — `bool(os.environ.get('GTX_DDR_REVERSED'))` 함수 진입에서 1회 read.
  - **이유:** P2 D-07 `GTX_NO_EXIT` 패턴 일치. `monkeypatch.setenv('GTX_DDR_REVERSED', '1')` fixture로 양 모드 테스트 단순. import 후 변경 무시되는 캐시 패턴 회피.
- **D-09:** **`ddr_dump_to_file`은 인자만 받음** (라이브러리 깨끗 유지).
  - `GTX_DDR_DUMP` / `GTX_DDR_DUMP_ADDR` / `GTX_DDR_DUMP_SIZE` 환경변수 처리는 **CLI/진입점**에서만.
  - **위치는 plan에서 결정** — `riscv/gtx/__main__.py` 통합 vs 별도 dump-hook 모듈 vs P6 `pyspike-verify` console script.
  - **자동 dump (spike 종료 시 GTX_DDR_DUMP를 보고 dump)는 P6 또는 별도 follow-up.** Phase 3 success criteria가 모두 programmatic 호출이라 P3에서는 라이브러리 함수만 제공해도 충분.
  - **이유:** 환경변수 처리는 application-level concern. 라이브러리는 testable한 순수 인터페이스.

### Test 전략 (D-10 ~ D-12)

- **D-10:** **P3 회귀는 Python-only programmatic만** — `.elf` 빌드/픽스처 0건.
  - 모든 input은 NumPy 배열 합성. `firmware_dma` / `exec_dma_2d` / `ddr_init_from_file` / `ddr_dump_to_file` 직접 호출.
  - 첫 `.elf` strict-mode 통과는 **P4 success #4** (mm_basic.elf)에서 첫 등장. P3는 .elf 인프라 비용을 들이지 않음.
  - **이유:** ROADMAP P3 success #1~#5 모두 programmatic 경로. .elf 인프라(어셈블리 + cross-toolchain) 비용 ↑ 이득 ↓ in P3.
- **D-11:** **Deferred-store 테스트 = 이중 검증.**
  - **큐 push 동작:** `assert len(npu.deferred_ddr_stores) == N`, `assert npu.deferred_ddr_stores[0].nest == 1`, `.length == 4096` 등 dataclass 속성 직접 비교.
  - **flush 동작:** pre-flush DDR snapshot (`mem._ddr_bytes.copy()`) → `npu.flush_deferred_ddr_stores()` → post-flush DDR diff로 byte-exact 확인.
  - **이유:** 큐 push 시점과 flush 시점이 별도 회색지대 — 한 테스트로 묶으면 어느 단계 fail인지 진단 어려움.
- **D-12:** **MMU mock (`load_uint64`/`store_uint64`)은 P3에서 추가 안 함** (YAGNI).
  - P3 `firmware_dma`는 `proc.get_state().XPR[insn.rs1]` (GPR 읽기)만 사용. P2 D-19 약속한 MMU 확장은 **P4 plan**으로 미룸.
  - P3에서는 GPR mock만 확장 (P2 `MockProcessor.get_state().XPR.read/write`로 충분).
  - **이유:** YAGNI. P4 firmware_mm_op이 실제 MMU 호출 시점에 추가하는 게 정확한 지점.

### dispatch / ensure_ddr 보조 (D-13 ~ D-14)

- **D-13:** **`riscv/gtx/ddr.py:ensure_ddr` doubling-grow로 교체** (Phase 1 stub 업그레이드).
  - 로직: `new_size = max(end_offset, current_size * 2, INITIAL_FLOOR)` 패턴. `INITIAL_FLOOR`는 plan에서 결정 (예: 1MB).
  - cap 초과 시 `ValueError` (D-02 동작 유지).
  - **이유:** C++ parity. 큰 .elf 회귀(P4+)에서 alloc 횟수 ↓. Phase 1 D-01에 "Phase 3 will replace this with the C++ doubling-grow strategy"로 미리 약속됨.
- **D-14:** **Mode 1/3 4-mode dispatch 라우팅 = `riscv/gtx/dispatch.py`에 함수.**
  - 시그니처(plan에서 정확화): `def dispatch_4mode(npu: 'GtxNpu', proc, insn, opcode: int, op1: int, op2: int, op3: int) -> int`.
  - 현재 `dispatch.py`(`build_custom0_table`/`build_custom1_table`)와 같은 파일에 공존. `firmware_dma`/`firmware_mm`(P4)/`firmware_vec`(P5)/`firmware_act`(P5) 등 모든 mode-aware op이 호출.
  - 4-mode 분기:
    - Mode 1 (`!is_ploop`) → broadcast all NEST×SPU (`for n in range(GTX_NEST_NUM): for s in range(GTX_SPU_NUM): dispatch_iss_opcode(n, s, ...)`)
    - Mode 2 (`is_ploop && !is_sloop && !is_tloop`) → broadcast SPU within `tmu_id`
    - Mode 3 (`is_ploop && is_sloop`) → `exec_dma_2d(tmu_id, ...)` single NEST
    - Mode 4 (`is_ploop && is_tloop`) → single `(tmu_id, curr_id)` SPU
  - `tmu_id` / `curr_id`는 `GtxNpu`의 워프 상태에서 read (P2 `WarpState`로부터 확장 — 정확한 필드 위치는 plan 단계).
  - **이유:** 모든 op이 공유하는 라우팅이므로 dispatch.py에 두는 게 자연. 핸들러는 결과(nest_id, spu_id)만 받음. ROADMAP P3 success #5 정합.

### Claude's Discretion

다음은 implementation detail로 Claude 판단 (plan/research 단계에서 정확화):

- `_registry.py`의 2-level dispatch 자료 구조 (`dict[int, dict[int, Callable]]` vs `dict[(int, int), Callable]`)
- `npu.custom0` 분기 정확한 형태 (funct3 sub-decomposition 없는 funct7 처리: sentinel `funct3=None` vs 별도 dict 유지)
- `dma_engine.py` 모듈명 (`dma_engine.py` vs `dma_kernels.py` vs `dma_helpers.py`)
- `DeferredDdrStore` 정의 위치 (`dma_engine.py` vs 신규 `dma_state.py`)
- `riscv/gtx/dispatch.py:dispatch_4mode` 인자 시그니처 정확한 형태 (`opcode` 파라미터 vs `npu`에서 직접 read)
- P3 disasm 항목 정확한 목록 (load/store/copy/load_svr/store_svr/load_3d/store_3d/tpose/fill 9개 + mcast 4개 P3 포함 여부 — 추천: P3는 9개 active + mcast 4개 stub-disasm-only 또는 P5 미룸, plan 결정)
- `tests/gtx/test_ddr_modes.py` `monkeypatch.setenv` fixture vs `os.environ`/`unittest.mock.patch.dict` 스타일
- `ensure_ddr` doubling-grow의 `INITIAL_FLOOR` 정확한 값 (1MB / 64KB / 32B 버스 단위)
- DDR hex parser의 `@offset` 라인 + 64-char hex 라인 처리 (line iteration vs mmap, half-density packing 처리는 C++ 직역)
- `flush_deferred_ddr_stores` 호출 트리거 정확한 위치 — research가 잠금
- `firmware_dma_op` rs1/rs2 비트 분해 helper (`encode_addr_pair(addr_hi, addr_lo) -> int` 같은 inverse helper 제공 여부 — 테스트 친화)

### Folded Todos

None — `gsd-tools todo match-phase 3`에서 매칭 0건.

</decisions>

<canonical_refs>
## Canonical References

**Downstream agents MUST read these before planning or implementing.**

### Project documents (locked context)
- `.planning/PROJECT.md` — Core Value, Constraints, Out of Scope, Key Decisions
- `.planning/REQUIREMENTS.md` — DMA-01..05, DISP-03 v1 acceptance criteria
- `.planning/ROADMAP.md` — Phase 3 success criteria 1-5 + research-flag 노트
- `.planning/STATE.md` — 현재 진행 (Phase 1, 2 완료; Phase 3 ready to plan)
- `.planning/phases/01-foundation/01-CONTEXT.md` — D-01..D-17 (특히 D-01/D-02 ensure_ddr·GTX_DDR_SIZE, D-03 DDR_REVERSED I/O 경계, D-10..D-12 layered API + view 보장, D-13 모듈 layout)
- `.planning/phases/02-skeleton-disasm/02-CONTEXT.md` — D-01..D-23 (특히 D-04 WarpState dataclass, D-13/D-14 per-op decorator registry + ops/{spr,control}.py, D-09 disasm 누적, D-19 mock spec, D-22 .elf fixture 패턴)

### Phase 3 research (생성 예정)
- `.planning/phases/03-dma-ddr-i-o/03-RESEARCH.md` — `/gsd:research-phase 3`이 plan 전에 생성. 다음을 잠금:
  - `firmware_dma_op` rs1/rs2/rs3 정확한 비트 분해 + HW convention 0=65536/0=1
  - deferred-store flush 트리거 위치 (end_p vs end_s vs credit_st_chk; ROADMAP 명시 vs C++ 실제 호출 3곳 정합)
  - DMA-3D/MCAST의 P3 포함 여부 (mcast_s2l/g2s/s2s/copy_mem/copy_l2/load_3d/store_3d)

### C++ ground-truth (via submodule, Phase 1 D-04)

**Primary (P3 핵심 직역 대상):**
- `vendor/gtx_cpp_reference/gtx/gtx_npu_dma.cc` — 558 LOC. 다음 함수 모두 P3 직역 대상:
  - `firmware_dma` (256-397): rs1/rs2/rs3 디코드 + S-loop/T-loop 분기 + deferred-store push
  - `exec_dma_2d` (25-) : DDR↔L2/L1 DMI 카피
  - `exec_load_svr` (97-) / `exec_store_svr` (118-) : L1↔L0 32B SVR 카피
  - `exec_transpose` (143-) / `exec_transpose_ddr` (175-) / `exec_fill` (230-)
  - `flush_deferred_ddr_stores` (415-435)
  - `ddr_init_from_file` (438-502): @offset + 64-char hex line + GTX_DDR_REVERSED 분기
  - `ddr_dump_to_file` (509-): 32 bytes per line + reversed 분기
- `vendor/gtx_cpp_reference/gtx/gtx_npu_dispatch.cc` (1154 LOC) — `dispatch()` (25-143) 4-mode router (D-14 직역 source), `dispatch_iss_opcode()` (151-)
- `vendor/gtx_cpp_reference/gtx/gtx_npu.h` — `deferred_ddr_store_t` struct 정의 (D-04 직역 source), `gtx_npu_t` 클래스의 `deferred_ddr_stores`, `ddr_data()`, `ensure_ddr()` 시그니처, GSPR_GTX_OPERAND1/2/3/OPCODE address 상수
- `vendor/gtx_cpp_reference/gtx/gtx_npu_disasm.inc` — DMA 관련 mnemonic (167-186): load/store/copy(0x40), load_svr/store_svr/load_3d/store_3d(0x41), mcast_*(0x42/0x44), load_svr_l1/store_svr_l1(0x43/0x45), tpose/fill(0x38/0x39)
- `vendor/gtx_cpp_reference/gtx/gtx_npu_loop.cc` (142 LOC) — flush_deferred_ddr_stores 호출 사이트 1 (line 53), warp 종료 핸들러 (D-06 트리거 candidate)

**Secondary (참고):**
- `vendor/gtx_cpp_reference/gtx/gtx_npu_core.cc` — `ensure_ddr` C++ doubling-grow 정확한 구현 (D-13 직역 source)
- `vendor/gtx_cpp_reference/gtx/gtx_params.h` — GTX_NUM_NESTS=4, GTX_SPUS_PER_NEST=16, GTX_L2_SIZE, GTX_DDR_SIZE, NSPR_THREAD_MASK 등
- `vendor/gtx_cpp_reference/gtx/CLAUDE.md` — 메모리 계층, FP16 LE 바이트 순서 규약, GTX_DDR_REVERSED 동작 명세 (Default LTR / =1 RTL/SystemC parity)

### Existing pyspike + Phase 1/2 산출물 (이미 wheel/repo에 land)

**Phase 1 자산:**
- `src/main/python/riscv/gtx/__init__.py` — 패키지 진입점, LE byteorder tripwire
- `src/main/python/riscv/gtx/params.py` — GTX_NEST_NUM=4, GTX_SPU_NUM=16, GTX_L0/L1/L2_SIZE_BYTES
- `src/main/python/riscv/gtx/encoding.py` — funct7/funct3/Mode/GSPR address 상수 (P3가 GSPR_GTX_OPERAND1/2/3/OPCODE 추가 가능)
- `src/main/python/riscv/gtx/memory.py` — GtxMemory layered API (l0_byte/l1_byte/l2_byte raw + l1_f16/l2_f16 named, _ddr_bytes lazy)
- `src/main/python/riscv/gtx/ddr.py` — **P3가 채움**: 현재 `get_ddr_cap` + stub `ensure_ddr`만 있음 → D-13 doubling-grow 교체 + D-07 init/dump 신규 함수 추가

**Phase 2 자산:**
- `src/main/python/riscv/gtx/npu.py` — GtxNpu 클래스 (P3가 `self.deferred_ddr_stores: list` 추가, reset()에 clear, custom0이 2-level dispatch로 확장)
- `src/main/python/riscv/gtx/_registry.py` — `@handler` 데코레이터 (P3가 `mask_funct3=True` 경로 활성화)
- `src/main/python/riscv/gtx/dispatch.py` — `build_custom0/1_table` (P3가 `dispatch_4mode` 함수 신규 추가)
- `src/main/python/riscv/gtx/disasm.py` — disasm 유틸리티 (P3가 add_rf3 호출 — 이미 존재)
- `src/main/python/riscv/gtx/warp_state.py` — `WarpState(is_ploop, is_tloop, is_sloop)` (P3가 `is_sloop` 활성화)
- `src/main/python/riscv/gtx/ops/spr.py` — P2 SPR 핸들러 (P3는 직접 수정 안 함)
- `src/main/python/riscv/gtx/ops/control.py` — P2 warp 제어 핸들러 (P3가 `end_p`/`end_s` 핸들러에 `npu.flush_deferred_ddr_stores()` 호출 추가)

**Phase 2 테스트 자산:**
- `tests/gtx/conftest.py` + `tests/gtx/_mocks.py` — Hybrid mock (D-17/D-19), P3가 GPR mock 확장 (MMU는 P4로 미룸)
- `tests/gtx/test_register.py` / `test_reset.py` / `test_dispatch.py` / `test_warp.py` / `test_skeleton.py` — P3 신규 테스트는 동일 패턴 따름 (`_RISCV_AVAILABLE` self-detect)

### Build / Distribution
- `pyproject.toml` — P3 직접 수정 없음 (PKG-01 ELF/golden hex 등록은 P5/P6)
- `MANIFEST.in` — Phase 1에서 vendor/gtx_cpp_reference prune 적용됨, P3 추가 작업 없음

</canonical_refs>

<code_context>
## Existing Code Insights

### Reusable Assets (Phase 1/2 산출물 — P3가 직접 사용/확장)

- **`riscv.gtx.memory.GtxMemory`** — layered API (l1_byte/l2_byte raw + l1_f16/l2_f16 named, view 보장 D-12). P3 dma_engine.py가 모든 메모리 카피의 source/destination으로 사용. `_ddr_bytes` lazy 속성을 D-13 doubling-grow로 통과.
- **`riscv.gtx.ddr.ensure_ddr` / `get_ddr_cap`** — P1 stub. D-13에서 교체. `mem._ddr_bytes` lazy 속성에 직접 접근.
- **`riscv.gtx._registry.handler`** — P2 D-13. P3 ops/dma.py 모든 핸들러 등록 통과. `mask_funct3=True` 경로가 disasm rf3 entry 생성 트리거 (현재 코드 line 92-94 이미 구현됨).
- **`riscv.gtx.dispatch.build_custom0_table`** — P2. P3에서 funct7=0x40/0x41이 추가되면 자동 등록. 단 D-03 2-level dispatch로 확장하려면 이 빌더 + npu.custom0 모두 수정.
- **`riscv.gtx.warp_state.WarpState`** — P2. P3가 `is_sloop` 필드를 처음 활성화. `start_s`/`end_s` 핸들러(`ops/control.py`)는 이미 P2에서 등록됨.
- **`riscv.gtx.encoding`** — funct7 0x40/0x41/0x38/0x39, GSPR_GTX_OPERAND1/2/3/OPCODE 주소 상수 P3 추가 위치. 또는 dma 모듈 내 local 상수.
- **`tests/gtx/_mocks.py`** + **`tests/gtx/conftest.py`** — `MockProcessor.get_state().XPR.read/write`만 P3에서 사용. MMU는 P4 미룸 (D-12).

### Established Patterns

- **모듈 명명:** lowercase + underscore (`dma_engine.py` 후보, `dma.py` ops/ 하위).
- **Type hints:** explicit, mypy-checked (`pytest --mypy`).
- **TDD:** test_*.py RED → 모듈 GREEN → 픽스 (P1, P2에서 확립).
- **Test 격리:** `--noconftest -o "addopts="` 우회 + 모듈 레벨 `_RISCV_AVAILABLE` self-detect (P2 plan 05 D-1).
- **`@handler(kind='custom0', funct7=, funct3=, mnemonic=, mask_funct3=)` 데코레이터** — P2 D-13. P3가 mask_funct3=True 경로 처음 사용.
- **TDD pattern (P2 확립):** `test_dma_*.py` 작성 → `dma_engine.py` 함수 작성 → `ops/dma.py` `@handler` 진입점 작성 → 통합 테스트.
- **C++ 직역 + 명시적 byte 조작:** `spu.l1[off] = lo; spu.l1[off+1] = hi` 패턴 P3에서도 LE 가정 유지.
- **반복적 함수 분리:** GPR read는 `proc.get_state().XPR[insn.rs1]` 직접 (CORE-04 우회), 단일 helper 미사용 (P2 D-05 데코레이터 wrap이 자동 처리).

### Integration Points

- **`riscv.gtx.npu.GtxNpu.custom0`** — D-03에 따라 funct3 sub-dispatch 추가 (현재 line 118-123 단일 lookup → 2-level).
- **`riscv.gtx.npu.GtxNpu.__init__`** — D-05에 따라 `self.deferred_ddr_stores: list = []` 추가.
- **`riscv.gtx.npu.GtxNpu.reset`** — D-05에 따라 `self.deferred_ddr_stores.clear()` 추가.
- **`riscv.gtx.dispatch`** — D-14에 따라 `dispatch_4mode` 함수 신규 추가. 기존 builder들과 같은 파일.
- **`riscv.gtx.ops.control`** — D-06 flush 트리거를 위해 `end_p` 또는 `end_s` 핸들러에 `npu.flush_deferred_ddr_stores()` 호출 추가 (research 결과로 정확한 트리거 잠금).
- **`riscv.gtx.ddr`** — D-07/D-13에 따라 P1 stub을 P3 doubling-grow + init/dump 함수로 확장.

### Anti-patterns to avoid (PITFALLS.md / 본 phase 추가)

- DMA 카피에서 host endian 가정 (`arr.view(np.float16)` 직접) — 항상 byte-level memcpy로 LE 보장 (Phase 1 PITFALLS Pitfall 1).
- DDR hex 파일 라인 단위 미세 read (성능 ↓) — 가능한 한 `f.read()` 후 split 또는 efficient line-by-line.
- `os.environ.get('GTX_DDR_REVERSED')`을 모듈 import 시점에 cache — D-08 위배. 매 호출 read.
- deferred_ddr_stores를 GtxMemory에 두기 — D-05 위배. GtxNpu가 보유.
- `flush_deferred_ddr_stores`을 GtxMemory에 두기 — D-06 위배. GtxNpu 메서드.
- `ddr_dump_to_file`이 환경변수 GTX_DDR_DUMP_ADDR/SIZE를 직접 read — D-09 위배. CLI/진입점 책임.
- ELF 빌드를 P3 테스트에 추가 — D-10 위배. P4까지 미룸.
- MMU mock을 P3 conftest에 추가 — D-12 위배 (YAGNI). P4 plan에서 추가.
- `ensure_ddr` 매번 새 ndarray + memcpy — D-13 위배. doubling-grow로 alloc 횟수 ↓.
- `firmware_dma` 단일 거대 함수 (300+ LOC) — D-01 위배. dma_engine.py로 순수 로직 추출.

</code_context>

<specifics>
## Specific Ideas

### dma_engine.py 표면 시안 (참고용 — plan 단계에서 정확화)

```python
# riscv/gtx/dma_engine.py
from dataclasses import dataclass
from .memory import GtxMemory
from .ddr import ensure_ddr  # D-13 doubling-grow

@dataclass
class DeferredDdrStore:
    nest: int
    l2_off: int
    ddr_off: int
    length: int
    height: int
    l2_stride: int
    ddr_stride: int

def firmware_dma_load(mem: GtxMemory, *, nest: int, ddr_off: int, l2_addr: int,
                     length: int, height: int, rd_stride: int, wr_stride: int,
                     is_sloop: bool, is_tloop: bool, spu: int = 0) -> int:
    """S-loop: DDR→L2 immediate. T-loop: L2→L1. Returns cycles (ignored in functional model)."""
    ddr = ensure_ddr(mem, ddr_off + (height * max(rd_stride, length)))
    if is_sloop:
        for row in range(height):
            ddr_o = ddr_off + row * rd_stride
            l2_o = (l2_addr + row * wr_stride) % GTX_L2_SIZE_BYTES
            copy_len = min(length, mem.l2_byte(nest).size - l2_o, ddr.size - ddr_o)
            if copy_len > 0:
                mem.l2_byte(nest)[l2_o:l2_o+copy_len] = ddr[ddr_o:ddr_o+copy_len]
    elif is_tloop:
        # L2 → L1 (per-SPU)
        ...
    return 0
```

### ops/dma.py 표면 시안

```python
# riscv/gtx/ops/dma.py
from .._registry import handler
from .. import dma_engine

@handler(kind='custom0', funct7=0x40, funct3=0, mnemonic='load', mask_funct3=True)
def _firmware_dma_load(npu, proc, insn, xs1, xs2):
    rs1 = proc.get_state().XPR[insn.rs1]
    rs2 = proc.get_state().XPR[insn.rs2]
    rs3 = npu.gspr.get(GSPR_GTX_OPERAND3, 0)
    addr_hi = (rs1 >> 27) & 0x1FFFFFFFFF
    addr_lo = rs1 & 0x7FFFFFF
    height = (rs2 >> 48) & 0xFFFF or 1
    length = (rs2 >> 32) & 0xFFFF or 0x10000
    rd_stride = rs2 & 0xFFFFFFFF
    wr_stride = rs3 & 0xFFFFFFFF
    if npu.warp.is_sloop:
        nest = npu.tmu_id   # warp_state field, plan에서 정확화
        return dma_engine.firmware_dma_load(npu.mem, nest=nest, ddr_off=addr_hi, l2_addr=addr_lo, ...)
    elif npu.warp.is_tloop:
        nest, spu = npu.tmu_id, npu.curr_id
        return dma_engine.firmware_dma_load(npu.mem, nest=nest, ddr_off=addr_hi, l2_addr=addr_lo, spu=spu, ...)
    return 0
```

### test_dma_roundtrip.py assertion 골자

```python
# Programmatic round-trip — D-10
def test_dma_l1_to_ddr_roundtrip():
    npu = GtxNpu()
    npu.reset(MockProcessor())  # sp init
    pattern = np.arange(4096, dtype=np.float16)
    npu.mem.l1_f16(0, 0)[0:4096] = pattern   # write L1

    # Build firmware_dma STORE instruction (synthetic encoding)
    # ... call exec_store_svr to push to L2, exec_dma_2d to push to DDR

    # Dump to hex file
    ddr_dump_to_file(npu.mem, '/tmp/dump.hex', addr=0, size=8192)

    # Round-trip: reload from hex, run reverse, compare
    npu2 = GtxNpu()
    ddr_init_from_file(npu2.mem, '/tmp/dump.hex')
    ...
    assert np.array_equal(npu2.mem.l1_f16(0, 0)[0:4096], pattern)


def test_ddr_modes_differ():
    # D-08: monkeypatch.setenv pattern
    pattern = np.arange(64, dtype=np.float16)
    npu = GtxNpu()
    npu.mem._ddr_bytes = pattern.view(np.uint8).copy().tobytes()  # 또는 ensure_ddr

    ddr_dump_to_file(npu.mem, '/tmp/ltr.hex', 0, 128)   # default LTR

    monkeypatch.setenv('GTX_DDR_REVERSED', '1')
    ddr_dump_to_file(npu.mem, '/tmp/rev.hex', 0, 128)

    # 파일 내용이 다르고, 각 모드 round-trip은 자기 자신과 일치
    assert open('/tmp/ltr.hex').read() != open('/tmp/rev.hex').read()
```

### Deferred-store 이중 검증 (D-11)

```python
def test_deferred_store_queue_push():
    # 큐 push 모양 검증
    npu = GtxNpu()
    npu.warp.is_ploop = True
    npu.warp.is_sloop = True
    # synthetic firmware_dma STORE invocation
    proc = MockProcessor()
    proc.get_state().XPR.write(1, encode_addr_pair(0x1000, 0x100))   # rs1
    proc.get_state().XPR.write(2, encode_dim_pair(4, 256, 256))      # rs2: height=4, length=256, stride=256
    insn = MockInsn(funct=0x40, xd=0, xs1=0, xs2=1, rs1=1, rs2=2)    # STORE
    npu.custom0(proc, insn, 0, 0)

    assert len(npu.deferred_ddr_stores) == 1
    assert npu.deferred_ddr_stores[0].nest == 0
    assert npu.deferred_ddr_stores[0].length == 256
    assert npu.deferred_ddr_stores[0].height == 4

def test_deferred_store_flush_diff():
    # flush 동작 검증
    npu = GtxNpu()
    # ... setup ...
    snapshot = npu.mem._ddr_bytes.copy() if npu.mem._ddr_bytes is not None else None
    npu.flush_deferred_ddr_stores()
    # post-flush DDR diff
    assert (snapshot != npu.mem._ddr_bytes).any()   # bytes changed
    # specific bytes match expected L2 content
```

### 사용자 강하게 선호한 패턴 (Phase 1/2/3 일관)

- **순수 함수 + spike 의존 0의 단위 테스트** (D-01 dma_engine.py / D-07 ddr.py 디자인 동기)
- **C++ 직역 우선, 그러나 module split은 Python 응집도에 맞춤** (D-01 ops/ + helper 분리, D-06 flush API 명명 직역)
- **실제 사용 시점까지 confer 미룸** (D-12 MMU mock YAGNI, D-09 자동 dump hook P6 미룸)
- **테스트가 양 모드/이중 검증** (D-11 큐 push + flush, D-08 매 호출 환경변수 read로 양 모드 fixture)

</specifics>

<deferred>
## Deferred Ideas

### Phase 3 plan/research가 정확화 (이번 discuss 비범위)

- `firmware_dma_op` rs1/rs2/rs3 비트 분해 정확한 마스크 + HW convention edge case (length=0=65536, height=0=1, addr_hi 27-bit boundary) — `/gsd:research-phase 3` 결과로 잠금
- deferred-store flush 트리거 정확한 위치 (end_p / end_s / credit_st_chk; ROADMAP success #4는 end_p, C++ 호출 사이트 3개 발견) — research 잠금
- DMA-3D / IM2COL-N / IM2COL-D / MCAST 핸들러의 P3 포함 여부 — research 잠금 (회귀 .elf가 요구하면 v1 승격, 아니면 v2 미룸)
- `mexec` full microcode loop — v1 펌웨어가 트리거하면 stub 작성 (P5+ deferred per STATE.md)

### Out of scope for Phase 3 (다른 페이즈로)

- **MM op 핸들러 + gemm_core + mxe_accum write** → P4 MM-01..05
- **첫 .elf strict-mode 회귀 (mm_basic.elf)** → P4 success #4 (P3는 programmatic only)
- **VEC/ACT/Pool op** → P5 VEC-01..05, ACT-01..05
- **`verify.py` 포팅 → `riscv.gtx._verify`** → P6 VRF-01
- **`tests/gtx/data/{golden,elf}/` 패키지 데이터 등록** → P6 PKG-01
- **`pyspike-verify` console script** → P6 PKG-03
- **MMU mock (`load_uint64`/`store_uint64`)** → P4 plan에서 추가 (D-12)
- **자동 DDR dump (spike 종료 시 GTX_DDR_DUMP 환경변수 처리)** → P6 또는 별도 follow-up (D-09)

### v2 / 향후 마일스톤

- **PY-OVRD-01** (per-op before/after 후크 — DMA에도 적용 가능)
- **PY-FUNCT7-01** (외부 funct7 등록 API — 사용자 정의 DMA op)
- **DMA-V2-01** (DMA-LOAD-3D / IM2COL-N/D / MCAST 풀 구현)
- **CYC-01/02** (사이클 카운팅 — 현재 모든 DMA가 instantaneous)
- **DEV-01** (PCIe-EP `riscv.dev.MMIO` 재구현)

### Reviewed Todos (not folded)

None — `gsd-tools todo match-phase 3` 매칭 0건.

### Defer to user follow-up

- **D-09 자동 dump hook 위치 결정** — P6 plan 또는 별도 P3.x follow-up. Phase 3 success #1 자체는 programmatic이라 차단 없음.
- **mcast_* / load_3d/store_3d disasm-only stub vs 미포함** — research 결과 + plan 단계 합의. ROADMAP P3 success는 명시 안 함.
- **dispatch.py 단일 파일 vs `dispatch_4mode.py` 분리** — D-14 단일 파일 결정했으나 line count 부풀면 plan 단계에서 재고 가능.

</deferred>

---

*Phase: 03-dma-ddr-i-o*
*Context gathered: 2026-05-05 via /gsd:discuss-phase 3*
