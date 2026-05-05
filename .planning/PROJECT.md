# pyspike + GTX NPU (Python RoCC Port)

## What This Is

기존 C++ Spike RoCC 확장으로 구현된 GTX NPU functional model
(`~/NIGHTLY/gtx_spike/gtx/`)을 **pyspike의 `riscv.isa.ROCC` 서브클래스**로
**순수 Python(NumPy 백엔드)으로 재작성**하는 프로젝트. 결과물은 pyspike wheel
패키지에 동봉되어, 사용자가 `pip install spike` 후 한 줄로 GTX NPU 시뮬레이션을
띄우고 ISA/op를 Python에서 자유롭게 변형·검증할 수 있게 한다.

## Core Value

**기존 NPU 펌웨어(.elf) 회귀 테스트가 pyspike+Python NPU에서도 그대로 통과하고
DDR 결과가 C++ libgtx_npu.so(SystemC HW sim과 ULP 내 일치 검증 완료된 golden)와
ULP 허용오차 내로 일치한다 — 이게 안 되면 다른 어떤 기능도 의미가 없다.**

## Requirements

### Validated

<!-- pyspike 측에서 이미 동작 중인 기존 코드베이스 능력 (codebase mapping에서 파생) -->

- ✓ **PYS-EXT-01**: pybind11 트램폴린(`py_extension_t`, `py_rocc_t`)으로
  `extension_t`/`rocc_t` 가상 메서드를 Python에서 오버라이드 가능 — existing
- ✓ **PYS-EXT-02**: `@riscv.isa.register("name")` 데코레이터 + `py_register_extension`
  공장 람다로 spike 확장 레지스트리에 자동 등록 — existing
- ✓ **PYS-EXT-03**: `PYSPIKE_LIBS` 환경변수 + `PythonBridge::bootstrap`로 Python
  확장 모듈 동적 로드 — existing
- ✓ **PYS-EXT-04**: `pyspike` CLI 래퍼가 `--extlib=foo.py` / `--extlib=lib.so`를
  C++/Python 분리 후 spike에 전달 — existing
- ✓ **PYS-EXT-05**: `rocc_insn_t`(opcode 0x0b/0x2b/0x3b/0x7b) → `custom0/1/2/3()`
  Python 디스패치 경로 동작 확인 — existing
- ✓ **PYS-EXT-06**: cibuildwheel manylinux2014_x86_64 wheel 빌드 파이프라인
  (Python 3.8–3.12 baseline; **Phase 1 discuss에서 cp310–cp312로 축소 결정 — D-08**) — existing
- ✓ **PYS-EXT-07**: `riscv.dev.MMIO` 베이스 클래스로 Python MMIO 디바이스 모델
  지원 — existing
- ✓ **GTX-MEM-01**: 모든 L1/L0 FP16 접근을 little-endian 바이트 순서로 처리
  (SystemC TLM 일치 — bit-exact 필수 조건) — **Validated in Phase 1: Foundation**
  (D-17 LE byte-order assertion 8/8 PASS; `__init__.py` non-LE host tripwire active)
- ✓ **GTX-REF-01**: 기존 C++ gtx 소스를 `vendor/gtx_cpp_reference/` 스냅샷으로 두어
  golden 비교용 baseline + 포팅 시 ground-truth로 활용 — **Validated in Phase 1: Foundation**
  (D-04 `https://github.com/Sudo42b/gtx_spike` submodule 등록, D-06 wheel 미포함)
- ✓ **GTX-DMA-01**: DMA load/store/copy + SVR + transpose + DDR init/dump
  (left-to-right / `GTX_DDR_REVERSED` 양 모드) 구현 — **Validated in Phase 3: DMA & DDR I/O**
  (DMA-01..05 + DISP-03 closed; 6 exec_* helpers + DeferredDdrStore + 2-level custom0 dispatch
  + 4-mode router + ddr doubling-grow + GTX_DDR_REVERSED bit-exact round-trip; 179/179 P3 tests green)

### Active

<!-- 이번 마일스톤에서 빌드해 갈 가설들. 출하 시점에 Validated로 이동. -->

- [ ] **GTX-CORE-01**: `riscv.isa.ROCC` 서브클래스로 `GtxNpu` 클래스 작성, NEST(4)×SPU(16)
  메모리 계층(GSPR/NSPR/LSPR + L0/L1/L2/DDR)을 NumPy ndarray(`np.float16`)로 표현
- [ ] **GTX-CORE-02**: `custom0()`/`custom1()` 진입점에서 funct7 디스패치 + P/S/T
  워프 루프 상태머신(4-mode 라우팅) 구현
- [ ] **GTX-MM-01**: MM 서브시스템 (`gemm_core`, `exec_mm*`, `mxe_accum`,
  `firmware_mm_op`) 우선 구현 — **NPU 핵심**
<!-- GTX-DMA-01 → Validated in Phase 3 (moved above) -->
- [ ] **GTX-VEC-01**: VEC 서브시스템 (SASMD add/sub/mul/div, DOT/VSUM, CLAMP,
  L1/L0 분기) 구현 — VSUM FP32 누적 정밀도 규약 포함
- [ ] **GTX-ACT-01**: ACT 서브시스템 (RELU/SOFTMAX/ESUM 정방향, PRELU/GELU/TANH/SIGM
  역방향, pooling, format_cvt) 구현
- [ ] **GTX-SPR-01**: GSPR(0x000–0x3FF) / NSPR(0x400–0x7FF) / LSPR(0x800–0xBFF)
  읽기·쓰기 (`wr_spr`, `rd_spr`, `RDSPR` writeback) 구현
- [ ] **GTX-DISP-01**: 통합 오피코드 라우터 `dispatch_iss_opcode` (0=MM, 1=VEC,
  2=ACT, 3=DMA) + gem5 간소화(0x04–0x07) / ISS full(0x00–0x7F) 양 인코딩 공존
  — Phase 3 부분 진행: 4-mode 라우터(`dispatch_4mode`) + DMA-only `dispatch_iss_opcode`
  스텁 land (DISP-03). MM/VEC/ACT funct7 fillers는 P4/P5 잔여
- [ ] **GTX-DISASM-01**: `gtx_npu_disasm.inc`에 대응하는 `disasm_insn_t` 테이블을
  Python `get_disasms()`로 노출
- [ ] **GTX-VERIFY-01**: 기존 `verify.py` (DDR FP16 ULP/atol 비교) 자산을 pyspike에
  동봉하고 Python NPU 결과 ↔ C++ libgtx_npu.so 결과 비교 회귀 셋 통과
- [ ] **GTX-VERIFY-02**: `verify_ref.py`(32개 op host-side scalar 검증)을 Python
  NPU 단위 테스트로 흡수해 op별 ULP 일치 확인
- [ ] **GTX-FW-01**: 기존 펌웨어 회귀 (`run_tests_n1s16.sh`,
  `run_llext_tests.sh`에 대응되는 .elf 시퀀스)를 pyspike+Python NPU 환경에서 100% 통과
- [ ] **GTX-PKG-01**: `pip install spike` 후 한 줄 (`from riscv.gtx import GtxNpu`)
  으로 NPU 인스턴스 생성·실행 가능, 펌웨어/golden hex 자산도 wheel에 포함
- [ ] **GTX-RST-01**: `reset()` 시 `sp = 0x80100000` 초기화 + WJOIN 자동 종료
  (`GTX_NO_EXIT` 미설정 시) 등 호환 동작 유지
<!-- GTX-MEM-01 / GTX-REF-01 → Validated in Phase 1 (moved above) -->

### Out of Scope

<!-- v1에서 제외. 사유 명시 — 재추가 방지. -->

- **CUDA 가속 경로 (`gtx/cuda/*.cu`)** — Python 재작성 + NumPy 백엔드 결정으로
  GPU 가속이 의미 없어짐. wheel 배포 복잡도(드라이버/툴킷 의존)도 매우 높음
- **PCIe-EP / vfio-user 도관 (`gtx_pcie_ep.{cc,h}`, `gtx_vfio.{cc,h}`,
  libvfio-user 의존성)** — wheel 빌드/배포 복잡도가 NPU 핵심에 비해 과도하며,
  v1 사용 사례가 ISA/펌웨어 회귀에 집중됨. v2에서 Python `riscv.dev.MMIO`로
  재검토
- **GTX commitlog (`--enable-gtxcommitlog`) / GTX_QUIET 등 디버그 트레이스 빌드
  옵션** — 사용자가 명시 제외. 회귀/검증과 직교한 부가 기능
- **non-Linux / non-x86_64 플랫폼** — pyspike 자체가 manylinux2014_x86_64 베이스
  라인. Windows/macOS/aarch64는 v1 비대상
- **C++ libgtx_npu.so를 wheel에 동봉하기** — Python 재작성이 일차 산출물이므로
  C++ 바이너리는 `vendor/gtx_cpp_reference/`에 소스 스냅샷만 유지(개발 시 검증용,
  배포 wheel에는 미포함)

## Context

**기술 환경:**
- pyspike 코드베이스 분석은 이미 완료됨(`.planning/codebase/STACK.md`,
  `ARCHITECTURE.md`, `STRUCTURE.md`, `INTEGRATIONS.md`, `CONVENTIONS.md`,
  `TESTING.md`, `CONCERNS.md`). RoCC 트램폴린 표면이 이미 정리되어 있음
  (commit `c9cf7c4 docs: map RoCC extension surface in pyspike binding layer`)
- 기존 GTX RoCC 구현은 `~/NIGHTLY/gtx_spike/gtx/`에 11개 .cc + .h들로 분리,
  `gtx_npu_t : rocc_t` 단일 클래스가 모든 동작을 담당. CLAUDE.md에 메모리 계층/
  인코딩/주의사항이 잘 정리되어 있어 포팅 명세로 활용 가능
- 호환 인코딩 두 종 공존: gem5 간소화(funct7=0x04–0x07, GSPR에 operand 설정 후
  dispatch) + ISS full(funct7=0x00–0x7F, 연산별 고유 funct7)

**검증 환경:**
- C++ libgtx_npu.so는 SystemC HW sim / RTL과 ULP 내 일치 검증 완료(사용자 확인) →
  Python 포팅의 bit-exact 비교 대상은 **C++ libgtx_npu.so 출력**으로 단순화 가능
- DDR hex 바이트 순서 규약: 표준 LTR이 기본, `GTX_DDR_REVERSED=1`이면 RTL/SystemC
  호환 RTL. Python 포팅도 두 모드 모두 지원해야 회귀 통과 가능
- FP16 정밀도 규약: VSUM은 FP32 내부 누적 후 1회 FP16 변환, 행별 분할 시
  부분합 재합산 — 그대로 따라야 일치

**알려진 함정:**
- xs1=0이면 Spike가 rs1에 -1을 전달 → C++ 코드는 `p->get_state()->XPR[insn.rs1]`로
  우회. Python에서도 동일 우회 필요 (pybind11이 `processor_t*`/`rocc_insn_t`를 그대로 넘김)
- Activation 방향성: RELU/SOFTMAX/ESUM은 ADDRA→ADDRR, PRELU/GELU/TANH/SIGM은
  ADDRR→ADDRA로 역방향 — 헷갈리기 쉬운 지점
- WJOIN에서 `exit(0)` 호출 동작은 펌웨어 무한 루프 종료에 필수 → Python에서도
  동등한 종료 메커니즘 필요(SystemExit 등)

## Constraints

- **Tech stack**: Python **3.10+** / **NumPy ≥ 2.0** / pyspike의 pybind11 트램폴린.
  C++ 추가 코드 금지(순수 Python 재작성이라는 사용자 결정) — 성능 핫스팟이 발견되면
  v2에서 cython/C 확장 검토.
  **Phase 1 discuss-phase 결정 (D-07/D-08):** NumPy 2.x FP16 IEEE 754 binary16 RNE 시맨틱
  채택 + cp38/cp39 드롭으로 research 권장 (`>=1.20,<2.0`)을 번복함
- **Compatibility**: `riscv.isa.ROCC` 가상 메서드 시그니처(`custom0/1/2/3(proc, insn,
  xs1, xs2) -> reg_t`)를 정확히 따라야 함. processor_t/rocc_insn_t는 pybind11
  바인딩 객체 그대로 사용
- **Performance**: NumPy 백엔드 가정. NEST(4)×SPU(16)×L1(384KB) 메모리 표현은
  ndarray로, FP16 연산은 `np.float16` view (D-09; FP32 내부 누적은 reduction 시에만).
  회귀가 한 세션 내(≤ 수십 분 수준) 끝나야 실용
- **Dependencies**: NumPy 외부 추가 런타임 의존성 신규 도입 금지(wheel 배포 단순성).
  검증 단계에서만 기존 C++ libgtx_npu.so 참조(개발 환경)
- **Bit-exact**: ULP 허용오차 내(`verify.py --fp16 --ulp 1 --atol 0.001` 수준)
  C++ 결과와 일치 필수. 회귀 1개라도 깨지면 출하 보류
- **Testing**: pytest 기반(이미 구축됨). 신규 op마다 verify_ref.py 대응 단위 테스트
  + 적어도 1개의 .elf 회귀 통과 묶음 추가
- **Platform**: Linux x86_64 / glibc 2.17+ (manylinux2014).
  **cibuildwheel 매트릭스: cp310-cp312** (Phase 1 D-08; cp38/cp39 드롭됨)

## Key Decisions

| Decision | Rationale | Outcome |
|----------|-----------|---------|
| C++ libgtx_npu.so를 wheel에 동봉하지 않고 Python으로 재작성 | Python에서 ISA/op 변형 실험·해킹 용이성 우선. C++ 코드는 검증 레퍼런스로만 유지 | — Pending |
| NumPy를 메모리/연산 백엔드로 채택 | Pure Python은 너무 느려 펌웨어 회귀가 비실용. cython/C 추가는 wheel 복잡도↑. NumPy가 균형점 | — Pending |
| NumPy ≥ 2.0 + cp310-cp312 (cp38/cp39 드롭, D-07/D-08) | NumPy 2.x FP16 RNE 표준화·코드 단순. cibuildwheel 매트릭스 축소는 사용자 명시 결정 | ✓ Validated in Phase 1 (pyproject.toml 패치 완료) |
| `np.float16` view 채택 (D-09) — 순수 비트 변환 거부 | NumPy 2.x view가 IEEE 754 binary16 보장. P4/P5에서 strict 모드로 차이 측정 후 필요시 비트 fallback | ✓ Validated in Phase 1 (helper-level, 65536 round-trip + NaN 보존; op-level은 P4/P5 deferred) |
| C++ 레퍼런스 = git submodule (`https://github.com/Sudo42b/gtx_spike`, D-04) | 자동 업스트림 동기화. 공개 레포라 CI 익명 clone 가능 | ✓ Validated in Phase 1 (`vendor/gtx_cpp_reference` 등록, MANIFEST.in prune) |
| Bit-exact 비교 대상은 C++ libgtx_npu.so | C++가 SystemC와 이미 일치 검증 완료(사용자 확인) → SystemC 직접 접근 없이도 유효한 golden 확보 | — Pending |
| MM 서브시스템 우선 구현 | GEMM이 NPU 핵심 부하, 다른 op들이 간접적으로 의존(operand staging, mxe_accum). MM 동작이 가장 강한 진척 신호 | — Pending |
| PCIe-EP / vfio-user / CUDA 경로 v1 제외 | wheel 배포 복잡도가 v1 핵심 가치(ISA/펌웨어 회귀)와 비례하지 않음. v2에서 재평가 | — Pending |
| 기존 C++ gtx 소스 스냅샷을 `vendor/gtx_cpp_reference/`에 두기 | Ground-truth로 동시에 비교 baseline. 외부 경로(`~/NIGHTLY/gtx_spike/gtx/`) 의존 제거 | — Pending |

## Evolution

This document evolves at phase transitions and milestone boundaries.

**After each phase transition** (via `/gsd:transition`):
1. Requirements invalidated? → Move to Out of Scope with reason
2. Requirements validated? → Move to Validated with phase reference
3. New requirements emerged? → Add to Active
4. Decisions to log? → Add to Key Decisions
5. "What This Is" still accurate? → Update if drifted

**After each milestone** (via `/gsd:complete-milestone`):
1. Full review of all sections
2. Core Value check — still the right priority?
3. Audit Out of Scope — reasons still valid?
4. Update Context with current state

---
*Last updated: 2026-05-05 after Phase 3 complete (DMA & DDR I/O: 6 DMA exec_* helpers + DeferredDdrStore + 2-level custom0 dispatch + 4-mode router + DDR doubling-grow + GTX_DDR_REVERSED bit-exact round-trip; DMA-01..05 + DISP-03 closed; 179/179 P3 tests pass)*
