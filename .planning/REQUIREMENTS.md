# Requirements: pyspike + GTX NPU (Python RoCC Port)

**Defined:** 2026-05-04
**Core Value:** 기존 NPU 펌웨어(.elf) 회귀 테스트가 pyspike+Python NPU에서도
그대로 통과하고 DDR 결과가 C++ libgtx_npu.so(SystemC HW sim과 ULP 내 일치 검증
완료된 golden)와 ULP 허용오차 내로 일치한다.

## v1 Requirements

각 요구사항은 PROJECT.md `Active`와 `.planning/research/SUMMARY.md`의 6단계
페이즈 구조에 매핑된다. 카테고리는 연구 FEATURES.md의 8개 영역을 따름.

### Foundation (FP/Memory/Package skeleton)

- [x] **FOUND-01**: `fp16_to_fp32` / `fp32_to_fp16` 헬퍼를 `np.float16` view 기반으로 구현 (D-09)
  — NumPy 2.x IEEE 754 binary16 RNE 시맨틱 사용, 65536개 FP16 값 전수 round-trip 멱등성 검증
  (`f16 → f32 → f16 == f16`). C++ `gtx_npu.h:89-151`와의 strict 차이는 P4/P5에서 측정·대응
- [ ] **FOUND-02**: L0(1KB×SPU) / L1(384KB×SPU) / L2(16MB×NEST) / DDR을
  `np.uint8` 단일 ndarray + halfword view(`view(np.uint16)`/`view(np.float16)`)로
  표현, 모든 FP16 접근이 little-endian 바이트 순서 유지
- [x] **FOUND-03**: `src/main/python/riscv/gtx/` 패키지 스켈레톤(`__init__.py`,
  `params.py`, `encoding.py`, `fp.py`, `memory.py`) — wheel 동봉 가능한 import
  경로 확보
- [ ] **FOUND-04**: 기존 C++ gtx 소스 스냅샷을 `vendor/gtx_cpp_reference/`에
  복사해 검증 baseline + ground-truth로 영구 보관

### Core (ROCC subclass / Reset / WJOIN)

- [x] **CORE-01**: `riscv.isa.ROCC` 서브클래스 `GtxNpu` 작성 +
  `@riscv.isa.register("gtx")`로 자동 등록 — `pyspike --extlib=riscv.gtx`로 로드 가능
- [x] **CORE-02**: `reset()` 시 `XPR.write(2, 0x80100000)`로 sp 초기화 +
  `mxe_accum`/SPR/L0/L1/L2 영(zero) 초기화
- [x] **CORE-03**: WJOIN 시 `GTX_NO_EXIT` 환경변수 미설정이면 `SystemExit(0)`
  raise — 펌웨어 무한 루프 종료 메커니즘
- [x] **CORE-04**: `xs1=0` 우회 패턴 — `proc.get_state().XPR[insn.rs1]`로 직접
  레지스터 읽어 Spike의 -1 마샬링 회피.
  **Phase 2 구현 결정 (CONTEXT.md D-05):** 데코레이터로 자동 wrap —
  `(proc, insn, xs1, xs2)` 4-arg signature를 wrap하면서 xs1==0이면 GPR 직접 read로 교체.

### SPR (Special-Purpose Register)

- [x] **SPR-01**: GSPR(0x000–0x3FF) / NSPR(0x400–0x7FF) / LSPR(0x800–0xBFF)을
  Python `dict[int, int]`로 표현, `wr_spr` / `rd_spr` 라우팅 구현
- [x] **SPR-02**: WRSPR(funct7=0x00, gem5 simplified) / RDSPR writeback 경로
  완성 — RDSPR 결과가 GPR에 정확히 기록됨

### Disasm

- [x] **DISASM-01**: `gtx_npu_disasm.inc`(~140 entries)에 1:1 대응되는
  `disasm_insn_t` 리스트를 `get_disasms()`로 반환 — `pyspike` 트레이스에 정상 표시.
  **Phase 2 구현 결정 (CONTEXT.md D-09/D-10/D-13):** Per-op registry 패턴 —
  각 op 모듈이 자신의 `disasm_insn_t` 항목을 데코레이터(`@gtx.handler(...)`)로 동반 등록.
  `disasm.py`가 누적/조회 API 제공. 항목 수는 phase 진행에 따라 누적
  (P2: ~10 SPR/control, P3: +DMA, P4: +MM, P5: +VEC/ACT, 최종 ~140).

### Dispatch

- [x] **DISP-01**: `custom0()` funct7 디스패치 dict (gem5 simplified
  0x04–0x07 + ISS full 0x00–0x7F 양 인코딩 공존, funct7=0x00 충돌 시
  `insn.rs1 != 0` 휴리스틱으로 WRSPR / MM 분기).
  **Phase 2 구현 결정 (CONTEXT.md D-01/D-02):** 단일 dict-of-handlers
  `self._custom0_handlers: dict[int, Callable]`. funct7=0x00 충돌 시
  `if insn.rs1 != 0: WRSPR (gem5 marker), else: MM/no-op fallback`.
- [x] **DISP-02**: `custom1()` warp 루프 제어 (start/end P/S/T, split/join) +
  P/S/T 상태 머신 정확히 동작
- [x] **DISP-03**: 4-mode dispatch router (Mode 1: no loop / Mode 2: P only /
  Mode 3: P+S DMA / Mode 4: P+T compute) — NEST/SPU 라우팅 그룹이 C++과 일치

### DMA + DDR I/O

- [x] **DMA-01**: `exec_dma_2d`, `exec_load_svr`, `exec_store_svr`,
  `exec_transpose`, `exec_fill` 전체 ops 구현
- [x] **DMA-02**: `firmware_dma_op` 패킹 인코딩 디코딩 — funct3 = 000(LOAD) /
  001(STORE) / 010(COPY) 분기
- [ ] **DMA-03**: S-loop L2→DDR 스토어 deferred-store 큐, `endp`에서 일괄 flush —
  순서가 C++ 동작과 동일
- [x] **DMA-04**: DDR hex I/O 두 모드(`ddr_init_from_file`, `ddr_dump_to_file`)
  — 표준 LTR(기본) + `GTX_DDR_REVERSED=1` (256-bit 버스 워드 역순) 모두 동작
- [ ] **DMA-05**: DMA 라운드트립 테스트 — Python에서 L1에 패턴 쓰고 → DDR로 store
  → DDR에서 다시 load → 원본과 bit-exact 일치

### MM (Matrix Multiplication — NPU 핵심)

- [ ] **MM-01**: `gemm_core` — `np.matmul`을 `dtype=np.float32`로 호출하고 단일
  `np.float16` cast로 결과 산출
- [ ] **MM-02**: `exec_mm`, `exec_mm_s`, `exec_mm_o`, `exec_mm_v`, `exec_mm_t`,
  `exec_mmc`, `exec_mmc_s`, `exec_mmc_o`, `exec_mmc_v`, `exec_mmc_t` 전 변형 동작
- [ ] **MM-03**: `firmware_mm_op` 패킹 인코딩 디코딩 (`colB[63:48] | colA[31:16]
  | rowA[15:0]`, HW 컨벤션 0=65536) + funct3 변형(MM/MMC) 분기
- [ ] **MM-04**: `mxe_accum` per-(NEST,SPU) FP32 영구 상태 — `is_accumulate` 플래그
  존중, `mm.s → mmc.s → mmc` 체인이 C++과 동일하게 누적
- [ ] **MM-05**: 첫 .elf 펌웨어 회귀(GEMM 단순) 통과 — DDR dump가
  `verify.py --fp16 --ulp 1 --atol 0.001` 통과

### VEC (Vector ops)

- [ ] **VEC-01**: SASMD (add/sub/mul/div) 4종 × IS/VS variant 구현 (funct7=0x10)
- [ ] **VEC-02**: DOT / VSUM 리덕션 — VSUM은 FP32 내부 누적 후 단일 FP16 cast (행별
  분할 시 FP16 부분합 재합산 규약 준수)
- [ ] **VEC-03**: CLAMP (min/max/arange/accum), L1(VV) / L0(II) 분기 (funct7=0x18–0x1F)
- [ ] **VEC-04**: VEC scalar / immediate variants (`exec_vec_scalar`,
  `_imm`)
- [ ] **VEC-05**: `firmware_vec_op` 패킹 인코딩 디코딩

### ACT (Activation / Pooling / Format)

- [ ] **ACT-01**: 정방향 활성화 (RELU/SOFTMAX/ESUM): ADDRA → ADDRR
- [ ] **ACT-02**: 역방향 활성화 (PRELU/GELU/TANH/SIGM): ADDRR → ADDRA — 비대칭
  방향 테이블 명시적으로 구현
- [ ] **ACT-03**: Pooling (`exec_pooling`) 전체
- [ ] **ACT-04**: Format conversion (`exec_format_cvt`) — FP16/FP32/FP8 + scale/offset
- [ ] **ACT-05**: `_imm` 변형 활성화 (L0 경로, funct7=0x28/0x2A/0x2C/0x2D & 4)

### Verification (Op-level + .elf regression)

- [ ] **VRF-01**: `verify.py`(388 LOC, FP16 ULP/atol diff) → `riscv.gtx._verify`로
  포팅 + 모듈로 import 가능
- [ ] **VRF-02**: `verify_ref.py` 32개 host-side scalar oracle을 pytest 단위 테스트로
  변환 — 각 op이 oracle과 ULP 1 내 일치
- [ ] **VRF-03**: `tests/gtx/data/{golden,elf}/`에 회귀 .elf 펌웨어 + golden DDR hex
  자산 동봉 (기존 `run_tests_n1s16.sh` / `run_llext_tests.sh` 시퀀스 대응)
- [ ] **VRF-04**: 회귀 `.elf` 100% 통과(strict mode: `exact_matches == total_fp16`)
  — gem5 simplified 인코딩 펌웨어 + ISS full 인코딩 펌웨어 양 스위트 모두

### Distribution (Wheel packaging)

- [ ] **PKG-01**: `pyproject.toml` `[tool.setuptools.package-data]`에 `riscv.gtx.data/`
  추가 — `.elf` / `.hex` 자산이 wheel에 포함
- [x] **PKG-02**: NumPy 의존성 `numpy>=2.0,<3` 추가 (D-07).
  cp310+가 베이스라인이므로 `importlib.resources`는 stdlib 사용, 백포트 불필요.
  `requires-python = ">=3.10"`로 변경 (D-08)
- [ ] **PKG-03**: `pip install spike` 후 한 줄(`from riscv.gtx import GtxNpu`)로
  NPU 인스턴스 생성·실행 가능
- [ ] **PKG-04**: cibuildwheel manylinux2014_x86_64 매트릭스 (**cp310–cp312**, D-08)
  통과 — `[tool.cibuildwheel].build`에서 cp38/cp39 라인 제거, wheel 빌드 깨짐 없음

## v2 Requirements

향후 마일스톤. 현재 로드맵에 미포함.

### Cycle/Performance

- **CYC-01**: 사이클 카운팅 모델 (`-DGTX_FUNCTIONAL_ONLY` 미설정 시 동작) — 현재는
  count return 0 또는 상수
- **CYC-02**: 뱅크 충돌 모델링 / TLB / credit 시스템 정확화

### mexec / DMA 확장

- **MEXEC-01**: mexec full microcode 페치-디코드 루프 (DDR에서 마이크로코드
  fetch + decode) — v1에서는 회귀가 트리거하지 않으면 stub
- **DMA-V2-01**: DMA-LOAD-3D / IM2COL-N / IM2COL-D / MCAST — 회귀 .elf가
  요구하면 v1으로 승격

### Python 차별화 기능

- **PY-OVRD-01**: per-op `before_<op>` / `after_<op>` 후크 — 라이브 수치 실험
- **PY-FUNCT7-01**: `gtx.register_funct7(0x7E, my_handler)` — Python에서 사용자
  정의 funct7 등록(ISA 실험)
- **PY-VIEW-01**: `gtx.l1_view(nest, spu, dtype=np.float16)` — 외부에서 NumPy view
  반환
- **PY-TRACE-01**: 런타임 `gtx.enable_trace()` — `--enable-gtxcommitlog` 빌드 플래그
  대체

### MMIO / Device

- **DEV-01**: PCIe-EP 핵심을 `riscv.dev.MMIO` 서브클래스로 재구현 (외부 vfio
  소켓 없이) — 호스트 모델링 용도
- **SNAP-01**: NPU 상태 스냅샷/복원 (replay debugging용)

## Out of Scope

명시 제외. 재추가 방지를 위한 사유 명시.

| Feature | Reason |
|---------|--------|
| CUDA 가속 경로 (`gtx/cuda/*.cu`) | Python 재작성 + NumPy 백엔드 결정으로 GPU 가속 의미 없음. wheel 배포 복잡도(드라이버/툴킷 의존) 매우 높음 |
| libvfio-user 외부 소켓 / vfio-user 어댑터 | 외부 의존성 + 외부 소켓 도관, wheel 배포 복잡도가 v1 핵심(ISA/펌웨어 회귀)에 과도. v2에서 재검토 |
| GTX commitlog (`--enable-gtxcommitlog`) | 사용자 명시 제외. 회귀/검증과 직교한 부가 기능. v2에서 Python `gtx.enable_trace()`로 대체 가능 |
| GTX_QUIET 등 디버그 트레이스 빌드 옵션 | 사용자 명시 제외. Python `enable_trace()`로 대체 |
| non-Linux / non-x86_64 플랫폼 | pyspike 자체가 manylinux2014_x86_64 베이스. Windows/macOS/aarch64 v1 비대상 |
| C++ libgtx_npu.so를 wheel에 동봉 | Python 재작성이 일차 산출물. C++ 바이너리는 `vendor/gtx_cpp_reference/` 소스 스냅샷만 (개발 시 검증용) |
| 온라인 shadow run vs C++ libgtx_npu.so | 검증은 오프라인 golden hex diff로만 수행 — wheel 의존성 단순성 우선 |
| numba / cython / JAX / torch / scipy | NumPy 단독으로 회귀 시간 예산 충족. 추가 시 wheel 빌드 복잡도/사이즈 폭증 |
| `match` statement (PEP 634) — 사용 검토 가능 | ~~PROJECT.md 호환 베이스라인 3.8~~ → Phase 1 D-08로 cp310+ 베이스라인 됨, `match` 사용 가능. 그러나 dict-of-handlers가 이미 디스패치 표준 — 일관성 유지를 위해 신규 코드도 dict-of-handlers 사용 권장 |
| GELU_ERF (scipy.special.erf 의존) | scipy 의존성 회피. 회귀가 요구하면 NumPy 시리즈 근사로 대체 |

## Traceability

페이즈 매핑은 ROADMAP.md (2026-05-04 작성)에 따라 `gsd-roadmapper`가 채움.

| Requirement | Phase | Status |
|-------------|-------|--------|
| FOUND-01 | Phase 1 | Complete |
| FOUND-02 | Phase 1 | Pending |
| FOUND-03 | Phase 1 | Complete |
| FOUND-04 | Phase 1 | Pending |
| CORE-01 | Phase 2 | Complete |
| CORE-02 | Phase 2 | Complete |
| CORE-03 | Phase 2 | Complete |
| CORE-04 | Phase 2 | Complete |
| SPR-01 | Phase 2 | Complete |
| SPR-02 | Phase 2 | Complete |
| DISASM-01 | Phase 2 | Complete |
| DISP-01 | Phase 2 | Complete |
| DISP-02 | Phase 2 | Complete |
| DISP-03 | Phase 3 | Complete |
| DMA-01 | Phase 3 | Complete |
| DMA-02 | Phase 3 | Complete |
| DMA-03 | Phase 3 | Pending |
| DMA-04 | Phase 3 | Complete |
| DMA-05 | Phase 3 | Pending |
| MM-01 | Phase 4 | Pending |
| MM-02 | Phase 4 | Pending |
| MM-03 | Phase 4 | Pending |
| MM-04 | Phase 4 | Pending |
| MM-05 | Phase 4 | Pending |
| VEC-01 | Phase 5 | Pending |
| VEC-02 | Phase 5 | Pending |
| VEC-03 | Phase 5 | Pending |
| VEC-04 | Phase 5 | Pending |
| VEC-05 | Phase 5 | Pending |
| ACT-01 | Phase 5 | Pending |
| ACT-02 | Phase 5 | Pending |
| ACT-03 | Phase 5 | Pending |
| ACT-04 | Phase 5 | Pending |
| ACT-05 | Phase 5 | Pending |
| VRF-01 | Phase 6 | Pending |
| VRF-02 | Phase 5 | Pending |
| VRF-03 | Phase 6 | Pending |
| VRF-04 | Phase 6 | Pending |
| PKG-01 | Phase 6 | Pending |
| PKG-02 | Phase 1 | Complete |
| PKG-03 | Phase 6 | Pending |
| PKG-04 | Phase 6 | Pending |

**Coverage:**
- v1 requirements: 42 total
- Mapped to phases: 42 ✓
- Unmapped: 0 ✓ (100% coverage)

**Phase distribution:**
- Phase 1 (Foundation): 5 (FOUND-01..04, PKG-02)
- Phase 2 (Skeleton & Disasm): 9 (CORE-01..04, SPR-01, SPR-02, DISASM-01, DISP-01, DISP-02)
- Phase 3 (DMA & DDR I/O): 6 (DMA-01..05, DISP-03)
- Phase 4 (MM Subsystem): 5 (MM-01..05)
- Phase 5 (VEC/ACT/Pool): 11 (VEC-01..05, ACT-01..05, VRF-02)
- Phase 6 (Verification & Wheel): 6 (VRF-01, VRF-03, VRF-04, PKG-01, PKG-03, PKG-04)

---
*Requirements defined: 2026-05-04*
*Phase mappings filled: 2026-05-04 by gsd-roadmapper*
*Last updated: 2026-05-04 after Phase 1 discuss (FOUND-01/PKG-02/PKG-04 NumPy 2.x + cp310 pivot, D-07/D-08/D-09)*
