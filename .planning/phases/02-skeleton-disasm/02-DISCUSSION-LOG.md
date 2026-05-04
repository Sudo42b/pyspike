# Phase 2: Skeleton & Disasm - Discussion Log

> **Audit trail only.** Do not use as input to planning, research, or execution agents.
> Decisions are captured in CONTEXT.md — this log preserves the alternatives considered.

**Date:** 2026-05-04
**Phase:** 02-skeleton-disasm
**Areas discussed:** Dispatch 테이블 구조 + 충돌 해결, Disasm 테이블 출처 (DISASM-01), `_riscv.so` 해결 경로, Test 전략, GtxNpu 내부 모듈 경계, P2-P5 op handler API, ROADMAP/REQUIREMENTS 수정 범위, Mock processor_t/rocc_insn_t 자세

---

## Initial Selection

| Option | Description | Selected |
|--------|-------------|----------|
| Dispatch 테이블 구조 + 충돌 해결 | DISP-01 funct7=0x00 충돌 해결, dict-of-handlers vs gem5/ISS 계층 | ✓ |
| Disasm 테이블 출처 (DISASM-01) | Hand-port vs build-time vs per-op registry | ✓ |
| `_riscv.so` 해결 경로 | pybind11 3.0.4 csr_t 충돌 해결 단계 | ✓ |
| Test 전략 (로컬에서 _riscv 부재) | Mock vs skipif vs hybrid | ✓ |

---

## Dispatch 테이블 구조

| Option | Description | Selected |
|--------|-------------|----------|
| 단일 dict-of-handlers (권장) | self._funct7_handlers: dict[int, Callable] 하나, xhuimt/mylrsc.py 패턴 일치 | ✓ |
| 계층화 (gem5 router → ISS router) | _dispatch_gem5(funct7) if 0x04<=funct7<=0x07 else _dispatch_iss(funct7) | |
| Per-op 모듈이 funct7 등록 (registry) | @register_funct7(0x49) 데코레이터 + GtxNpu 추가 시 collect | |

**User's choice:** 단일 dict-of-handlers
**Notes:** xhuimt/mylrsc.py 패턴과 일치, C++ switch-case의 직역. 충돌 시 분기 메서드 하나가 rs1을 읽고 다시 라우팅.

---

## funct7=0x00 충돌 디스패치 휴리스틱 디테일

| Option | Description | Selected |
|--------|-------------|----------|
| DISP-01 명시: insn.rs1 != 0 → WRSPR (gem5 marker), 그 외 → MM/no-op fallback | REQUIREMENTS DISP-01 단어 그대로 | ✓ |
| xs1==1 AND xs2==1 모두 체크 (엄격) | rocc_insn_t.xs1/xs2 둘 다 1이어야 WRSPR로 판정 | |
| xs1=0 우회와 함께 보다 안전한 xs1==1 AND xd==0 체크 | WRSPR은 destination register 없음 (xd=0) | |

**User's choice:** DISP-01 명시 그대로
**Notes:** REQUIREMENTS 단어 그대로 따름. C++ ground-truth와 plan 단계에서 일치 검증.

---

## WRSPR / RDSPR 레지스터 주소 어디서 추출?

| Option | Description | Selected |
|--------|-------------|----------|
| xs1가 SPR 주소, xs2가 값 (gem5) | gem5 WRSPR 마커 표준 | |
| ISS full 명시 올바른 세맨틱 확인 후 논의 | ISS full에서 WRSPR funct7=0x49 | |
| Claude 재량 | C++ ground-truth를 plan 단계에서 직접 참조 | ✓ |

**User's choice:** Claude 재량
**Notes:** plan 단계에서 vendor/gtx_cpp_reference/gtx/gtx_npu_spr.cc 직접 읽고 일치.

---

## WJOIN SystemExit 캡처 테스트 패턴

| Option | Description | Selected |
|--------|-------------|----------|
| pytest.raises(SystemExit) + GTX_NO_EXIT=1 fixture로 양 모드 둘 다 유닛 검증 | monkeypatch.setenv 활용 | ✓ |
| GTX_NO_EXIT=1 경로만 유닛 테스트, SystemExit 경로는 CLI 통합 테스트에만 | 유닛에서 SystemExit 제어가 부수효과 우려 | |

**User's choice:** 양 모드 둘 다 유닛 검증
**Notes:** pytest.raises는 안전, 환경 격리 fixture로 충분.

---

## custom1 warp 루프 상태 머신 레이아웃 (DISP-02)

| Option | Description | Selected |
|--------|-------------|----------|
| Plain attribute on GtxNpu (single warp) | self.is_ploop, self.is_tloop bool | |
| WarpState dataclass | @dataclass class WarpState | ✓ |
| Per-NEST list (병렬 warp) | self.warps: list[WarpState] = [WarpState() for _ in range(NEST_NUM)] | |

**User's choice:** WarpState dataclass
**Notes:** P3+에서 is_sloop 필드 추가. 명시적, mutable, pytest assertion 단순.

---

## mxe_accum 레이아웃 P2에서 조기 잠금?

| Option | Description | Selected |
|--------|-------------|----------|
| P2에서 shape/dtype 잠금, reset()에서 제로만 (권장) | self._mxe_accum: np.ndarray (shape=(NEST,SPU,M_TILE,N_TILE), dtype=np.float32) | ✓ |
| P2는 필드만 두고 (size=0), P4에서 shape 결정 | self._mxe_accum: np.ndarray | None = None | |
| P2에서 완전히 미포함 (아예 P4 일) | reset()이 특별한 필드 안 초기화 | |

**User's choice:** P2 shape/dtype 잠금
**Notes:** P4 결정 단순화. M_TILE/N_TILE 정확한 값은 plan 단계에서 C++ ground-truth 참조.

---

## xs1=0 우회 패턴(CORE-04) 적용 위치

| Option | Description | Selected |
|--------|-------------|----------|
| Helper 함수 _read_rs1(proc, insn) 만들어 모든 핸들러에서 사용 (권장) | _helpers.py에 명시적 호출 | |
| Mixin 클래스로 GtxNpu가 상속 | class _XPRReadMixin: def _read_rs1(...) | |
| 데코레이터로 자동 wrap | @xs1_safe def custom0(...) | ✓ |

**User's choice:** 데코레이터로 자동 wrap
**Notes:** 호출 사이트 코드 최대 단순. 구현 디테일은 plan 단계 — 4-arg signature wrap, xs2도 동시 처리. Hot path 측정 후 helper 전환 고려는 P4-P5에서.

---

## custom1 funct3 디스패치 구조

| Option | Description | Selected |
|--------|-------------|----------|
| custom0과 동일하게 dict-of-handlers (funct3 키) | self._custom1_handlers: dict[int, Callable] | ✓ |
| Direct if/elif 체인 | match/case 사용 가능 (cp310+) 하지만 out-of-scope 명시 | |

**User's choice:** dict-of-handlers
**Notes:** custom0와 통일. C++ funct3 매핑 plan 단계에서 확정.

---

## ~140 disasm 항목을 어떻게 생산?

| Option | Description | Selected |
|--------|-------------|----------|
| Hand-port → Python 리스트 리터럴 | src/main/python/riscv/gtx/disasm.py에 ~140 disasm_insn_t(...) 직접 나열 | |
| Build-time 변환 스크립트 (.inc → 자동 생성) | scripts/gen_disasm.py가 setup.py 호출 | |
| Per-op 핸들러가 disasm 동반 (registry 패턴) | P4-P5 op이 자신의 disasm 항목 동시 등록 | ✓ |

**User's choice:** Per-op registry
**Notes:** P2에서는 SPR/control 관련 ~10개만, P3+에서 점진 추가. **상위 영향:** P2 ROADMAP success criterion 2(~140개) 수정 필요.

---

## disasm_table.py 파일 소재지

| Option | Description | Selected |
|--------|-------------|----------|
| src/main/python/riscv/gtx/disasm.py | 명확한 그룹, encoding.py와 분리 | ✓ |
| src/main/python/riscv/gtx/encoding.py에 머지 | 단일 ISA 명세 source-of-truth | |
| src/main/python/riscv/gtx/ops/_disasm.py | ops/ 하위 통합 일관성 | |

**User's choice:** src/main/python/riscv/gtx/disasm.py
**Notes:** encoding.py(상수)와 분리. Per-op registry가 등록한 항목을 누적/조회.

---

## Sample 5 테스트 (mm/mm_s/mm_t/dma_load/wsplit) 테스트 대상

| Option | Description | Selected |
|--------|-------------|----------|
| test_disasm_table.py: get_disasms() 길이 + sample 5 mnemonics 확인 | Mock 또는 실제 사용 | ✓ |
| test_disasm_table.py + spike trace 실제 고도 (CLI integration) | _riscv 필요, skipif | |

**User's choice:** test_disasm_table.py 유닛
**Notes:** 단, Sample 5 중 'mm', 'mm_s', 'mm_t'는 P4 op이므로 P2에서는 skipif/xfail. 또는 P2 sample을 ['wrspr', 'rdspr', 'wsplit', 'wjoin', 'start_p']로 축소 (CONTEXT.md D-12).

---

## encoding.py(P1)에 disasm_table 관련 상수(funct7) 추가 포함

| Option | Description | Selected |
|--------|-------------|----------|
| P2에서 encoding.py 증축 (funct7 더 추가 + funct3 + mode 상수) | 단일 source-of-truth | ✓ |
| encoding.py 그대로 두고 disasm.py에 자체 상수 도입 | 이중 정의 위험 | |

**User's choice:** encoding.py 증축
**Notes:** P1의 funct7 stub 8개 → P2에서 gem5 0x04-0x07 + ISS 전체 0x00-0x7F + funct3 (custom1) + Mode 1-4.

---

## P2 종단 CLI 검증을 위한 _riscv.so 해결 경로

| Option | Description | Selected |
|--------|-------------|----------|
| Pin pybind11<3.0.4 in [build-system].requires (config-only, 권장) | pyproject.toml만 수정, CLAUDE.md 'no new C++ code' 완전 준수 | ✓ |
| Lambda 래퍼로 py_module.cc 1-2줄 수정 | CLAUDE.md no-C++ 완화 | |
| CI cibuildwheel에 전적 위임, 로컬은 _riscv 없이 검증 | 로컬 pytest는 _riscv 부재로 skipif | |

**User's choice:** pybind11<3.0.4 pin
**Notes:** CLAUDE.md 'no new C++ code' 완전 준수. cibuildwheel도 이 pin을 따름.

---

## submodule init 이슈 (vendor/gtx_cpp_reference SHA 재적용 실패)

| Option | Description | Selected |
|--------|-------------|----------|
| URL/SHA 재검증 후 재등록 (P2 첫 작업으로) | git config -f, git ls-remote 확인, 필요시 git submodule sync + git submodule update | ✓ |
| submodule 보류, 포팅 작업은 C++ 파일 직접 참조 (환경 외부) | submodule 없이 외부 경로 fallback | |

**User's choice:** URL/SHA 재검증 + 재등록 (P2 첫 작업)
**Notes:** disasm.py 작성 시 ground-truth 직접 참조 가능해야 함.

---

## P2 유닛 테스트가 _riscv 없이 돌아가도록 어떻게?

| Option | Description | Selected |
|--------|-------------|----------|
| Hybrid: _riscv 있으면 실제 사용, 없으면 모킹 (권장) | tests/gtx/conftest.py try/except | ✓ |
| Pure-mock: 항상 모킹 사용 | 모든 유닛 테스트가 모킹 사용 | |
| skipif-only: _riscv 없으면 스킵 | 로컬에서 P2 유닛도 일부면 skip | |

**User's choice:** Hybrid
**Notes:** 양 환경 모두 같은 테스트 코드. mock 사용 여부는 import 시점에 결정.

---

## P1 conftest.py 충돌 처리 (tests/conftest.py가 _riscv 강제 import)

| Option | Description | Selected |
|--------|-------------|----------|
| tests/conftest.py 이존 try/except로 감싸기 (다른 P2 plan에 포함) | conftest 재활성화 | ✓ |
| tests/gtx/conftest.py 독립적으로 구성, tests/conftest.py 손대지 않음 | 더 보수적, --noconftest 유지 | |

**User's choice:** tests/conftest.py try/except 가공
**Notes:** P2 별도 plan으로 처리.

---

## nop_wjoin.elf 테스트 픽스처 제공

| Option | Description | Selected |
|--------|-------------|----------|
| 프리빌드 바이너리를 tests/gtx/data/elf/nop_wjoin.elf로 커밋 (권장) | RISC-V cross-toolchain 사전 빌드, ~1KB | ✓ |
| 테스트 fixture에서 riscv-gcc로 원장 렌더링 | subprocess.run([..., 'gcc']) at fixture time | |
| 어셈블리 리터럴 또는 바이트 리터럴로 인라인 | Python 코드로 ELF 직접 조립 | |

**User's choice:** 프리빌드 커밋
**Notes:** 재현성 + 테스트 시간. 소스(`nop_wjoin.S`)도 같은 디렉토리에 commit.

---

## test_warp.py에서 warp 루프 상태 머신 어떻게 계측?

| Option | Description | Selected |
|--------|-------------|----------|
| Direct npu.custom1(proc, insn, xs1, xs2) 호출 다음 WarpState 필드 고결 (권장) | Mock proc 또는 실제 사용. funct3=000(start_p) 호출 → assert npu.warp.is_ploop == True | ✓ |
| Property 쿠어리만 (npu.is_in_warp(['p','t'])) | API 변경 가능성 | |

**User's choice:** Direct custom1 호출 + WarpState 필드 assertion
**Notes:** test_spr.py와 동일 패턴.

---

## GtxNpu 내부 모듈 경계

| Option | Description | Selected |
|--------|-------------|----------|
| 단일 npu.py 클래스 (모든 디스패치/상태/SPR 라우팅) | C++ gtx_npu_t 구조와 일치 | |
| 계층 분리 (권장) | npu.py (shell) + dispatch.py + warp_state.py + spr_router.py | ✓ |
| Claude 재량 (plan 단계에서) | gsd-planner가 구체 파일 수 결정 | |

**User's choice:** 계층 분리
**Notes:** 각 책임 명확. 테스트 용이.

---

## Per-op registry 프로토콜 (P3+ op module API)

| Option | Description | Selected |
|--------|-------------|----------|
| Module-level register: ops/spr.py.register(gtx) | 명시적 순서, GtxNpu.__init__()에서 호출 | |
| Decorator-based: @gtx.handler(0x49, mnemonic='wrspr', mask=...) | 명시적, 함수 정의 옆 메타데이터, import time 부작용 | ✓ |
| Manifest-based: HANDLERS dict, DISASM list 명시 선언 | 단순 데이터, decorator 없음 | |

**User's choice:** Decorator-based
**Notes:** 명시적, 함수 정의 옆 메타데이터. 내부 API (PY-FUNCT7-01 v2와 별개).

---

## ROADMAP/REQUIREMENTS 조정 범위

| Option | Description | Selected |
|--------|-------------|----------|
| P2 plan 내에서 한 태스크로 일괄 조정 (권장) | plan-phase에서 명시 단계로 포함 | |
| discuss-phase 종료 직후 별도 커밋 | CONTEXT.md 커밋과 동시에 ROADMAP.md도 쓰임 | ✓ |

**User's choice:** discuss-phase 종료 직후 별도 커밋
**Notes:** 다른 phases 읽을 때 이미 올바른 명세 보게.

---

## Mock processor_t / rocc_insn_t 속성 목록

| Option | Description | Selected |
|--------|-------------|----------|
| 최소세트: get_state().XPR (read/write), insn fields, MMU.load/store_uint64 | P2부터 ELF 로드까지 다양 | |
| P2 최소: XPR + insn fields만, MMU는 P3 시점 추가 | SPR/warp만 테스트 | ✓ |
| Per-test 목표 모킹 (그때그때 fixture로) | 유연성 ↑, 일관성 ↓ | |

**User's choice:** P2 최소
**Notes:** SPR/warp만 테스트. CLI는 _riscv 있을 때 skipif.

---

## Mock 클래스 공개 범위

| Option | Description | Selected |
|--------|-------------|----------|
| tests/gtx/_mocks.py 내부만, P3-P6은 경로로 import (권장) | production wheel에 미포함 | ✓ |
| src/main/python/riscv/gtx/_test_helpers.py에 공개 | wheel에도 포함, public API 오염 우려 | |

**User's choice:** tests/gtx/_mocks.py 내부만
**Notes:** production wheel 사이즈 절약, public API 오염 회피.

---

## @isa.register('gtx') 동작 검증 방법

| Option | Description | Selected |
|--------|-------------|----------|
| test_register.py: PYSPIKE_LIBS=riscv.gtx _riscv import 의존 — _riscv 있을 때만 skipif | _riscv 없이는 검증 불가 | |
| test_register.py + isa.ROCC 설계 계약 테스트 (래키 공점 방지) | mock 환경에서 issubclass + name + 메서드 확인, _riscv 환경에서 register_extension 동작 검증 | ✓ |

**User's choice:** 설계 계약 + skipif 통합
**Notes:** mock 환경에서도 부분 검증, _riscv 있을 때 종단 검증.

---

## ELF 패키징 레이아웃 (PKG-01 초석)

| Option | Description | Selected |
|--------|-------------|----------|
| tests/gtx/data/elf/nop_wjoin.elf 커밋, P2에서는 테스트 전용 | PKG-01은 P5/P6 일 | ✓ |
| tests/gtx/data/elf/ + src/main/python/riscv/gtx/data/elf/ 자동 심볼링 | P2에서 PKG-01 액세서리 절반 | |

**User's choice:** 테스트 전용 커밋
**Notes:** PKG-01 (P5/P6) 시점에 wheel 포함.

---

## Claude's Discretion

다음은 사용자가 "Claude 재량"으로 명시했거나 plan 단계에서 결정:
- WRSPR/RDSPR rs1/rs2 정확한 의미 (D-03)
- 데코레이터 정확한 구현 (D-05/D-13)
- `mxe_accum` shape의 M_TILE/N_TILE 값 (D-06)
- Mock 클래스 정확한 메서드 시그니처 (D-19)
- ELF 빌드 스크립트 형태
- `test_disasm_table.py` 샘플 5개 (P2 op 한정)

## Deferred Ideas

- discuss-phase 직후 별도 커밋으로 ROADMAP.md/REQUIREMENTS.md 동기화 (D-09 참조)
- P3-P6 op handler API (decorator-based registry 프로토콜 D-13)
- PKG-01 ELF wheel 포함 (P5/P6)
- VRF-01..04 verify.py 포팅 (P6)
- v2: PY-FUNCT7-01 (외부 API), CYC-01/02 (사이클 카운팅), MEXEC-01
