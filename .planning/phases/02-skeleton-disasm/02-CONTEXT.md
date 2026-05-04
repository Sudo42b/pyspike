# Phase 2: Skeleton & Disasm - Context

**Gathered:** 2026-05-04
**Status:** Ready for planning

<domain>
## Phase Boundary

Phase 2는 GTX NPU의 **dispatch/control shell**을 구축한다. 구체적으로:

1. **`GtxNpu` 클래스** (`riscv.isa.ROCC` 서브클래스, `@isa.register("gtx")`로 자동 등록) — `pyspike --extlib=riscv.gtx`로 NOP 펌웨어 로드 가능
2. **`reset()` 시 sp 초기화** + `mxe_accum`/SPR/L0/L1/L2 zero-init (CORE-02)
3. **WJOIN SystemExit 메커니즘** + `GTX_NO_EXIT` 환경변수 분기 (CORE-03)
4. **xs1=0 우회 패턴** — Spike의 -1 마샬링을 `proc.get_state().XPR[insn.rs1]`로 회피 (CORE-04)
5. **SPR 라우팅** — `wr_spr/rd_spr` (GSPR/NSPR/LSPR 0x000-0xBFF), WRSPR/RDSPR 호출 경로 (SPR-01/02)
6. **disasm.py 진입점** (DISASM-01) — per-op registry 프레임워크 + P2에서 ~10 SPR/control 항목 등록
7. **custom0/custom1 디스패치 셸** (DISP-01/02) — funct7/funct3 dict-of-handlers + warp 루프 상태 머신

다음 모두는 **Phase 2 비범위(out-of-scope)** — 다른 페이즈가 다룬다:
- DMA op 핸들러 + DDR hex I/O (Phase 3)
- 4-mode dispatch router 풀 활성화 (DISP-03 P3+)
- `gemm_core` / MM op 변형 (Phase 4)
- VEC/ACT/Pool op 핸들러 (Phase 5)
- `verify.py` 포팅 (Phase 6)
- ELF/golden hex 자산 패키지 데이터 등록 (PKG-01, P5/P6)

</domain>

<decisions>
## Implementation Decisions

### Dispatch 테이블 구조

- **D-01:** **단일 dict-of-handlers** — `self._custom0_handlers: dict[int, Callable]` (funct7 키), `self._custom1_handlers: dict[int, Callable]` (funct3 키). xhuimt/mylrsc.py 패턴 일치, C++ switch-case의 Python 직역.
  - **이유:** 일관성, 타입 명료성, P3-P5에서 `npu.add_funct7(0x49, handler)` 한 줄로 op 추가 가능.
- **D-02:** **funct7=0x00 충돌 휴리스틱 = `insn.rs1 != 0` → WRSPR (gem5 marker), 그 외 → MM/no-op fallback** (DISP-01 명시).
  - **이유:** REQUIREMENTS DISP-01 단어 그대로. gem5 simplified WRSPR은 `xs1=xs2=1` 마커이므로 rs1!=0이 1인 GPR 인덱스를 가리킴 (보통 x1=ra). C++ 동작과 일치.
- **D-03:** **WRSPR/RDSPR 레지스터 주소 추출은 Claude 재량** (plan 단계에서 C++ ground-truth 직접 확인하고 일치시킴).
  - **이유:** ISS-full WRSPR funct7=0x49의 정확한 rs1/rs2 의미가 REQUIREMENTS에 명시되지 않음. C++ `gtx_npu_spr.cc`를 plan 시점에 읽고 결정.

### Warp 루프 상태 머신 + xs1=0 우회

- **D-04:** **WarpState dataclass** — `@dataclass class WarpState: is_ploop: bool = False; is_tloop: bool = False; is_sloop: bool = False`. Phase 2는 `is_ploop`/`is_tloop`만 사용, P3+ DMA에서 `is_sloop` 활성화.
  - **이유:** 명시적, mutable, pytest assertion 단순. C++ `gtx_npu_t`의 비트 플래그를 데이터클래스로 직역.
- **D-05:** **xs1=0 우회 = 데코레이터로 자동 wrap** (CORE-04). 모든 `custom0`/`custom1` 핸들러가 데코레이터에 의해 wrap되어 자동으로 GPR 직접 read하도록.
  - **이유:** 호출 사이트 코드 단순 — handler signature는 깨끗하게 `(proc, insn, xs1_safe, xs2_safe)`처럼 표시.
  - **구현 디테일은 plan 단계** — `(proc, insn, xs1, xs2)` 4-arg signature를 wrap하면서 xs1==0이면 `proc.get_state().XPR[insn.rs1]`로 교체.
  - **위험:** 데코레이터로 매번 GPR read는 약간의 오버헤드. P4-P5 hot path에서 측정 후 helper 함수로 전환 가능 — Claude 재량.
- **D-06:** **`mxe_accum` 레이아웃 P2에서 조기 잠금** — `self._mxe_accum: np.ndarray` 필드 채움 (shape=(NEST_NUM, SPU_NUM, M_TILE, N_TILE), dtype=np.float32). P4 MM-04에서 read/write. Reset()에서 `self._mxe_accum.fill(0.0)`.
  - **이유:** P2에서 데이터 레이아웃 잠금으로 P4 결정 단순화. M_TILE/N_TILE 정확한 값은 C++ 참조에서 추출 (plan 단계).

### WJOIN / SystemExit

- **D-07:** **WJOIN 시 `GTX_NO_EXIT` 환경변수 매번 read** (cached 안 함). `os.environ.get('GTX_NO_EXIT')`이 unset/falsy → `raise SystemExit(0)`. set/truthy → return 0.
  - **이유:** 테스트 친화성 — `monkeypatch.setenv('GTX_NO_EXIT', '1')` fixture로 동작 변경 가능. 성능 영향 미미 (WJOIN은 펌웨어당 1-2번).
- **D-08:** **WJOIN 단위 테스트는 양 모드 모두 유닛 검증** — `pytest.raises(SystemExit) as exc_info: npu.custom1(...)` (default mode) AND `monkeypatch.setenv('GTX_NO_EXIT', '1'); ret = npu.custom1(...); assert ret == 0`.

### Disasm 테이블 (DISASM-01)

- **D-09:** **Per-op registry 패턴** — 각 op 모듈(`ops/spr.py`, `ops/dma.py`, ...)이 자신의 funct7 핸들러 + disasm 항목 동반 제공. P2는 SPR/control 관련 ~10개만, P3+에서 점진 늘림.
  - **이유:** 모듈 응집도 — handler와 disasm 항목 한 곳. 새 op 추가 시 한 파일만 수정.
  - **상위 영향:** **P2 ROADMAP success criterion 2 수정 필요** — "len(get_disasms()) == ~140"을 "P2: get_disasms() 구조 + ~10 SPR/control 항목; full ~140은 P5/P6 누적 목표"로 변경. **discuss-phase 종료 직후 별도 커밋으로 ROADMAP.md 동기화**.
- **D-10:** **disasm.py 파일은 `src/main/python/riscv/gtx/disasm.py`** (Phase 1 D-13에는 없는 신규 파일). encoding.py(funct7 상수)와 분리, op 모듈 등록 시 disasm 항목 누적/조회 API 제공.
- **D-11:** **`encoding.py` 증축** — Phase 1의 8개 funct7 stub에서 P2 시점에 완전한 funct7 세트 (gem5 0x04-0x07, ISS 전체 0x00-0x7F) + funct3 (custom1 start_p/end_p/start_t/end_t/wsplit/wjoin) + Mode 상수 (Mode 1-4) 추가.
- **D-12:** **Sample 5 disasm 테스트** — `test_disasm_table.py::test_get_disasms_length_and_samples`: `assert len(disasms) >= EXPECTED_P2_COUNT` (수정된 success criterion 2 기준), `for m in ['mm', 'mm_s', 'mm_t', 'dma_load', 'wsplit']: assert any(d.name == m for d in disasms)`.
  - **상위 영향:** Sample 5 중 'mm', 'mm_s', 'mm_t' 등은 P4 MM op이므로 P2에서는 PASS 안 함 — 테스트 자체는 P2에서 skipif/xfail 처리, P4 시점에 활성화. 또는 P2 sample을 ['wrspr', 'rdspr', 'wsplit', 'wjoin', 'start_p']로 축소.

### Per-op registry 프로토콜 (P3-P5 op 추가용 API)

- **D-13:** **Decorator-based registry** — `@gtx.handler(funct7=0x49, funct3=None, mnemonic='wrspr', mask=..., kind='custom0')` 데코레이터가 함수를 GtxNpu의 dispatch dict와 disasm.py에 자동 등록. PY-FUNCT7-01(v2 deferred)과 별개의 내부 API.
  - **이유:** 명시적, 함수 정의 옆에 메타데이터, P3-P5에서 op 모듈 작성 시 API 표면 단순.
  - **고려 사항:** 데코레이터는 module import 시점에 GtxNpu 클래스 메서드로 등록 (class-level이 아닌 instance-level은 `__init__()`에서 module을 import하면 자동 처리). 실제 패턴은 plan 단계.
- **D-14:** **op 모듈 위치** — `src/main/python/riscv/gtx/ops/spr.py` (P2 시작), `ops/control.py` (warp 루프), 추후 `ops/dma.py` (P3), `ops/mm.py` (P4), `ops/vec.py` / `ops/act.py` (P5).

### `_riscv.so` 빌드 + submodule 해결 경로

- **D-15:** **pybind11<3.0.4 pin in `[build-system].requires`** (config-only, Phase 1 deferred-items 해결).
  - **이유:** CLAUDE.md "no new C++ code" 완전 준수. cibuildwheel도 이 pin을 따름. 단점: pybind11 최신 기능 못 씀 (현재는 별 송결 아님).
  - **위치:** `pyproject.toml [build-system].requires`에 `"pybind11>=3,<3.0.4"`로 변경 (Phase 1에서는 `pybind11>3` 형태였음).
- **D-16:** **submodule SHA 재검증 + 재등록** — P2 첫 작업으로 `git config -f .gitmodules`로 URL 확인, `git ls-remote https://github.com/Sudo42b/gtx_spike` 로 SHA 존재 확인 후 필요 시 `git submodule sync` + `git submodule update --init`. 현재 SHA `80d524293407ceb9654b6e9c3aef0186b4e3af98`가 fetch 안 됨 (P1 plan 05의 의도된 SHA가 origin에 push되지 않았을 가능성).
  - **이유:** disasm.py 작성 시 `vendor/gtx_cpp_reference/gtx/gtx_npu_disasm.inc` 직접 참조 가능해야 함. WRSPR/RDSPR rs1/rs2 의미 (D-03)도 C++ 소스 참조 필요.

### Test 전략 (로컬 + CI)

- **D-17:** **Hybrid mock 전략** — `tests/gtx/conftest.py`에 `try: from riscv.processor import processor_t; except ImportError: from ._mocks import MockProcessor as processor_t`. CI에서는 _riscv 빌드되므로 실제, 로컬에서는 mock.
  - **이유:** 양 환경 모두 같은 테스트 코드. mock 사용 여부는 import 시점에 결정.
- **D-18:** **`tests/conftest.py` 이존 try/except 가공** — 별도 P2 plan으로 처리. 현재 `from riscv.cfg import ...`가 _riscv 강제 import → conftest 로드 실패. P2에서 try/except로 감싸고 fixture 조건부 로드.
- **D-19:** **Mock 최소 spec (P2)** — `MockProcessor.get_state().XPR.read(i)/write(i, val)`, `MockInsn.rs1/rs2/funct/xs1/xs2/xd`, `MockState.XPR` (배열 형태). MMU는 P3 시점에 `load_uint64/store_uint64` 추가 (DMA 핸들러 작성 시).
  - **이유:** P2 unit test는 SPR/warp만 테스트하므로 MMU 불필요. CLI 테스트(_riscv 필요)에서만 실제 MMU 사용.
- **D-20:** **Mock 클래스는 `tests/gtx/_mocks.py` 내부만** — production wheel에 미포함. P3-P6 tests는 `from .._mocks import *` 경로로 재사용. `riscv.gtx._test_helpers`로 노출 안 함 (public API 오염 회피).
- **D-21:** **`@isa.register('gtx')` 검증 = 설계 계약 + skipif 통합** — `test_register.py` 안에서 1) 항상 (mock 환경) `assert issubclass(GtxNpu, isa.ROCC)`, `assert GtxNpu.name == 'gtx'`, `assert hasattr(GtxNpu, 'custom0')` 등. 2) `_riscv` 있을 때만 (`@pytest.mark.skipif(not _RISCV_AVAILABLE)`) register_extension 동작 검증 (factory 등록 + name lookup).
- **D-22:** **`nop_wjoin.elf` 프리빌드 바이너리** — `tests/gtx/data/elf/nop_wjoin.elf`로 RISC-V cross-toolchain (`/opt/riscv/`) 결과 커밋. P2는 테스트 픽스처로만 사용. `[tool.setuptools.package-data]`에 등록은 PKG-01 (P5/P6) 일.
  - **이유:** 재현성 + 테스트 시간 절약. 바이너리 ~1KB. 소스(`nop_wjoin.S`)도 같은 디렉토리에 commit하여 빌드 재현 가능.

### `test_warp.py` / `test_spr.py` 패턴

- **D-23:** **Warp 상태 검증은 직접 `npu.custom1(proc, insn, xs1, xs2)` 호출 후 WarpState 필드 직접 assertion**. `funct3=0(start_p) → assert npu.warp.is_ploop == True`, `funct3=2(end_p) → assert npu.warp.is_ploop == False` 등. test_spr.py도 동일한 직접 호출 패턴.

### Claude's Discretion

다음은 implementation detail로 Claude 판단:
- 데코레이터 정확한 구현 (D-05/D-13) — 4-arg signature wrap 방식, class-level vs instance-level
- `mxe_accum` shape의 정확한 M_TILE/N_TILE 값 (D-06 plan 단계)
- WRSPR/RDSPR rs1/rs2 의미 (D-03 plan 단계)
- Mock 클래스 정확한 메서드 시그니처 (D-19) — `riscv.processor` 실제 API에 맞춰
- xs1=0 우회 데코레이터 vs helper 함수 trade-off 측정 (P4 hot path)
- ELF 빌드 스크립트 및 소스 파일 정확한 형태 (`nop_wjoin.S`, `Makefile` 또는 단일 명령)
- `test_disasm_table.py` 샘플 5개 (D-12) — P2에 포함된 op만 사용 (`wrspr`, `rdspr`, `wsplit`, `wjoin`, `start_p`)

### Folded Todos
None — `gsd-tools todo match-phase 2`에서 매칭 0건.

</decisions>

<canonical_refs>
## Canonical References

**Downstream agents MUST read these before planning or implementing.**

### Project documents (locked context)
- `.planning/PROJECT.md` — Core value, validated requirements (PYS-EXT-01..07, GTX-MEM-01, GTX-REF-01), Active GTX-CORE-01..04 + GTX-SPR-01 등
- `.planning/REQUIREMENTS.md` — CORE-01..04, SPR-01/02, DISASM-01, DISP-01..02 v1 acceptance criteria
- `.planning/ROADMAP.md` — Phase 2 success criteria 1-5 (단, **D-09에 의해 #2는 plan 단계에서 수정**)
- `.planning/STATE.md` — 현재 진행 (Phase 1 완료, Phase 2 ready to plan)
- `.planning/phases/01-foundation/01-CONTEXT.md` — D-01..D-17 locked decisions (특히 D-09 FP16 view, D-10..D-12 memory layered API, D-13 모듈 layout)
- `.planning/phases/01-foundation/01-VERIFICATION.md` — Phase 1 결과 (13/13 tests pass)
- `.planning/phases/01-foundation/deferred-items.md` — pybind11 3.0.4 csr_t 이슈 (D-15에서 해결)

### C++ ground-truth (via submodule, D-04 from Phase 1)
- `vendor/gtx_cpp_reference/gtx/CLAUDE.md` — 메모리 계층, 인코딩, byte-order 규약
- `vendor/gtx_cpp_reference/gtx/gtx_npu.h` — `gtx_npu_t` 클래스 시그니처 (NPU 본체)
- `vendor/gtx_cpp_reference/gtx/gtx_npu_core.cc` — `reset()`, `custom0/custom1` dispatch (D-01/D-02 ground-truth)
- `vendor/gtx_cpp_reference/gtx/gtx_npu_spr.cc` — WRSPR/RDSPR 라우팅 (D-03 ground-truth)
- `vendor/gtx_cpp_reference/gtx/gtx_npu_warp.cc` — warp 루프 상태 머신 (D-04)
- `vendor/gtx_cpp_reference/gtx/gtx_npu_disasm.inc` — ~140 disasm 매크로 (D-09 누적 목표)
- `vendor/gtx_cpp_reference/gtx/gtx_params.h` — HW 파라미터 (M_TILE/N_TILE for D-06)
- `vendor/gtx_cpp_reference/gtx/gtx_encoding.h` — funct7/funct3/Mode 상수 (D-11 source)
- **참고:** D-16에 따라 P2 첫 작업으로 submodule SHA 재검증 + init 필요

### Existing pyspike RoCC patterns (read for conventions)
- `src/main/python/riscv/isa.py` — `ROCC`, `ISA` 베이스 클래스, `@register("name")` 데코레이터 (line 21+)
- `examples/xhuimt/__init__.py` — RoCC 확장 예시 (HuiMtISA, custom0/1 가까운 analog)
- `examples/xhuimt/mylrsc.py` — `reset()`, `get_instructions()`, `get_disasms()`, `get_csrs()` 패턴
- `examples/xhuimt/mycsrs.py` — CSR 정의 패턴 (Phase 2는 SPR이라 직접 적용 안 되지만 패턴 참고)
- `tests/conftest.py` — pyspike 기존 fixture 패턴 (D-18에서 try/except 추가)
- `tests/test_extension.py` — 기존 extension 테스트 패턴
- Phase 1 `tests/gtx/test_fp_roundtrip.py`, `test_memory_layout.py` — `--noconftest -o "addopts="` 우회 패턴

### Phase 1 Phase 2 의존 자산 (이미 wheel에 land)
- `src/main/python/riscv/gtx/__init__.py` — 패키지 진입점, LE byteorder tripwire (D-09 P1)
- `src/main/python/riscv/gtx/params.py` — HW 상수 (`GTX_NEST_NUM=4`, `GTX_SPU_NUM=16`, `GTX_L1_SIZE_BYTES`, ...)
- `src/main/python/riscv/gtx/encoding.py` — funct7 stub 8개 (P2에서 D-11에 따라 증축)
- `src/main/python/riscv/gtx/fp.py` — FP16↔FP32 (P4 사용)
- `src/main/python/riscv/gtx/memory.py` — `GtxMemory` (P2 GtxNpu 인스턴스가 보유)
- `src/main/python/riscv/gtx/ddr.py` — DDR lazy alloc (P3 사용)
- `src/main/python/riscv/gtx/ops/__init__.py` — ops 패키지 마커 (P2 시작 시 ops/spr.py, ops/control.py 추가)

### Build system 자산
- `pyproject.toml` — D-15 (pybind11<3.0.4 pin) 적용 위치, `[build-system].requires`
- `setup.py` — pybind11 호출 + 빌드 옵션 (P2는 직접 수정 안 함, build-system pin만)
- `MANIFEST.in` — `prune vendor/gtx_cpp_reference` (Phase 1에서 추가됨)
- `.gitmodules` — `vendor/gtx_cpp_reference` URL/SHA (D-16 재검증 대상)

</canonical_refs>

<code_context>
## Existing Code Insights

### Reusable Assets (Phase 1 산출물)
- **`riscv.gtx.memory.GtxMemory`** — Phase 2 `GtxNpu`가 `self.mem = GtxMemory()`로 보유. SPR routing이 `mem.spr` dict 사용.
- **`riscv.gtx.fp`** — Phase 4 MM op이 사용. P2는 직접 의존 안 함.
- **`riscv.gtx.params`** — `GTX_NEST_NUM`, `GTX_SPU_NUM`, L1/L2 크기 등 모든 HW 상수. P2의 reset()이 메모리 초기화에 사용.
- **`riscv.gtx.encoding`** — funct7 stub. D-11에 따라 P2 시점에 funct7/funct3/Mode 상수 완전 세트 추가.

### Established Patterns
- **모듈 명명**: lowercase + underscore. `gtx/npu.py`, `gtx/dispatch.py`, `gtx/disasm.py`, `gtx/warp_state.py`, `gtx/spr_router.py` (D-가 따른 layered split).
- **Type hints**: explicit, mypy-checked.
- **Pylint 설정**: max-line-length 120, missing-docstring 비활성화.
- **`@isa.register("name")` 데코레이터**: `examples/xhuimt/__init__.py:24` 기준 패턴. `name`은 lowercase.
- **`get_instructions/get_disasms/get_csrs/reset` 오버라이드**: `examples/xhuimt/__init__.py:36+` 패턴.
- **TDD 워플로우** (Phase 1에서 확립): `test_*.py` RED → 모듈 구현 GREEN → 추가 픽스/리팩터링.
- **Test 격리** (Phase 1에서 확립): `pytest tests/gtx/ --noconftest -o "addopts="` — _riscv 의존 회피. P2부터는 D-17/D-18 hybrid 적용.

### Integration Points
- **`riscv` namespace**: `from riscv import isa` 가 `_riscv` 의존 → mock fallback 필요 (D-17/D-18).
- **`pyspike --extlib=riscv.gtx`** CLI: `src/main/python/riscv/__main__.py` 가 `PYSPIKE_LIBS` env로 모듈 로드. P2는 추가 수정 불필요 (Phase 1 packaging discovery glob 픽스로 자동 발견).
- **`tests/conftest.py`** D-18에 따라 try/except로 가공 — `riscv.cfg/sim` 등의 강제 import 안전화.

### Anti-patterns to avoid
- `riscv/gtx/__init__.py`에 무거운 import 추가 (`from . import npu`) — `import riscv` 하기만 해도 NumPy + 모든 gtx 모듈 로드됨. lazy import 권장 (P1에서 이미 확립).
- 단일 거대 dispatch dict in `npu.py` (D-01의 layered split이 더 나음, D-가 결정).
- Mock 클래스 production wheel에 노출 (D-20).

</code_context>

<specifics>
## Specific Ideas

### `nop_wjoin.S` 어셈블리 골자 (참고용, plan 단계에서 정확화)
```asm
.global _start
_start:
    addi sp, sp, -16          # sp init = 0x80100000 - 16
    # ... NOP body (없을 수도) ...
    .insn r 0x2b, 0, 0, x0, x0, x0   # custom1 funct3=0(WJOIN equivalent)
    j .                        # safety loop (WJOIN raise SystemExit이면 도달 안 함)
```
ELF 헤더는 spike 표준 entry point (`_start = 0x80000000`)에 맞춤.

### WarpState dataclass 정확한 형태
```python
from dataclasses import dataclass

@dataclass
class WarpState:
    is_ploop: bool = False
    is_tloop: bool = False
    is_sloop: bool = False  # P3+에서 활성화
```

### Hybrid mock 전략 conftest 골자
```python
# tests/gtx/conftest.py
try:
    from riscv.processor import processor_t
    from riscv.extension import rocc_insn_t
    _RISCV_AVAILABLE = True
except ImportError:
    from ._mocks import MockProcessor as processor_t  # type: ignore
    from ._mocks import MockInsn as rocc_insn_t       # type: ignore
    _RISCV_AVAILABLE = False
```

### GtxNpu 클래스 헤더 골자 (D-가 layered split 적용 시)
```python
# src/main/python/riscv/gtx/npu.py
from typing import List
import os
import numpy as np
from riscv import isa
from riscv.disasm import disasm_insn_t
from riscv.processor import insn_desc_t, processor_t
from .memory import GtxMemory
from .warp_state import WarpState
from .dispatch import build_custom0_table, build_custom1_table
from .spr_router import wr_spr, rd_spr
from .params import GTX_NEST_NUM, GTX_SPU_NUM
from . import disasm as _disasm

@isa.register("gtx")
class GtxNpu(isa.ROCC):
    """GTX NPU functional model — Phase 2 dispatch shell."""
    def __init__(self):
        super().__init__()
        self.mem = GtxMemory()
        self.warp = WarpState()
        self._mxe_accum = np.zeros((GTX_NEST_NUM, GTX_SPU_NUM, M_TILE, N_TILE), dtype=np.float32)
        self._custom0 = build_custom0_table(self)
        self._custom1 = build_custom1_table(self)
    # ...
```

### Decorator-based op registry 골자
```python
# src/main/python/riscv/gtx/_registry.py
_HANDLERS = {}  # (kind, funct7or3) → (handler_fn, mnemonic, mask)

def handler(funct7=None, funct3=None, mnemonic=None, mask=0xFFFFFFFF, kind="custom0"):
    def decorator(fn):
        key = (kind, funct7 if funct7 is not None else funct3)
        _HANDLERS[key] = (fn, mnemonic, mask)
        return fn
    return decorator

# src/main/python/riscv/gtx/ops/spr.py
from .._registry import handler

@handler(funct7=0x49, mnemonic='wrspr', mask=0x...)
def _wrspr(npu, proc, insn, xs1, xs2):
    ...
```

</specifics>

<deferred>
## Deferred Ideas

### Phase 2 끝 직후 별도 ROADMAP/REQUIREMENTS 동기화 커밋
**discuss-phase 종료 직후 별도 커밋 (사용자 결정)** —
- ROADMAP.md Phase 2 success criterion 2 수정: "len(get_disasms()) == ~140" → "P2: get_disasms() 구조 + ~10 SPR/control 항목 등록; full ~140은 P5/P6 누적 목표"
- ROADMAP.md Phase 2 의 "Plans: TBD" 자리 plan-phase 후 갱신
- DISP-01 명시 명확화 ("`insn.rs1 != 0` heuristic")
- CORE-04 명시 명확화 ("xs1=0 우회는 데코레이터 자동 wrap")

### v2 / 향후 마일스톤
- **PY-FUNCT7-01** (사용자가 명시 정의 funct7 등록 외부 API) — v2. P2 내부 registry는 별개.
- **CYC-01/02** 사이클 카운팅 — v2.
- **MEXEC-01** mexec full microcode 페치-디코드 — v1 펌웨어가 트리거 안 하면 stub.

### Out of scope for Phase 2 (다른 페이즈)
- **DMA op 핸들러** + DDR hex I/O → P3 DMA-01..05
- **MM op 변형** (`exec_mm`, `mxe_accum` write) → P4 MM-01..05
- **VEC/ACT op** → P5 VEC-01..05, ACT-01..05
- **DISP-03** 4-mode dispatch router 풀 활성화 → P3+
- **VRF-01..04** verify.py 포팅 → P6
- **PKG-01** ELF/golden hex `[tool.setuptools.package-data]` 등록 → P5/P6

### Defer to user follow-up
- D-15 pybind11<3.0.4 pin이 cibuildwheel 빌드 매트릭스에 영향 미치는지 — Phase 2 첫 plan에서 검증 (사용자 명시: 별도 plan 생성 가능)
- submodule 재등록 시 SHA가 origin에 push되었는지 (D-16) — Phase 2 첫 작업으로 처리

### Reviewed Todos (not folded)
None — `gsd-tools todo match-phase 2` 매칭 0건.

</deferred>

---

*Phase: 02-skeleton-disasm*
*Context gathered: 2026-05-04 via /gsd:discuss-phase 2*
