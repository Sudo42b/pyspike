# Phase 1: Foundation - Context

**Gathered:** 2026-05-04
**Status:** Ready for planning

<domain>
## Phase Boundary

Phase 1은 GTX NPU 포트의 **인프라 레이어**를 구축한다. 구체적으로:

1. **FP16↔FP32 변환 헬퍼** (`riscv.gtx.fp`) — NumPy `float16` view 기반 (사용자 결정)
2. **NumPy 백엔드 메모리 레이어** (`riscv.gtx.memory`) — L0/L1/L2/DDR을 `np.uint8` ndarray + halfword view로 표현, GSPR/NSPR/LSPR을 단일 dict로 통합
3. **`riscv.gtx` 패키지 스켈레톤** — `__init__.py`, `params.py`, `encoding.py`, `fp.py`, `memory.py`, `ddr.py` (P3+가 채울 빈 모듈 포함)
4. **C++ 레퍼런스 git submodule** — `vendor/gtx_cpp_reference/`에 `https://github.com/Sudo42b/gtx_spike` submodule
5. **NumPy/cp310+ packaging 베이스라인** — `pyproject.toml` 수정, cibuildwheel 매트릭스 cp310-cp312로 축소

다음 모두는 **Phase 1 비범위(out-of-scope for THIS phase)** — 다른 페이즈가 다룬다:
- ROCC subclass / custom0/1 dispatch (Phase 2)
- SPR routing 비즈니스 로직 / WRSPR/RDSPR 핸들러 (Phase 2)
- disasm 테이블 (Phase 2)
- DMA op / DDR hex I/O 파싱 (Phase 3)
- MM/VEC/ACT op 핸들러 (Phase 4-5)
- verify.py 포팅 (Phase 6)

</domain>

<decisions>
## Implementation Decisions

### DDR 할당 전략
- **D-01:** **Lazy `ensure_ddr` 패턴** — DDR 버퍼는 첫 접근 시 필요한 범위만 할당 (C++ `gtx_npu_t::ensure_ddr` 동치). `GtxNpu` 인스턴스 생성 시 4GB 사전 할당 안 함.
  - **이유:** 테스트 시작 시간 단축, 메모리 사용량 절약, C++ 동작 동등성
- **D-02:** **DDR 최대 크기는 환경변수 `GTX_DDR_SIZE`로 제어** — 기본값 4GB. cap 초과 접근은 명시적 에러 raise.
  - **이유:** C++ 동등 유연성. 펌웨어/golden hex가 4GB 미만이면 작은 값으로 CI 메모리 압박 회피.
- **D-03:** **`GTX_DDR_REVERSED=1` 모드는 I/O 경계(`ddr_init_from_file`/`ddr_dump_to_file`)에서만 변환.** 내부 DDR 버퍼는 항상 LE 저장. 32-byte 버스 워드 단위 역순.
  - **이유:** 내부 일관성. 모든 op 코드가 단일 LE 가정을 공유 — 분기 부담 제거.

### C++ 레퍼런스 스냅샷
- **D-04:** **git submodule** at `vendor/gtx_cpp_reference/` pointing to **`https://github.com/Sudo42b/gtx_spike`** (공개 레포).
  - **이유:** 업스트림 동기화 자동화 (`git pull --recurse-submodules`). CI/cibuildwheel이 익명 clone 가능.
- **D-05:** **submodule 범위 = `gtx/` 디렉토리 + spike 패치 (`riscv-isa-sim/` 변경 지점)** — 독립 빌드 재현 가능한 완결 세트.
  - **이유:** Ground-truth로서 재현성 우선. P4/P5에서 C++ 동작 비교 빌드 시 필요할 수 있음.
- **D-06:** **vendor/gtx_cpp_reference/는 wheel에 미포함.** 개발자 레퍼런스 전용 — `MANIFEST.in` exclude + `[tool.setuptools.package-data]`에서 미선언.
  - **이유:** wheel 사이즈 절약, 라이선스/IP 단순화 (사용자는 pip install시 C++ 소스 미수령).

### NumPy & Python 버전 (★ 프로젝트 레벨 변경)
- **D-07:** **NumPy 의존성 = `numpy>=2.0`.** 이전 research 권장(`>=1.20,<2.0`)을 **번복**.
  - **이유:** NumPy 2.x의 FP16 시맨틱이 더 결정적·일관적. 사용자 명시 결정.
  - **상위 영향:** NumPy 2.0이 cp38/cp39 드랍 → cibuildwheel 매트릭스도 cp310-cp312로 축소되어야 함.
- **D-08:** **`requires-python = ">=3.10"` (cp38/cp39 드랍)** — pyproject.toml 변경 + classifiers 업데이트.
  - **상위 영향:** pyspike "Validated PYS-EXT-06" 가정 무효화 — PROJECT.md/REQUIREMENTS.md 업데이트 필요.
- **D-09:** **FP16 변환 = `np.float16` view (NOT 순수 Python 비트 조작).** 이전 research 권장(`gtx_npu.h:89-151` 비트 포팅)을 **번복**.
  - **이유:** NumPy 2.x FP16 RNE가 IEEE 754 binary16과 일치. 코드 단순. 사용자 명시 결정.
  - **위험 (P4/P5에서 측정):** subnormal/NaN payload/halfway-rounding edge case에서 C++ `gtx_fp32_to_16`(명시적 sticky/round-half-to-even)와 차이 가능. `verify.py --strict` 모드 통과 측정 필요. 차이 발견 시 P4/P5에서 부분적 비트 조작 폴백 추가.

### Memory 클래스 API 표면
- **D-10:** **Layered API.** 두 레벨 동시 노출:
  - **Raw view (low-level):** `mem.l0[nest][spu]` / `mem.l1[nest][spu]` / `mem.l2[nest]` / `mem.ddr` → `np.uint8` ndarray (halfword view 호출자 책임)
  - **Named accessor (high-level):** `mem.l1_f16(nest, spu, addr, length) -> np.ndarray[float16]` 등 — view 반환 보장
  - **이유:** op handler는 주로 helper 사용해 가독성 확보, edge case는 raw로 fall through.
- **D-11:** **SPR 통합 dict + 주소 기반 라우팅.** `mem.spr` 는 `dict[int, int]`. 주소 0x000-0x3FF=GSPR, 0x400-0x7FF=NSPR, 0x800-0xBFF=LSPR (NEST/SPU 인덱스는 키에 인코딩, C++와 동일).
  - **이유:** C++ `gtx_npu_t`의 `unordered_map<uint16_t, uint64_t>` 매칭. WRSPR/RDSPR 라우팅이 단일 dict 액세스로 단순.
- **D-12:** **모든 named accessor가 non-copying view 반환 보장.** `arr.base is not None` 단위 테스트로 매 helper 검증. 쓰기는 원본에 반영.
  - **이유:** op이 `np.copyto(mem.l1_f16(...), result)` 또는 `mem.l1_f16(...)[i] = x` 어느 형태든 in-place 작동해야 bit-exact 유지.

### Module Layout (이미 잠긴 사항 — 재확인)
- **D-13:** Phase 1에서 생성하는 파일: `src/main/python/riscv/gtx/{__init__.py, params.py, encoding.py, fp.py, memory.py, ddr.py}` + `src/main/python/riscv/gtx/ops/__init__.py` (ops 디렉토리 마커, 빈 파일들은 P2-P5에서 채움).
- **D-14:** `__init__.py`는 `GtxNpu` re-export를 P2 시점에 추가하지만, P1에서는 `fp`, `memory`, `params`만 노출.

### Test Scaffolding (Phase 1에 함께 들어감)
- **D-15:** 테스트 위치 = `tests/gtx/` (기존 `tests/test_extension.py` 옆).
  - 파일: `tests/gtx/__init__.py`, `tests/gtx/test_fp_roundtrip.py`, `tests/gtx/test_memory_layout.py`
  - `pytest tests/gtx/` 로 단독 실행 가능, 상위 `pytest tests/` 통합도 자동.
- **D-16:** **FP round-trip 전수 테스트** — 65536개 FP16 값 모두 round-trip (`fp16 → fp32 → fp16 == fp16`). 현대 CPU에서 1초 미만 예상.
- **D-17:** **LE byte-order 어서션 테스트** — `mem.l1_f16(0,0)[0] = np.float16(1.0)` 후 `mem.l1[0][0][:2] == bytes([0x00, 0x3C])` 확인. NumPy host endian과 무관하게 LE 보장.

### Claude's Discretion
다음은 implementation detail로 Claude 판단:
- `np.float16` view 헬퍼 내부 구현 디테일 (e.g., `arr.view(np.float16)` 호출 위치, view 슬라이싱 alignment 가드)
- `params.py` 상수 명명 규칙 (C++ 매크로 그대로 vs Python 컨벤션 변환) — 명시되지 않은 경우 C++ 매크로 이름 그대로 (`GTX_NEST_NUM=4`, `GTX_SPU_NUM=16`, `GTX_L1_SIZE_BYTES=384*1024` 등)
- `encoding.py`에 어떤 상수까지 포함할지 (전체 disasm.inc 매크로는 P2 disasm.py로, P1은 funct7 상수만)
- `MANIFEST.in` 정확한 exclude 패턴
- pyproject.toml의 NumPy 정확한 표기 (`numpy>=2.0` vs `numpy>=2.0,<3`) — 보수적으로 `numpy>=2.0,<3` 권장
- CI 매트릭스 변경 시 `[tool.cibuildwheel].build` 정확한 항목 (cp38/cp39 라인 제거)

</decisions>

<canonical_refs>
## Canonical References

**Downstream agents MUST read these before planning or implementing.**

### Project documents (locked context)
- `.planning/PROJECT.md` — Core Value, Constraints, Out of Scope, Key Decisions (NumPy/cp310 변경 사항은 phase 1 실행 시 동기화 필요)
- `.planning/REQUIREMENTS.md` — FOUND-01..04, PKG-02 v1 requirements (PKG-04는 cp310-cp312로 축소 필요)
- `.planning/ROADMAP.md` — Phase 1 success criteria (cp38 가정 부분 cp310로 업데이트 필요)
- `.planning/STATE.md` — 현재 진행 상황, locked decisions

### Research (already produced)
- `.planning/research/STACK.md` §"NumPy" — NumPy 1.x 권장은 사용자 결정으로 번복됨, 그러나 §"FP16 storage" 권장은 부분 채택 (uint8 ndarray + halfword view)
- `.planning/research/STACK.md` §"Endianness" — `np.frombuffer(buf, dtype='<f2')` 패턴은 D-12 참조
- `.planning/research/PITFALLS.md` Pitfall 1, 2, 8, 13 — LE 바이트 순서, FP16 cast precision (D-09 위험 평가에 반영)
- `.planning/research/ARCHITECTURE.md` §"Module layout", §"Memory layout" — Phase 1 모듈 분할의 근거

### C++ ground-truth (via submodule when D-04 적용)
- `vendor/gtx_cpp_reference/gtx/CLAUDE.md` — 메모리 계층, 인코딩, byte-order 규약
- `vendor/gtx_cpp_reference/gtx/gtx_npu.h` lines 85-151 — `gtx_fp16_to_32`, `gtx_fp32_to_16` 비트 동작 (D-09 fallback 발생 시 포팅 대상)
- `vendor/gtx_cpp_reference/gtx/gtx_params.h` — HW 파라미터 상수 (params.py 포팅 source)
- `vendor/gtx_cpp_reference/gtx/gtx_npu_core.cc` — `ensure_ddr` 패턴 (D-01 구현 시 참조)
- `vendor/gtx_cpp_reference/gtx/gtx_npu_dma.cc` — DDR_REVERSED I/O 처리 (D-03)

### Existing pyspike (no change required by Phase 1, but to read for conventions)
- `.planning/codebase/CONVENTIONS.md` — RoCC pattern, `@isa.register` decorator
- `.planning/codebase/STRUCTURE.md` — `src/main/python/riscv/` 모듈 위치 결정의 근거
- `src/main/python/riscv/__init__.py` — 기존 export 패턴 (`gtx`를 어떻게 추가할지 참고)
- `src/main/python/riscv/isa.py` — `ROCC`, `ISA` 베이스 (P2에서 사용; P1은 직접 의존 안 함)
- `pyproject.toml` — D-08 변경 대상 (cp38/cp39 라인 제거, NumPy>=2.0 추가)
- `setup.py` — C++ 빌드 옵션, RISCV env var 처리 — Phase 1은 직접 수정 안 함
- `MANIFEST.in` — D-06 exclude 패턴 추가 위치

### NumPy 2.x docs (확인 필요)
- NumPy 2.0 release notes — FP16 시맨틱 변경/안정성 명시 부분 (P1 시작 시 빠르게 점검 권장)
- NumPy `dtype.byteorder` / `np.frombuffer` doc — D-12 view 보장 검증

</canonical_refs>

<code_context>
## Existing Code Insights

### Reusable Assets
- **`src/main/python/riscv/__init__.py` (1.9K)** — 현재 `riscv` 네임스페이스 export. P1에서 `gtx` 서브패키지 추가 시 이곳에 `from . import gtx` 추가 (또는 lazy import).
- **`src/main/python/riscv/_utils.py` (4.5K)** — 기존 헬퍼 (셸 호출, env var 등). FP/memory 헬퍼는 도메인이 다르므로 재사용 안 하고 `gtx/` 내부에 둠.
- **`tests/test_extension.py`** — 기존 RoCC 테스트 패턴. `tests/gtx/test_*.py` 작성 시 `pytest --pylint --mypy` 통과 보장 위해 동일 import/typing 스타일 따름.
- **`pyproject.toml [tool.setuptools.package-data]`** — 이미 `data/bin/`, `data/lib/` 등 패턴 존재. `gtx/data/` 하위 자산은 P3+에서 추가 (P1은 미해당).

### Established Patterns
- **모듈 명명**: lowercase + underscore (`isa.py`, `dev.py`, `_utils.py`) — `gtx/` 하위 파일도 동일 (`fp.py`, `memory.py`, NOT `FP.py`).
- **Type hints**: explicit, mypy-checked. `gtx/` 모든 public 함수에 type hints 강제 (`pytest --mypy`가 검사).
- **Pylint 설정**: max-line-length 120, missing-docstring 비활성화. 적용.
- **`riscv` namespace package**: `__init__.py`에서 명시적 import. lazy하게 `gtx`를 import하면 numpy 미설치 환경에서 `import riscv`가 깨질 수 있음 — `try/except ImportError`로 감싸 안전 noticeable warning.
- **Test layout**: `tests/test_*.py`. 새 디렉토리 `tests/gtx/`도 `pytest`가 자동 발견 (`testpaths = ["tests"]`).

### Integration Points
- **`riscv` namespace re-export**: `src/main/python/riscv/__init__.py`에 `from . import gtx`(또는 lazy) 추가.
- **`pyproject.toml` package discovery**: `[tool.setuptools.packages.find].include = ["riscv"]` — 자동으로 `riscv.gtx` 발견. 추가 설정 불필요.
- **mypy / pylint**: `gtx/` 신규 코드도 자동 적용.
- **cibuildwheel build matrix**: D-08에 따라 `[tool.cibuildwheel].build`에서 cp38/cp39 라인 제거.

### Anti-patterns to avoid (from PITFALLS.md)
- `arr.view(np.float16)` 직접 호출 — host-native endian 위험. 항상 `np.frombuffer(buf, dtype='<f2')` 또는 명시적 dtype byteorder.
- `np.zeros(GTX_DDR_SIZE_BYTES, dtype=np.uint8)` eager 할당 — D-01에 위배. `GtxMemory.ensure_ddr(size)` 내부에서만 lazy 확장.
- in-place ops에 `arr.copy()` 끼어들기 — D-12 위배. helper 작성 시 `assert result.base is not None` 가드.

</code_context>

<specifics>
## Specific Ideas

### NumPy 2.0 FP16 view 채택 정당성 (D-09 사용자 결정)
사용자는 코드 단순성과 NumPy 2.x의 FP16 RNE 표준화를 신뢰한다. P4/P5에서 verify.py
--strict 모드 통과를 측정해 차이 발생 시:
- **차이 0건**: D-09 정상 작동, 추가 작업 없음
- **차이 < 0.1% & ULP 1 이내**: 유지 가능, atol/ULP 슬랙으로 흡수
- **차이 ≥ 0.1% 또는 strict 실패**: 부분적으로 `gtx_npu.h:89-151` 비트 포팅 패치를
  `gtx/fp.py`에 fallback으로 추가 (sticky bit, RNE half-to-even, NaN payload 보존)

### git submodule URL — `https://github.com/Sudo42b/gtx_spike` (공개)
P1 실행 시 `git submodule add` 단일 명령. CI 영향: `git submodule update --init --recursive`
한 단계가 cibuildwheel 사전 단계에 추가됨 — `[tool.cibuildwheel.linux] before-all`에 포함 또는
`scripts/setup-vendor.sh` 등 작은 wrapper.

### `GTX_DDR_SIZE` 기본값
4GB. 펌웨어 회귀 .elf가 4GB 이상의 DDR을 요구하지 않으므로 안전한 상한.
CI에서 압박 시 `GTX_DDR_SIZE=64M` 등으로 다운사이즈.

### Layered Memory API 사용 예시
```python
# Low-level (op이 raw view 직접 다루는 edge case)
buf = mem.l1[nest][spu]  # np.ndarray[uint8], shape=(GTX_L1_SIZE_BYTES,)
buf[off:off+2] = bytes([fp16 & 0xFF, (fp16 >> 8) & 0xFF])

# High-level (일반적인 op handler)
view = mem.l1_f16(nest, spu, addr=0x100, length=64)  # ndarray[float16], view of l1
view[i] = np.float16(result)  # in-place; mem.l1[nest][spu]에 즉시 반영
assert view.base is not None  # always
```

### SPR 단일 dict + 주소 라우팅 예시
```python
# 0x000-0x3FF: GSPR (글로벌)
mem.spr[GSPR_GTX_OPCODE] = 0x10  # SASMD opcode

# 0x400-0x7FF: NSPR (NEST별, key에 nest_id 인코딩)
mem.spr[NSPR_BASE | (nest_id << NSPR_NEST_SHIFT) | NSPR_OFFSET_X] = ...

# 0x800-0xBFF: LSPR (SPU별)
mem.spr[LSPR_BASE | (nest_id << LSPR_NEST_SHIFT) | (spu_id << LSPR_SPU_SHIFT) | LSPR_OFFSET_X] = ...
```
정확한 인코딩은 P2 SPR-01 실행 시 C++ `gtx_npu_spr.cc` 직접 포팅.

</specifics>

<deferred>
## Deferred Ideas

### 사용자가 발생하지 않았지만 Claude가 인지한 향후 고려 사항
- **`@isa.register("gtx")` 데코레이터 사용** — Phase 2에서 `GtxNpu`에 적용. P1에서 패키지 자체는 `riscv.gtx`로 import 가능하지만 spike `--extlib` 등록은 P2 책임.
- **FP16 비트 포팅 fallback 라이브러리** — D-09 위험 발생 시 P4/P5에서 추가될 수 있음. 가능성 낮으면 cleanup. 위치 후보: `gtx/fp_strict.py` (별도 모듈로) 또는 `gtx/fp.py`에 옵션 함수.
- **`GtxNpu` 인스턴스에 `_memory: GtxMemory` 필드 노출 vs 캡슐화** — P2 결정. P1은 `GtxMemory`를 독립 클래스로만 만든다.

### Reviewed Todos (not folded)
None — `gsd-tools todo match-phase 1`은 매칭 없음.

### Out of scope for Phase 1 (다른 페이즈로)
- WRSPR/RDSPR 비즈니스 로직 → P2 SPR-01/SPR-02
- DMA op 핸들러 → P3
- DDR hex 파싱 (`ddr_init_from_file` 본체) → P3 DMA-04 (Phase 1은 stub만)
- MM gemm_core → P4
- 활성화 방향 비대칭 → P5
- verify.py 포팅 → P6

### Defer to user follow-up
- **상위 문서 동기화 작업**: D-07/D-08/D-09 결정에 따라 PROJECT.md / REQUIREMENTS.md / STATE.md / ROADMAP.md 업데이트 필요. discuss-phase 종료 후 즉시 일괄 처리 권장 (별도 커밋).

</deferred>

---

*Phase: 01-foundation*
*Context gathered: 2026-05-04*
