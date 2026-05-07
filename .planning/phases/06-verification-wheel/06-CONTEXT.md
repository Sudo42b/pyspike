# Phase 6: Verification & Wheel - Context

**Gathered:** 2026-05-07
**Status:** Ready for planning

<domain>
## Phase Boundary

Phase 6은 v1의 **ship gate**. Phase 1–5에서 만들어진 GTX NPU functional model을 **외부 사용자가 `pip install spike` 한 번으로 받아서 .elf 회귀를 strict-mode로 통과시킬 수 있도록 패키징**한다. 구체적으로:

1. **VRF-01: `verify.py` → `riscv.gtx._verify` 포팅** — vendor verify.py(388 LOC)의 argparse + verbose report 표면을 P4 `_verify_minimal.compare_hex`(78 LOC, 이미 검증됨) 코어 위에 하이브리드로 얹는다. importable + CLI(console_script `pyspike-verify` + `python -m riscv.gtx._verify`) 둘 다 노출.
2. **GTX_DDR_DUMP atexit hook (P5에서 P6로 미룬 미해결)** — Python 측에서 처음 구현. vendor C++ `gtx_npu_core.cc:61-127`의 `std::atexit(gtx_atexit_ddr_dump)` + 3개 env vars 시맨틱 직역. WJOIN의 `SystemExit(0)` flow에서 atexit이 정상 발화하여 `tests/gtx/test_regression_fw_act.py`가 graceful-skip 5단계를 벗어나 hard PASS로 전환됨.
3. **VRF-03: 회귀 .elf + golden hex 자산 lock-in** — vendor `test/<OP>/n1s16/data/{kernel}_ref.txt` (이미 캡처되어 있음)를 source로 가져와 P4/P5 `.hex` 포맷으로 변환. vendor C++ libgtx_npu.so 신규 빌드 zero. 코어 op 셋 ~10-20개 (MM/VEC/ACT/POOL/CVT 대표).
4. **VRF-04: 회귀 100% strict-mode pass** — `pytest tests/gtx/test_regression_fw_full.py`가 모든 .elf을 `@pytest.mark.parametrize`로 순회하여 zero failures + zero `within_tolerance` (모든 byte exact). vendor `n1s16_<op>.c` 1:1 직역 .elf은 GSPR-staged + per-op funct7 양 dispatch path를 자연스럽게 mix.
5. **PKG-01: package-data + 빌드 시점 복사** — `tests/gtx/data/` 단일 source-of-truth, setup.py custom build가 `src/main/python/riscv/gtx/data/{firmware,golden}/`로 복사. pyproject.toml `[tool.setuptools.package-data]`에 `riscv.gtx = ["data/firmware/*", "data/golden/*"]` 등록.
6. **PKG-03: clean cp310 venv install + 한 줄 import + `_verify` helper API** — `pip install dist/spike-*.whl` 후 `from riscv.gtx import GtxNpu`, `from riscv.gtx._verify import compare_hex, bundled_elfs, load_golden` 모두 동작. ROADMAP success #3의 `importlib.resources.files('riscv.gtx').joinpath('data','firmware').iterdir()`는 helper 내부 구현으로 노출.
7. **PKG-04: cibuildwheel cp310-cp312 manylinux2014 매트릭스 통과** — pre-existing matrix(P1 D-08에서 이미 잠김)에 P6 자산 등록만 추가하고 빌드 zero regression 확인. 50MB cap 우선 합침 검증; 근접 시 plan-stage에서 gzip 또는 extras split 재검토.

다음 모두는 **Phase 6 비범위(out-of-scope)** — 다른 페이즈/마일스톤이 다룬다:

- **Numba @njit 동적 최적화** → Phase 7 (P6에서 회귀 그린 후 진입; P5 D-01/D-02 stateless 커널이 P7 candidate boundary).
- **vendor 98개 op 디렉토리 전부 빌드** — 코어 op 셋 ~10-20개 외 op는 v1.x patch 또는 v2 (DMA-3D / IM2COL / MCAST는 PROJECT.md v2 영역).
- **gem5-simplified vs ISS-full 별도 빌드 매트릭스** — 단일 .elf이 양 dispatch path 자연 mix (P5 activation_relu_gelu.elf 패턴 직역). 명시적 분리는 v1.x patch 검토.
- **vendor C++ libgtx_npu.so 빌드 + CI 회귀 shadow run** — golden은 vendor 자산 직접 차용, vendor binary 신규 빌드 없음.
- **`_verify` Python idiomatic 재설계 / `--mode strict|tolerant`** — vendor argparse 1:1 호환 결정으로 reject. 사용자 마이그레이션 zero-friction 우선.
- **vendor `_ref.txt` 포맷 wheel 동봉** — 현 P4/P5 `.hex` 단일 파일 형식 유지. 변환 스크립트 one-shot at P6 dev-stage.
- **PCIe-EP / vfio-user / CUDA / OMP** — PROJECT.md Out of Scope (v2 reconsider).
- **Online shadow run vs C++ libgtx_npu.so during CI** — REQUIREMENTS.md Out of Scope: "검증은 오프라인 golden hex diff로만 수행".

</domain>

<decisions>
## Implementation Decisions

### `_verify` 모듈 표면 (D-01 ~ D-03)

- **D-01:** **하이브리드 base** — `riscv/gtx/_verify.py`는 P4 `tests/gtx/_verify_minimal.compare_hex` 78 LOC 코어를 그대로 흡수(이미 strict-mode 동작 검증됨, BE bit-pair compare per `vendor/.../verify.py:235`)하고, vendor `verify.py` 388 LOC 중 argparse/main/report printing 부분(~80 LOC)을 래퍼로 얹는다.
  - **이유:** `_verify_minimal`은 P4 04-05 + P5 strict-pass에서 동작 검증된 자산. vendor verify.py를 풀 직역하면 BE bit-pair compare 로직이 이중 구현되어 drift 위험.
  - **위험:** `_verify_minimal`이 stats dict의 어떤 키를 노출하는지가 vendor verbose report 포맷과 맞아야 함 — plan-stage에서 mapping 잠그기.

- **D-02:** **CLI 진입점 둘 다** — `pyproject.toml [project.scripts]`에 `pyspike-verify = riscv.gtx._verify:main` 등록 + `riscv/gtx/_verify.py` 모듈 끝에 `if __name__ == "__main__": main()` (또는 `__main__.py`) 추가.
  - **이유:** ROADMAP P6 success #1은 `python -m riscv.gtx._verify ...`를 명시, success #5는 `pyspike-verify` console script가 strict-mode PASS를 명시. 둘 다 만족.
  - **참고:** 현재 `pyproject.toml`에 `[project.scripts]` 섹션 자체가 없음 — P6에서 신설.

- **D-03:** **argparse vendor 1:1 호환 + `--strict` 신규** — vendor: `result.hex golden.hex [--ulp N] [--atol F] [--fp16]`. P6: 동일 + `--strict` 추가 (ROADMAP success #1 명시). default `--ulp 1 --atol 0.001`.
  - **이유:** Drop-in replacement. 기존 vendor 사용자 스크립트가 `pyspike-verify`로 그대로 마이그레이션. Python idiomatic 재설계는 zero-friction 원칙 위배.
  - **`--strict`의 시맨틱:** strict ON이면 `exact_matches == total_fp16` 만 PASS; OFF면 ULP/atol 슬랙 허용. P4 D-14 lineage.

### GTX_DDR_DUMP atexit hook (D-04 ~ D-06)

- **D-04:** **atexit 등록 = `riscv/gtx/__init__.py` import 시점, env var 조건부.**
  - 코드 모양: `import os, atexit; from .ddr import _atexit_ddr_dump; if os.getenv('GTX_DDR_DUMP'): atexit.register(_atexit_ddr_dump)`.
  - **이유:** vendor `gtx_npu_core.cc:127` `std::atexit(gtx_atexit_ddr_dump)` 직역. WJOIN의 `SystemExit(0)` flow에서도 Python `atexit`은 정상 발화 (interpreter shutdown hook). per-instance 등록이 아니어서 다중 NPU 등록 충돌 zero.
  - **검증 plan:** Wave 1a RED scaffold에 `test_atexit_dump_fires_on_systemexit` (subprocess pyspike 실행 후 dump 파일 존재 + 비-zero 크기) + `test_atexit_dump_skips_when_env_unset` 두 테스트.

- **D-05:** **dump 함수 본체 = `riscv/gtx/ddr.py`의 `_atexit_ddr_dump()` 래퍼.**
  - 본체는 P3 D-09에서 args-only로 묶인 `ddr_dump_to_file(addr, size, path)` 그대로 재사용. atexit 래퍼는 env vars 파싱(`GTX_DDR_DUMP` path, `GTX_DDR_DUMP_ADDR` hex addr, `GTX_DDR_DUMP_SIZE` hex size) → args 변환 → 기존 함수 호출.
  - **이유:** P3 D-09 "args-only" 결정은 그대로 보존(기존 `test_ddr_modes.py` 회귀 zero). atexit 래퍼는 env-var-aware 별도 function — 두 호출 경로 명확 분리.
  - **NPU 인스턴스 참조:** atexit 함수가 GtxNpu 인스턴스의 ddr 버퍼를 어떻게 잡을지 — `riscv.gtx.npu` module-level weakref 또는 PythonBridge에 등록된 인스턴스 lookup. plan-stage에서 정확화.

- **D-06:** **env vars vendor 1:1.**
  - 정확히 3개: `GTX_DDR_DUMP` (file path), `GTX_DDR_DUMP_ADDR` (hex int, e.g., "0x100"), `GTX_DDR_DUMP_SIZE` (hex int, e.g., "32"). 셋 모두 set 되어야 dump.
  - **이유:** vendor `gtx_npu_core.cc:66-69` 직역. 기존 P5 `test_regression_fw_act.py:117-118`이 이미 이 시맨틱으로 set 중 — P6에서 이 테스트가 5-tier graceful-skip을 벗어나 hard PASS로 전환되는 것이 D-04/D-05/D-06 acceptance signal.

### Regression matrix scope (D-07 ~ D-09)

- **D-07:** **회귀 .elf = 코어 op 셋 ~10-20개.** 정확 리스트는 plan-stage에서 잠금. 후보 영역 (vendor `test/` 디렉토리 매핑):
  - **MM**: GEMM 변형 1-2개 (mm_basic.elf 기존 + 1-2개 추가 후보 from vendor MUL_MAT/ADD_REL_POS).
  - **VEC**: SASMD 4 op (ADD/SUB/MUL/DIV) × IS/VS = 후보 ~4 .elf, DOT/VSUM/CLAMP 각 1-2.
  - **ACT**: RELU/SOFTMAX/ESUM forward + PRELU/GELU/TANH/SIGM reversed 양 방향 대표 ~4-5.
  - **POOL/CVT**: max/avg pool + FP16↔FP8 cvt 대표 ~3.
  - **이유:** ROADMAP success #2 "every bundled .elf 100% strict-mode pass"의 의미 있는 충족은 dispatch + compute coverage 충분한 코어셋. 50MB cap 안전 (각 .elf ~1-3KB + golden ~100B-1KB → 수MB).
  - **재고 시점:** v1.x patch에서 op 추가 또는 v2 milestone에서 vendor 98개 풀 sweep로 확장.

- **D-08:** **vendor `n1s16_<op>.c` 1:1 직역 단일 빌드.** 각 .elf이 GSPR-staged dispatch path + per-op funct7 dispatch path를 자연스럽게 mix (P5 `activation_relu_gelu.elf`가 forward funct7=0x06 GSPR-staged + reversed funct7=0x2A per-op을 단일 .elf에서 동시 검증한 패턴 직접 연장).
  - **이유:** ROADMAP "gem5-simplified + ISS-full encoding sweep" 표현은 historical artifact — vendor `test/`에는 `run_tests_n1s16.sh` 단일 스위트만 있고 `run_llext_tests.sh`는 존재하지 않음. 인코딩 분기는 vendor `intrin.h` / `gtx_csr.h` 매크로 레벨이지 별도 빌드 매트릭스가 아님.
  - **재고 시점:** P6 회귀 그린 후, 양 path를 명시적으로 분리해서 검증할 가치가 있다고 판단되면 v1.x patch (별도 .elf 빌드 매크로 도입). 사용자 명시 결정: "(a)로 일단하고 나중에 다시 볼게" — defer 보류.

- **D-09:** **단일 parametrize 롤 — `tests/gtx/test_regression_fw_full.py`.**
  - `@pytest.mark.parametrize('elf_path', sorted(BUNDLED_ELFS), ids=lambda p: p.stem)` — pytest test ID가 .elf 이름 그대로 노출되어 실패 격리 우수.
  - 기존 `test_regression_fw_mm.py`(P4) / `test_regression_fw_act.py`(P5)는 단일 high-stress sentinel로 유지(역사적 lineage + 빠른 smoke).
  - **이유:** P5 `test_regression_fw_act.py`의 5-tier graceful-skip + subprocess pyspike 패턴을 그대로 재사용. parametrize는 P6 신규 wrapper.
  - **stress test 시나리오:** 한 .elf 실패 시 다른 .elf 영향 zero (pytest-isolated subprocess), CI 리포트에서 어느 op이 깨졌는지 즉각 식별.

### Golden hex 출처 + 포맷 (D-10 ~ D-12)

- **D-10:** **golden source = vendor `test/<OP>/n1s16/data/{kernel}_ref.txt` (또는 `_result.hex`).**
  - vendor가 이미 ISS run 결과를 ref로 lock-in해 둔 자산을 그대로 차용. P6에서 vendor C++ libgtx_npu.so를 신규 빌드할 필요 zero.
  - **이유:** vendor 자산이 PROJECT.md Core Value의 ground-truth(C++ ↔ SystemC ULP 일치 검증 완료)를 거쳐 만들어진 결과. `--update-ref` 기록 그대로 차용.
  - **검증 단서:** `vendor/.../test/CONCAT/n1s16/data/n1s16_concat_result2.hex` 등 자산 실재 확인됨. 코어 op 셋 ~10-20개 모두 동일 디렉토리 패턴 자산 보유 가정 — plan-stage에서 누락 op 식별.

- **D-11:** **포맷 = 현 P4/P5 `<elf>.hex` 단일 파일 형식 유지.** vendor `_ref.txt` (또는 `_result.hex`)를 P6 dev-stage 변환 스크립트가 `<elf>.hex`로 변환하여 `tests/gtx/data/golden/`에 lock-in.
  - 기존 자산: `tests/gtx/data/golden/mm_basic_n1s16.hex`, `activation_relu_gelu.hex` (둘 다 zero-init oracle, P4/P5에서 작동).
  - 변환 스크립트: P6 plan-stage에서 작성. `vendor/.../test/<OP>/n1s16/data/*ref.txt` → 32-byte/line `<elf>.hex` 변환 + git에 lock-in.
  - **이유:** `_verify_minimal.compare_hex`(D-01의 코어)가 이미 `.hex` 포맷 소비 중. P4/P5 자산과 형식 일관성. vendor 포맷 직접 사용 시 _verify 또는 자산 재작업 필요.

- **D-12:** **vendor C++ 빌드 의존 = 개발 시점 only.**
  - P6 dev-stage에 vendor `_ref.txt` → `.hex` 변환 1회 수행 → 결과 자산 git lock-in. cibuildwheel CI에는 변환된 자산만 동봉, vendor C++ 빌드 zero.
  - **이유:** PROJECT.md "vendor/gtx_cpp_reference는 wheel 미포함". CI에서 vendor C++ 빌드는 시간/복잡도 폭증. 개발자 머신 한 번 + lock-in이 가장 실용적.
  - **자산 갱신:** vendor 업데이트(`git submodule update`) 시 dev-stage 변환 스크립트 재실행으로 자산 refresh. plan-stage에서 자동화 스크립트 정의.

### Wheel 자산 위치 + 사이즈 (D-13 ~ D-15)

- **D-13:** **자산 위치 = 빌드 시점 복사.** `tests/gtx/data/{elf,golden}/` 단일 source-of-truth, `setup.py` custom `build_py` 또는 `MANIFEST.in`이 wheel 빌드 시 `src/main/python/riscv/gtx/data/{firmware,golden}/`로 복사.
  - **이유:** drift zero. tests/ 자산이 단일 truth, src/ 자산은 wheel-only 사본. 개발자가 파일 두 곳 동기화하는 부담 없음.
  - **위험:** setup.py custom build hook이 cibuildwheel과 호환되는지 plan-stage에서 검증. 대안: `MANIFEST.in include` + `[tool.setuptools.package-data]` 직접 포함도 가능 (자산을 `src/`에 두지 않고도 wheel include 가능한지 setuptools 동작 확인).
  - **package-data 등록:** `riscv.gtx = ["data/firmware/*", "data/golden/*"]` 추가. 디렉토리 이름은 `firmware` (ROADMAP success #3 `joinpath('data','firmware')`), `golden` 신규.

- **D-14:** **자산 접근 API = `_verify` helper.**
  - `riscv.gtx._verify` 모듈에 `bundled_elfs() -> list[Path]` (wheel에 동봉된 모든 .elf 경로 list 반환), `load_golden(name: str) -> bytes` (golden hex 바이트 로드) 두 helper 노출.
  - 사용자가 `importlib.resources` 직접 다루지 않음. ROADMAP success #3의 `r.files('riscv.gtx').joinpath('data','firmware').iterdir()` 패턴은 helper 내부 구현으로만 등장 (docstring 예제로 보존).
  - **이유:** 사용자 친화적. importlib.resources API는 Python 3.9/3.10/3.11/3.12 사이 시그니처 변동 있음 (legacy `.path()` deprecation 등) — helper가 버전 추상화.

- **D-15:** **wheel size = 우선 합침 검증.** P6 plan-stage에서 코어셋 ~10-20개 자산 합산 측정. 50MB 근접 시(예상 1MB 미만이지만 마진 확보) gzip 압축 또는 `spike[gtx-regression]` extras split 재검토.
  - **현 예상:** .elf 각 ~1-3KB × 20 = ~60KB; golden ~100B-1KB × 20 = ~20KB. 총 ~80KB-수MB. 50MB cap 대비 충분히 fit.
  - **재고 트리거:** plan-stage에서 wheel size 1MB 초과 시 압축 우선; 10MB 초과 시 extras split 검토; 50MB 근접 시 ROADMAP success #4 위반 — 코어셋 축소.

### Wave / Plan 분할 (D-16 ~ D-18)

- **D-16:** **5 plans / 3 wave 구조** (P5 lineage 직접 mirror, plan 1개 줄임).
  - **Wave 1a (parallel, 3 plans):**
    - **Plan 01:** VRF-01 `riscv/gtx/_verify.py` 신규 (하이브리드 base + argparse + console_script entry). 의존: 없음.
    - **Plan 02:** atexit hook (`riscv/gtx/ddr.py` `_atexit_ddr_dump()` 래퍼 + `riscv/gtx/__init__.py` 등록 logic). 의존: 없음.
    - **Plan 03:** VRF-03 vendor asset import (변환 스크립트 `scripts/import_vendor_golden.py` + `tests/gtx/data/golden/{op}.hex` 자산 lock-in for 코어 op 셋 ~10-20개 + 코어 op 셋 .elf 빌드 또는 vendor 차용). 의존: 없음.
  - **Wave 1b (sequential, 1 plan):**
    - **Plan 04:** VRF-04 regression matrix (`tests/gtx/test_regression_fw_full.py` parametrize 롤 + 코어 op 셋 단일 패스 strict-mode pass). 의존: Plan 01 (_verify) + Plan 02 (atexit) + Plan 03 (assets) 세 개 모두 GREEN.
  - **Wave 2 (1 plan):**
    - **Plan 05:** PKG-01/03/04 통합 (build-time 복사 hook + pyproject.toml `[tool.setuptools.package-data]` + `[project.scripts]` 등록 + clean cp310 venv smoke test + cibuildwheel cp310-cp312 매트릭스 통과 검증). 의존: Plan 04 GREEN (회귀가 wheel-시점에서도 동작 보장).
  - **이유:** P6 5개 작업 단위가 자연스럽게 3-wave 의존 그래프 형성. PKG-01/03/04를 단일 plan으로 묶은 이유: 모두 packaging 도메인 (pyproject.toml + setup.py + cibuildwheel test command)이고, build-time 복사 hook이 cibuildwheel과 호환되는지가 PKG 전체의 acceptance 단일점.

- **D-17:** **Wave 1a uniform RED scaffold + 후속 GREEN-fill** (P4/P5 lineage 직접 mirror).
  - Wave 1a 첫 plan (or pre-wave scaffold)이 P6 모든 신규 테스트 stub을 RED로 깐다:
    - `test_verify_compare_hex_strict` / `test_verify_compare_hex_tolerant` / `test_verify_cli_help` / `test_verify_console_script_smoke` (Plan 01 GREEN-fill)
    - `test_atexit_dump_fires_on_systemexit` / `test_atexit_dump_skips_when_env_unset` (Plan 02 GREEN-fill)
    - `test_regression_fw_full[<op>]` parametrize stub (Plan 03 + Plan 04 GREEN-fill)
    - `test_wheel_install_one_liner` / `test_wheel_data_present` (Plan 05 GREEN-fill)
  - **scaffold 위치:** Plan 01 또는 별도 pre-Wave 1a "Plan 00 scaffold" (P3/P4/P5에서 첫 plan이 통합 scaffold 책임).
  - **이유:** P4/P5에서 검증된 패턴. 모든 P6 acceptance criteria가 RED → GREEN 명시적 lock-in.

- **D-18:** **Wave 1a 3 plans 병렬 실행 허용.**
  - 편집 면적 zero overlap: Plan 01은 `riscv/gtx/_verify.py` 신규 파일만, Plan 02는 `riscv/gtx/ddr.py` + `riscv/gtx/__init__.py` 뚜렷한 라인 추가, Plan 03은 `scripts/` + `tests/gtx/data/golden/`만.
  - 공통 편집 가능 파일: `pyproject.toml` (Plan 01의 `[project.scripts]` + Plan 05의 `[tool.setuptools.package-data]`) — 다른 wave라서 충돌 zero.
  - **이유:** P5 D-04 wave structure 직접 mirror. 사용자 시간 절약 + P3 04-04 lineage(편집 면적 겹칠 시 sequential)의 반례 (P6 Wave 1a는 자연 분리).
  - **commit 정책:** 각 Plan parallel agent가 `git commit --no-verify` 사용 (P3 03-01 D-7 lineage). 오케스트레이터가 wave 종료 후 hooks 1회 검증.

### Claude's Discretion

다음은 implementation detail로 Claude 판단 (research/plan 단계에서 정확화):

- vendor `_ref.txt` 정확 포맷 (32-byte/line vs 64-byte/line vs other) — 변환 스크립트가 dev-stage에서 한 번 해독.
- `setup.py` custom `build_py` vs `MANIFEST.in include` vs `[tool.setuptools.package-data]` 단독 — cibuildwheel 호환성 검증 후 가장 단순한 path 채택.
- atexit hook의 NPU 인스턴스 lookup 메커니즘 (`weakref.WeakValueDictionary` in `riscv.gtx.npu` module + `__init__.py`에서 등록 시 references) — plan-stage에서 PythonBridge와의 충돌 검증.
- `_verify.compare_hex` stats dict의 키 명세 (vendor verbose report와 호환되도록 필드 정렬) — plan-stage에서 vendor `verify.py` 출력 포맷 1:1 비교.
- 코어 op 셋 정확 리스트 (~10-20개) — plan-stage에서 vendor `test/<OP>/n1s16/data/` 자산 보유 확인 + dispatch coverage(GSPR-staged + per-op funct7 path 모두 exercise) 검증 후 lock-in.
- vendor `_ref.txt` → `.hex` 변환 스크립트 위치 (`scripts/import_vendor_golden.py` vs `tests/gtx/conftest.py` 헬퍼 vs 별도) — plan-stage에서 결정.
- Wave 1a Plan 03의 .elf 빌드 정책 (vendor 빌드 산출물 차용 vs P6에서 새로 빌드) — vendor 자산 보유 여부에 따라 plan-stage 분기.
- Plan 05의 cibuildwheel test 매트릭스 verification 방법 (`pip install dist/*.whl` in fresh venv + smoke import + `pyspike-verify --version` smoke) — P1 D-08 cibuildwheel 매트릭스 그대로 활용.

### Folded Todos

None — `gsd-tools todo match-phase 6`에서 매칭 0건.

</decisions>

<canonical_refs>
## Canonical References

**Downstream agents MUST read these before planning or implementing.**

### Vendor C++ source (operational truth — bit-exact 직역 대상)

- `vendor/gtx_cpp_reference/gtx/verify.py` — **VRF-01 직역 source.** 388 LOC. argparse + main() + report printing 부분 (~80 LOC)이 D-01 하이브리드 wrapper의 source.
- `vendor/gtx_cpp_reference/gtx/verify_ref.py` — **P5 VRF-02 source** (이미 P5에서 흡수, P6 비포함). 32-op host-side oracle.
- `vendor/gtx_cpp_reference/gtx/gtx_npu_core.cc` — **atexit hook 직역 source.**
  - **`gtx_atexit_ddr_dump()` 함수 본체:** lines 61-74 (env vars 파싱 + dump 로직).
  - **`std::atexit(gtx_atexit_ddr_dump)` 등록:** line 127 (D-04의 직역 대상).
- `vendor/gtx_cpp_reference/gtx/gtx_npu_loop.cc:55-58` — atexit과 별도로 WJOIN 루프에서 GTX_DDR_DUMP 처리하는 historical path (D-04 주의 — 이건 직역하지 않음, atexit hook이 single source-of-truth).
- `vendor/gtx_cpp_reference/gtx/gtx_npu_custom1.cc:116-117` — WJOIN의 DDR dump가 atexit으로 이전된 vendor 의도 코멘트 ("DDR dump moved to atexit handler... once at exit, not per WJOIN.") D-04 정당화 source.
- `vendor/gtx_cpp_reference/gtx/CLAUDE.md` — vendor-side guidelines.

### 레퍼런스 회귀 자산 (P6 plan-stage 식별 + 차용)

- `vendor/gtx_cpp_reference/test/run_tests_n1s16.sh` — **회귀 빌드 + 실행 메커니즘 source.** 18.3KB. `--generate KERNEL` Python script invocation, `--update-ref` ref lock-in flow, `KERNEL_DIR` mapping (98 op).
- `vendor/gtx_cpp_reference/test/generate_n1s16_tests.py` — vendor input/ref 생성 스크립트 (P6 plan-stage에서 변환 스크립트 작성 시 참고).
- `vendor/gtx_cpp_reference/test/<OP>/n1s16/data/{kernel}_ref.txt` — **D-10 golden source.** vendor가 이미 ISS-captured해 lock-in한 ref. P6 코어 op 셋 ~10-20개에 해당하는 디렉토리만 차용. e.g., `vendor/.../test/CONCAT/n1s16/data/n1s16_concat_result2.hex` 패턴 확인됨.
- `vendor/gtx_cpp_reference/test/<OP>/n1s16/n1s16_<op>.c` — **D-08 .elf source.** 1:1 직역 대상. vendor `intrin.h` / `gtx_csr.h` 매크로 통해 자동 dispatch path mix.
- `vendor/gtx_cpp_reference/test/CLAUDE.md` (있다면) — vendor test 디렉토리 가이드.

### Project documents (locked context)

- `.planning/PROJECT.md` — Core Value (bit-exact w/ C++ libgtx_npu.so), Constraints (NumPy ≥ 2.0 + cp310-cp312), Out of Scope (CUDA / OMP / vfio-user / GTX commitlog), Validated requirements (P1-P5 모두 closed로 P6의 base).
- `.planning/REQUIREMENTS.md` — VRF-01, VRF-03, VRF-04, PKG-01, PKG-03, PKG-04 v1 acceptance criteria (P6 6 requirements).
- `.planning/ROADMAP.md` Phase 6 섹션 — 5 success criteria 명시. 특히 success #1 `python -m riscv.gtx._verify --strict`, success #2 "every bundled .elf 100% strict", success #3 `importlib.resources.files('riscv.gtx').joinpath('data','firmware')`, success #4 cp310-cp312 + 50MB, success #5 `pyspike-verify` console script.
- `.planning/STATE.md` — 현재 진행 (P1-P5 완료; P6 ready to plan).

### Prior phase contexts (decision precedent)

- `.planning/phases/01-foundation/01-CONTEXT.md` — **D-04/D-06 (vendor submodule policy, wheel 미포함), D-07/D-08 (NumPy ≥ 2.0 + cp310-cp312 cibuildwheel), D-15 (tests/gtx/ test layout).** P6 D-12/D-13/D-15의 lineage source.
- `.planning/phases/03-dma-ddr-i-o/03-CONTEXT.md` — **D-09 `ddr_dump_to_file` args-only.** P6 D-05의 직접 부모 — 새 atexit 래퍼는 args-only `ddr_dump_to_file`을 그대로 wrap.
- `.planning/phases/04-mm-subsystem/04-CONTEXT.md` — D-13 (`_verify_minimal` mini-port 패턴), D-14 (strict mode), D-15 (op-level np.array_equal). P6 D-01/D-03 lineage.
- `.planning/phases/04-mm-subsystem/04-VERIFICATION.md` — `proc.state` is property (not method) per pybind11 binding `src/main/cpp/py_module.cc:711`. **P6의 모든 spike-bound code (특히 atexit hook의 npu lookup)는 `proc.state` 사용 필수.**
- `.planning/phases/05-vec-act-pool/05-CONTEXT.md` — D-04 (Wave structure), D-09 (VSUM dual-mode lineage). P6 D-16/D-17 wave structure 직접 mirror. P5 deferred items (production verify CLI, atexit hook, full regression matrix, package-data) = P6의 입력.

### Code context (existing pyspike assets to mirror or extend)

- `tests/gtx/_verify_minimal.py` — **D-01 코어 source.** 78 LOC, BE bit-pair compare. P6에서 그대로 흡수. **변경 시 P4/P5 회귀 깨짐 — 신중히 다룰 것.**
- `tests/gtx/test_regression_fw_mm.py` (P4) — high-stress sentinel for MM. 유지.
- `tests/gtx/test_regression_fw_act.py` (P5) — high-stress sentinel for ACT + 5-tier graceful-skip 패턴. **P6 D-04/D-05/D-06의 acceptance signal: 이 테스트가 5-tier skip을 벗어나 hard PASS로 전환됨.**
- `tests/gtx/data/elf/Makefile` — `mm_basic.elf`, `activation_relu_gelu.elf`, `nop_wjoin.elf` 빌드 룰. P6 코어 op 셋 추가 시 새 룰 추가.
- `tests/gtx/data/golden/` — `mm_basic_n1s16.hex`, `activation_relu_gelu.hex` (zero-init oracle 기존 자산). P6 D-11 포맷 직접 lineage.
- `src/main/python/riscv/gtx/__init__.py` — D-04 atexit 등록 위치. P6 plan 02에서 수정.
- `src/main/python/riscv/gtx/ddr.py` — D-05 `_atexit_ddr_dump()` 추가 위치. P3 D-09의 `ddr_dump_to_file(args-only)` 직접 옆.
- `src/main/python/riscv/gtx/npu.py` — `GtxNpu` 인스턴스 + ddr 버퍼 lookup mechanism (D-05 plan-stage 정확화).
- `pyproject.toml` — `[tool.cibuildwheel]` (P1 D-08 잠김), `[tool.setuptools.package-data]` (D-13 추가), `[project.scripts]` (D-02 신설).
- `setup.py` — D-13 build-time 복사 hook 후보 위치.
- `MANIFEST.in` — D-13 include 패턴 추가 후보.
- `scripts/` — D-12 vendor → `.hex` 변환 스크립트 위치 (P6 plan-stage에서 신설).

</canonical_refs>

<code_context>
## Existing Code Insights

### Reusable Assets

- **`tests/gtx/_verify_minimal.compare_hex`** (P4 78 LOC) — D-01 하이브리드 base의 코어. BE bit-pair compare per `vendor/.../verify.py:235`. signed-magnitude ULP per `vendor/.../verify.py:150-158`. strict mode per P4 D-14. **변경 시 P4 mm_basic.elf strict-pass + P5 activation_relu_gelu.elf strict-pass 동시 회귀 — extreme caution.**
- **`tests/gtx/test_regression_fw_act.py`** (P5) — 5-tier graceful-skip + subprocess pyspike + `GTX_DDR_DUMP` env vars set + `compare_hex(strict=True)` 패턴. P6 D-09 `test_regression_fw_full.py`가 이 패턴 직접 일반화.
- **`tests/gtx/conftest.py`** — `_RISCV_AVAILABLE` 검출 fixture. P6 신규 테스트도 동일 import 패턴.
- **`src/main/python/riscv/gtx/ddr.py`** — `ddr_dump_to_file(addr, size, path)` args-only (P3 D-09). D-05 atexit 래퍼가 이를 그대로 wrap — 본체 변경 zero.
- **`vendor/gtx_cpp_reference/test/<OP>/n1s16/data/{kernel}_ref.txt`** (예: CONCAT 등 다수 op) — D-10 golden source. P6 코어 op 셋이 보유한 디렉토리만 plan-stage에서 식별.
- **`vendor/gtx_cpp_reference/test/run_tests_n1s16.sh`** — D-12 변환 스크립트의 vendor flow 참조.

### Established Patterns

- **`_verify_minimal` mini-port → production 승격** (P4 D-13 → P6 D-01) — 검증된 mini-port 코드를 production 모듈로 흡수하는 패턴 직접 lineage.
- **5-tier graceful-skip discipline** (P5 `test_regression_fw_act.py:23-27`) — P6 D-09에서 hard PASS로 전환되는 마지막 ply.
- **Wave 1a uniform RED scaffold** (P4 04-01 + P5 05-01) — D-17의 직접 모델.
- **Wave 1a parallel 3 plans** (P5 D-04 wave structure) — D-18의 직접 모델.
- **vendor C++ 1:1 직역 + LE byte-order invariant** (P1-P5 invariant) — atexit hook + golden 변환 스크립트 모두 동일 규약 준수.
- **args-only 함수 + env-var-aware 래퍼 분리** (P3 D-09 → P6 D-05) — 함수 표면이 단일 책임 유지.

### Integration Points

- **`riscv/gtx/__init__.py`** — D-04 atexit 등록 logic 추가 (Plan 02). 기존 import 순서와 충돌하지 않도록 module-level effect는 try/except 가드.
- **`pyproject.toml [project.scripts]`** — D-02 `pyspike-verify` 신설 (Plan 01).
- **`pyproject.toml [tool.setuptools.package-data]`** — D-13 `riscv.gtx = ["data/firmware/*", "data/golden/*"]` 추가 (Plan 05).
- **`setup.py`** — D-13 custom `build_py` hook (Plan 05). cibuildwheel 호환성 plan-stage에서 검증.
- **`MANIFEST.in`** — D-13 include 패턴 (Plan 05의 대안 또는 보조).
- **`tests/gtx/data/{firmware,golden}/`** — D-13 source-of-truth (이미 P3/P4/P5에서 사용 중인 위치). Plan 03이 새 자산 lock-in.
- **`tests/gtx/test_regression_fw_full.py`** (신규) — D-09 parametrize 롤. Plan 04에서 GREEN-fill.
- **`scripts/import_vendor_golden.py`** (신규) — D-12 변환 스크립트. Plan 03에서 신설 + dev-stage 1회 실행 + 자산 git lock-in.

### Creative Options Enabled / Constrained

- **하이브리드 `_verify` 가능 (D-01)** — P4 mini-port가 이미 검증되어 있어 vendor verify.py 풀 직역 비용 회피. C++에는 이런 incremental promotion이 어려움.
- **Python `atexit` + `SystemExit(0)` 호환 (D-04)** — Python interpreter shutdown이 SystemExit에서도 atexit hooks 발화 (vendor C++ `std::atexit`보다 더 명시적). P2 CORE-03 `SystemExit(0)`과 자연 호환.
- **vendor 자산 직접 차용 (D-10)** — vendor가 이미 ISS run 결과를 ref로 lock-in해 둔 점이 v1 P6의 가장 큰 시간 절약. v2에서 자체 generate 시 generate_n1s16_tests.py 직접 활용.
- **Constraint: 50MB cap** (PROJECT.md "wheel size ≤50MB" + ROADMAP success #4) — D-15 우선 합침 검증 → 근접 시 gzip / extras split.
- **Constraint: vendor 빌드 zero in CI** (D-12 + P1 D-06) — golden은 dev-stage lock-in only. CI가 vendor 빌드 안 하므로 cibuildwheel 시간 안정.
- **Constraint: Python 3.10+ (P1 D-08)** — `match` 문 사용 가능, `importlib.resources.files()` 표준 (3.9 deprecation). D-14 helper API가 자연 안정.

</code_context>

<specifics>
## Specific Ideas

### "ROADMAP success #2 직접 호환" 가이드라인 (D-09 설계 핵심)

ROADMAP success #2: `pytest tests/gtx/test_regression_fw.py with parametrize over every bundled .elf ... reports zero failures and zero within_tolerance matches (every byte exact).`

D-09는 이를 그대로 만족: parametrize 롤 + strict-mode + zero `within_tolerance`. plan 04 acceptance command는 **literally** `pytest tests/gtx/test_regression_fw_full.py -v` + 모든 .elf 통과 + `assert stats['within_tolerance'] == 0` (D-01의 strict 의미).

### vendor `_ref.txt` 포맷 — 변환 스크립트 작성 가이드

vendor pattern (run_tests_n1s16.sh:get_output_size 추론):
```
@<addr>      # 주석/지시자 (변환 시 무시)
<32-byte hex line per row>  # 데이터 라인 — 32 byte = 16 FP16 (BE bit-pair)
```

P4/P5 `.hex` 포맷 (`_verify_minimal._parse_hex` 소비 형식):
```
@<addr>      # 주석/지시자 (변환 시 무시) — 동일
<hex bytes any whitespace>  # 32-byte/line OR 16-byte/line (P3 D-09 half-density carve-out)
```

→ vendor `_ref.txt`와 P4/P5 `.hex`는 사실상 같은 포맷. 변환은 단순 copy + name rewrite (`<kernel>_ref.txt` → `<elf>.hex`). plan-stage에서 실측 후 정확화.

### atexit hook의 NPU 인스턴스 lookup 패턴 후보 (D-05 plan-stage 정확화)

**Option A: Module-level WeakValueDictionary** (plan-stage 추천)
```python
# riscv/gtx/npu.py
import weakref
_NPU_INSTANCES: weakref.WeakValueDictionary = weakref.WeakValueDictionary()

class GtxNpu(...):
    def __init__(self, ...):
        ...
        _NPU_INSTANCES[id(self)] = self
```
```python
# riscv/gtx/ddr.py
def _atexit_ddr_dump():
    for npu in _NPU_INSTANCES.values():
        if npu and npu.mem is not None:
            ddr_dump_to_file(...)
```

**Option B: PythonBridge `references` 활용** (vendor C++ 직역에 더 가까움)
- `src/main/cpp/py_bridge.h:78` `PythonBridge::references` map에 모든 GtxNpu 인스턴스 등록되어 있음.
- atexit hook이 PythonBridge에 접근해 lookup. 단점: PythonBridge가 Python에서 expose되어야 함 (현재 미확인).

**Option C: 단일 글로벌 인스턴스 가정 (가장 단순)**
- `riscv.gtx`는 단일 hart에 하나의 NPU만 등록한다는 가정. `_LAST_NPU = None` + `__init__`에서 set.
- vendor C++ pattern 직역 (vendor도 process-global). 단점: 다중 hart 시나리오 미대응 (v1에서는 1 hart only).

plan-stage에서 P2 + P3 코드 + 벤더 동작 직접 비교 후 1개 채택.

### `pyspike-verify` console_script naming convention

vendor `verify.py`의 CLI는 `python3 verify.py result.hex golden.hex` — 이름 자체에 "verify"가 들어감. 사용자가 마이그레이션할 때 가장 자연스러운 이름은 `pyspike-verify` (pyspike CLI 패밀리 + verify 행위 명시). 대안 `riscv-verify`나 `gtx-verify`도 있지만 zero-friction 마이그레이션 가치 낮음. D-02 lock-in.

### "5-tier graceful-skip → hard PASS" 전환 조건 명시 (P5 → P6 acceptance signal)

P5 `test_regression_fw_act.py` 5-tier graceful-skip:
1. `_riscv.so` missing — skip (P1/P2 종속, P6 영향 없음)
2. `activation_relu_gelu.elf` missing — skip (P5 fixture 종속, P6 영향 없음)
3. `activation_relu_gelu.hex` golden missing — skip (P5 자산, P6 영향 없음)
4. `pyspike` CLI not on PATH — skip (P2 CLI 종속, P6 영향 없음)
5. **Subprocess clean-exits but no dump produced — skip (P6 atexit territory)** ← **P6 D-04/D-05/D-06가 이 5단계를 hard PASS로 전환.**

Wave 1a Plan 02 (atexit hook) 완료 후 P5 회귀가 5-tier #5에서 strict compare로 진입함을 verify. 만약 진입 안 하면 D-04/D-05/D-06 implementation 결함 — plan 02 acceptance 미달.

</specifics>

<deferred>
## Deferred Ideas

### Out of P6 scope (explicit deferrals to other phases / milestones)

- **Numba @njit 동적 최적화** → Phase 7 (P6 회귀 그린 후 진입; P5 D-01/D-02 stateless kernels = trivial @njit boundary).
- **vendor 98개 op 디렉토리 풀 sweep** → v1.x patch 또는 v2 (P6는 코어 op 셋 ~10-20개; DMA-3D / IM2COL / MCAST는 PROJECT.md v2 영역).
- **gem5-simplified vs ISS-full 별도 빌드 매트릭스 분리** → v1.x patch 검토 (사용자 명시: "(a)로 일단하고 나중에 다시 볼게"). 단일 빌드로 양 dispatch path 자연 mix가 P6 default.
- **vendor C++ libgtx_npu.so CI 회귀 shadow run** → REQUIREMENTS.md Out of Scope ("검증은 오프라인 golden hex diff로만"). v2 검토.
- **`_verify` Python idiomatic 재설계 (`--mode strict|tolerant`)** → v2. P6는 vendor argparse 1:1 호환.
- **vendor `_ref.txt` 포맷 wheel 동봉 (포맷 다양화)** → 우선 P4/P5 `.hex` 단일 포맷. vendor 포맷 추가 지원은 P6 plan-stage에서 _verify가 자동 detect 할지 결정.
- **`PYSPIKE_LIBS` 동적 NPU 선택** → 기존 PYSPIKE_LIBS 메커니즘 그대로. 신규 NPU 등록은 사용자 확장 영역.
- **PCIe-EP / vfio-user / CUDA / OMP / cuBLAS / GTX commitlog** → PROJECT.md Out of Scope (v2 reconsider).
- **MMIO Python 디바이스 모델로 NPU 노출** → v2 PY-OVRD-01 / PY-FUNCT7-01 영역.

### Within-domain ideas surfaced but not selected for discussion

- **`_verify` API surface — Verifier class vs functional API** — D-01 하이브리드는 functional (`compare_hex`, `bundled_elfs`, `load_golden` 단일 함수). class-based wrapper는 v2 검토 (state ful diff visualization 등).
- **`pyspike-verify --json` output mode** — vendor verify.py에 없음. P6 plan-stage에서 CI 친화적 옵션 추가 검토 가능. D-03 1:1 호환 보장 후 추가.
- **Wheel에 `tests/gtx/data/elf/Makefile` 동봉** — wheel 사용자가 .elf을 재빌드할 일이 없음 (riscv toolchain 없음). 미포함이 default. 개발자는 git checkout로 접근.
- **atexit hook의 dump format 옵션** — `GTX_DDR_DUMP_FORMAT={hex,bin,ref}` 같은 확장 — vendor에 없음. P6 1:1 호환 우선, 확장은 v2.
- **자산 hash check** (wheel 변조 검출) — sha256 verification at import time. v2 PY-VIEW-01 검토. P6 default off.
- **VRF-04 cross-verify against vendor live shadow run** — 수동 확인용 stub script. v2.

### Reviewed Todos (not folded)

None — `gsd-tools todo match-phase 6`은 매칭 0건.

### Defer to user follow-up

- **상위 문서 동기화 작업**: P6 완료 후 PROJECT.md `Active` → `Validated` 이동 (VRF-01/03/04, PKG-01/03/04 모두), STATE.md `last_updated`, ROADMAP.md Phase 6 progress table — `/gsd:complete-milestone v1.0` 흐름에 자동 처리 예상.
- **v1.0 ship announcement** — PR 머지 후 `pip install spike` 검증 완료 상태에서 사용자가 직접 announce.

</deferred>

---

*Phase: 06-verification-wheel*
*Context gathered: 2026-05-07*
