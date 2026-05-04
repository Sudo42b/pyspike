---
phase: 01-foundation
plan: 01
type: execute
wave: 1
depends_on: []
files_modified:
  - src/main/python/riscv/gtx/__init__.py
  - src/main/python/riscv/gtx/params.py
  - src/main/python/riscv/gtx/encoding.py
  - src/main/python/riscv/gtx/ops/__init__.py
  - tests/gtx/__init__.py
autonomous: true
requirements:
  - FOUND-03
must_haves:
  truths:
    - "`from riscv.gtx import fp, memory, params, encoding, ddr` succeeds (after Wave 1 모든 모듈 존재 시)"
    - "`from riscv.gtx.params import GTX_NEST_NUM, GTX_SPU_NUM, GTX_L1_SIZE_BYTES` succeeds; (GTX_NEST_NUM, GTX_SPU_NUM) == (4, 16); GTX_L1_SIZE_BYTES == 384*1024"
    - "Non-LE host (sys.byteorder != 'little')에서 `import riscv.gtx`가 명시적 RuntimeError를 raise"
    - "`riscv.gtx.ops` 패키지가 import 가능 (P2~P5에서 op 핸들러 채워질 placeholder)"
    - "`tests/gtx/` 디렉토리가 pytest collection에 자동 포함됨 (testpaths=['tests'] 재귀 발견)"
  artifacts:
    - path: "src/main/python/riscv/gtx/__init__.py"
      provides: "riscv.gtx 패키지 진입점, 서브모듈 re-export, LE byte-order 가드"
      contains: "from . import fp"
    - path: "src/main/python/riscv/gtx/params.py"
      provides: "HW 토폴로지 + 메모리 사이즈 + SPR base 상수 (D-13, gtx_params.h 포팅)"
      contains: "GTX_NEST_NUM"
    - path: "src/main/python/riscv/gtx/encoding.py"
      provides: "Phase 1 scope: funct7 상수만 (full disasm.inc은 P2)"
      contains: "GTX_F7_WRSPR"
    - path: "src/main/python/riscv/gtx/ops/__init__.py"
      provides: "ops 서브패키지 마커 (P2~P5에서 채움)"
    - path: "tests/gtx/__init__.py"
      provides: "pytest collection marker (라이선스 헤더만)"
  key_links:
    - from: "src/main/python/riscv/gtx/__init__.py"
      to: "sys.byteorder"
      via: "module-load tripwire"
      pattern: "sys.byteorder"
    - from: "src/main/python/riscv/gtx/__init__.py"
      to: "riscv.gtx.{fp,memory,params,encoding,ddr}"
      via: "from . import …"
      pattern: "from \\. import"
---

<objective>
`riscv.gtx` 패키지의 정적 스켈레톤을 만든다 — `__init__.py`, `params.py`,
`encoding.py`, `ops/__init__.py`, 그리고 `tests/gtx/__init__.py` 테스트 마커.
이 plan은 Wave 1의 다른 두 plan(`02-fp`, `03-memory`)이 의존하지 않는 정적 자산만
다룬다. `fp.py`, `memory.py`, `ddr.py`는 Wave 1의 별도 plan들이 작성한다.

Purpose: D-13/D-14 모듈 레이아웃 lock-in. Phase 2~5 op 핸들러가 import할 수 있는
import 경로 + HW 상수 + funct7 상수 확보. LE 가드는 비-x86_64 host에서 `np.float16
view`가 깨지는 것을 방지한다 (RESEARCH.md "Anti-Patterns").

Output: `src/main/python/riscv/gtx/{__init__.py, params.py, encoding.py, ops/__init__.py}`
+ `tests/gtx/__init__.py`. 이 파일들이 존재해야 Wave 1의 다른 plan과
Wave 2 packaging이 동작한다.
</objective>

<execution_context>
@$HOME/.claude/get-shit-done/workflows/execute-plan.md
@$HOME/.claude/get-shit-done/templates/summary.md
</execution_context>

<context>
@.planning/PROJECT.md
@.planning/ROADMAP.md
@.planning/STATE.md
@.planning/REQUIREMENTS.md
@.planning/phases/01-foundation/01-CONTEXT.md
@.planning/phases/01-foundation/01-RESEARCH.md
@.planning/phases/01-foundation/01-VALIDATION.md
@CLAUDE.md
@tests/__init__.py
</context>

<tasks>

<task type="auto" tdd="false">
  <name>Task 01-01: riscv.gtx 패키지 진입점 + LE 가드</name>
  <files>src/main/python/riscv/gtx/__init__.py</files>
  <read_first>
    - .planning/phases/01-foundation/01-CONTEXT.md (D-13, D-14 module layout lock; D-09 LE byte-order 가정)
    - .planning/phases/01-foundation/01-RESEARCH.md §"Pattern 4: Package skeleton + namespace re-export" (정확한 import 패턴) + §"Anti-Patterns to Avoid" (LE tripwire)
    - src/main/python/riscv/__init__.py (기존 namespace 패턴 — `riscv/__init__.py`는 변경하지 않음)
    - tests/__init__.py (라이선스 헤더 스타일 — `tests/gtx/__init__.py`는 같은 톤으로)
  </read_first>
  <action>
    `src/main/python/riscv/gtx/__init__.py` 파일을 새로 만들고 다음 정확한 내용을 작성:

    ```python
    #
    # Copyright 2026 WuXi EsionTech Co., Ltd.
    #
    # Licensed under the Apache License, Version 2.0 (the "License");
    # you may not use this file except in compliance with the License.
    # You may obtain a copy of the License at
    #
    #    http://www.apache.org/licenses/LICENSE-2.0
    #
    # Unless required by applicable law or agreed to in writing, software
    # distributed under the License is distributed on an "AS IS" BASIS,
    # WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
    # See the License for the specific language governing permissions and
    # limitations under the License.
    #
    """GTX NPU functional model — Phase 1 skeleton.

    Phase 1 exposes FP16 helpers (`fp`), memory layer (`memory`), HW parameter
    constants (`params`), funct7 encoding (`encoding`), and DDR helpers (`ddr`).
    `GtxNpu` (the ROCC subclass) is added in Phase 2 (D-14).
    """
    import sys

    # D-09 / RESEARCH.md "Anti-Patterns": np.float16 view assumes little-endian host.
    # manylinux2014_x86_64 is always LE; this tripwire defends against accidental
    # non-LE host (theoretical — not in v1 platform target).
    if sys.byteorder != "little":
        raise RuntimeError(
            f"riscv.gtx requires little-endian host (sys.byteorder='little'); "
            f"got '{sys.byteorder}'. NumPy float16 view semantics assume LE byte order."
        )

    from . import encoding
    from . import fp
    from . import memory
    from . import params
    from . import ddr

    __all__ = ["encoding", "fp", "memory", "params", "ddr"]
    ```

    중요한 점:
    - `from . import …` 5개 모두 `riscv.gtx` import 시 즉시 import됨 (lazy 안 함). RESEARCH.md
      open-question §3에 따라 `riscv/__init__.py`는 건드리지 않는다 — `import riscv`는
      NumPy 미설치 환경에서도 깨지지 않아야 한다 (`riscv.gtx`만 NumPy를 끌어옴).
    - `fp`/`memory`/`ddr` 모듈은 Wave 1 다른 plan(`02-fp`, `03-memory`)에서 작성됨.
      이 task에서는 import 라인만 작성하고 모듈 본체는 다른 plan이 채운다. Wave 1
      병렬 실행 후 Wave 2 wheel-verify 시점에 모든 모듈이 존재해야 한다.
    - `from . import gtx`를 `riscv/__init__.py`에 추가하지 않음 — 사용자는 명시적
      `from riscv.gtx import …`로 import (RESEARCH.md `<phase_specific_guidance>` #5).
  </action>
  <verify>
    <automated>test -f src/main/python/riscv/gtx/__init__.py &amp;&amp; grep -q 'sys.byteorder != "little"' src/main/python/riscv/gtx/__init__.py &amp;&amp; grep -q 'from . import fp' src/main/python/riscv/gtx/__init__.py &amp;&amp; grep -q 'from . import memory' src/main/python/riscv/gtx/__init__.py &amp;&amp; grep -q 'from . import params' src/main/python/riscv/gtx/__init__.py &amp;&amp; grep -q 'from . import encoding' src/main/python/riscv/gtx/__init__.py &amp;&amp; grep -q 'from . import ddr' src/main/python/riscv/gtx/__init__.py</automated>
  </verify>
  <acceptance_criteria>
    - `test -f src/main/python/riscv/gtx/__init__.py` 종료코드 0
    - `grep -q 'sys.byteorder != "little"' src/main/python/riscv/gtx/__init__.py` 종료코드 0 (LE tripwire 존재)
    - `grep -c 'from . import' src/main/python/riscv/gtx/__init__.py` 출력 >= 5 (fp/memory/params/encoding/ddr re-export)
    - `grep -q 'GtxNpu' src/main/python/riscv/gtx/__init__.py` 종료코드 1 (D-14: P1에서 GtxNpu 노출 금지)
    - `python -c "import ast; ast.parse(open('src/main/python/riscv/gtx/__init__.py').read())"` 종료코드 0 (syntax valid)
    - `git diff src/main/python/riscv/__init__.py` 출력이 비어 있어야 함 (RESEARCH.md §"open-question 3": 변경 안 함)
  </acceptance_criteria>
  <done>`riscv/gtx/__init__.py`가 존재하고 LE 가드 + 5개 서브모듈 re-export를 포함; `riscv/__init__.py`는 unchanged.</done>
</task>

<task type="auto" tdd="false">
  <name>Task 01-02: HW 파라미터 상수 (params.py) + funct7 상수 (encoding.py)</name>
  <files>src/main/python/riscv/gtx/params.py, src/main/python/riscv/gtx/encoding.py, src/main/python/riscv/gtx/ops/__init__.py</files>
  <read_first>
    - .planning/phases/01-foundation/01-CONTEXT.md "Claude's Discretion" (params naming = C++ macro 그대로; encoding scope = funct7 상수만)
    - .planning/phases/01-foundation/01-RESEARCH.md "Example 6: params.py constants" + "Example 7: encoding.py Phase 1 scope" (정확한 상수 값)
    - src/main/python/riscv/gtx/__init__.py (Task 01-01 출력 — 이 모듈들이 import 됨)
  </read_first>
  <action>
    **`src/main/python/riscv/gtx/params.py` 파일을 새로 만들고 다음 정확한 내용을 작성** (RESEARCH.md Example 6 기반, gtx_params.h 매크로 그대로):

    ```python
    #
    # Copyright 2026 WuXi EsionTech Co., Ltd.
    #
    # Licensed under the Apache License, Version 2.0 (the "License"); ...
    # (라이선스 헤더 — riscv/__init__.py와 동일 톤)
    #
    """Hardware parameter constants — direct port of vendor/gtx_cpp_reference/gtx/gtx_params.h.

    Naming follows the C++ macro convention verbatim (per CONTEXT.md Claude's Discretion).
    These values are referenced by tests/gtx/test_memory_layout.py and by Phase 2-5 op handlers.
    """
    # NEST x SPU topology
    GTX_NEST_NUM: int = 4
    GTX_SPU_NUM: int = 16          # SPUs per NEST
    GTX_SPUS_PER_NEST: int = GTX_SPU_NUM   # alias for clarity

    # Memory sizes (bytes)
    GTX_L0_SIZE_BYTES: int = 1024                      # 1 KB per SPU
    GTX_L1_SIZE_BYTES: int = 384 * 1024                # 384 KB per SPU
    GTX_L2_SIZE_BYTES: int = 16 * 1024 * 1024          # 16 MB per NEST

    # DDR (D-02: capped by GTX_DDR_SIZE env var; default below)
    GTX_DDR_DEFAULT_SIZE_BYTES: int = 4 * 1024 * 1024 * 1024   # 4 GiB

    # DDR I/O (D-03)
    GTX_DDR_BUS_WORD_BYTES: int = 32   # 32-byte bus word for GTX_DDR_REVERSED reversal

    # SPR address ranges (D-11)
    GSPR_BASE: int = 0x000
    GSPR_END: int = 0x3FF
    NSPR_BASE: int = 0x400
    NSPR_END: int = 0x7FF
    LSPR_BASE: int = 0x800
    LSPR_END: int = 0xBFF
    ```

    **`src/main/python/riscv/gtx/encoding.py` 파일을 새로 만들고 다음 정확한 내용을 작성** (RESEARCH.md Example 7 기반; full disasm.inc은 P2):

    ```python
    #
    # Copyright 2026 WuXi EsionTech Co., Ltd.
    # ... (라이선스 헤더)
    #
    """GTX RoCC instruction encoding constants.

    Phase 1 scope: funct7 constants (used by P2 dispatch + disasm).
    Full disasm_insn_t table moves to disasm.py in Phase 2 (D-13 scope).
    """
    # RoCC funct7 — gem5 simplified (operand staging via GSPR):
    GTX_F7_WRSPR: int = 0x00       # WRSPR (gem5) / MM ISS-full (rs1!=0 disambiguation in P4)
    GTX_F7_RDSPR: int = 0x01
    GTX_F7_WSPLIT: int = 0x02      # custom1 (warp split)
    GTX_F7_WJOIN: int = 0x03       # custom1 (warp join — exit semantics in P2)
    GTX_F7_DISPATCH_MM: int = 0x04
    GTX_F7_DISPATCH_VEC: int = 0x05
    GTX_F7_DISPATCH_ACT: int = 0x06
    GTX_F7_DISPATCH_DMA: int = 0x07

    # ISS full (per-op funct7) — selected; full table P2:
    # GTX_F7_MM = 0x00 (collides with WRSPR; resolved by insn.rs1 != 0 — P4)
    # GTX_F7_MMC = 0x01
    # GTX_F7_DMA_LOAD = 0x40
    # GTX_F7_OPSET = 0x4A
    # (...remaining 70+ constants in Phase 2)
    ```

    **`src/main/python/riscv/gtx/ops/__init__.py` 파일을 새로 만들고 다음 내용을 작성**:

    ```python
    #
    # Copyright 2026 WuXi EsionTech Co., Ltd.
    # ... (라이선스 헤더)
    #
    """Op handler package marker. Populated in Phases 2-5 (D-13 scope)."""
    ```

    중요한 점:
    - `params.py`는 C++ `gtx_params.h` 매크로 이름과 값 그대로 (CONTEXT.md "Claude's
      Discretion" 항목 #2에서 권장). vendor/gtx_cpp_reference/는 Wave 2의 `05-submodule`
      plan이 등록하므로 이 task 시점에는 디스크에 없음 — 값은 RESEARCH.md Example 6 +
      CONTEXT.md/STATE.md에서 정의된 (4, 16, 1024, 384*1024, 16MB, 4GB) 그대로.
    - `encoding.py`에 ISS full funct7는 주석으로만 (P2가 채움). funct7=0x00 충돌은
      P4에서 `insn.rs1 != 0` 휴리스틱으로 해결 (PROJECT.md PITFALLS #5 참조).
    - `ops/__init__.py`는 빈 마커 — P2~P5에서 `mm.py`, `vec.py`, `act.py`, `dma.py` 등이 추가됨.
  </action>
  <verify>
    <automated>python -c "import sys; sys.path.insert(0, 'src/main/python'); from riscv.gtx.params import GTX_NEST_NUM, GTX_SPU_NUM, GTX_L0_SIZE_BYTES, GTX_L1_SIZE_BYTES, GTX_L2_SIZE_BYTES, GTX_DDR_DEFAULT_SIZE_BYTES, GTX_DDR_BUS_WORD_BYTES, GSPR_BASE, NSPR_BASE, LSPR_BASE; assert (GTX_NEST_NUM, GTX_SPU_NUM) == (4, 16); assert GTX_L1_SIZE_BYTES == 384*1024; assert GTX_L2_SIZE_BYTES == 16*1024*1024; assert GTX_DDR_DEFAULT_SIZE_BYTES == 4*1024**3; assert GTX_DDR_BUS_WORD_BYTES == 32; assert (GSPR_BASE, NSPR_BASE, LSPR_BASE) == (0x000, 0x400, 0x800); from riscv.gtx.encoding import GTX_F7_WRSPR, GTX_F7_WJOIN, GTX_F7_DISPATCH_MM; assert GTX_F7_WRSPR == 0x00; assert GTX_F7_WJOIN == 0x03; assert GTX_F7_DISPATCH_MM == 0x04; import riscv.gtx.ops; print('OK')"</automated>
  </verify>
  <acceptance_criteria>
    - 위 `python -c` 명령 종료코드 0 + "OK" 출력
    - `grep -q 'GTX_NEST_NUM: int = 4' src/main/python/riscv/gtx/params.py` 종료코드 0
    - `grep -q 'GTX_SPU_NUM: int = 16' src/main/python/riscv/gtx/params.py` 종료코드 0
    - `grep -q 'GTX_L1_SIZE_BYTES: int = 384 \* 1024' src/main/python/riscv/gtx/params.py` 종료코드 0
    - `grep -q 'GTX_F7_WRSPR: int = 0x00' src/main/python/riscv/gtx/encoding.py` 종료코드 0
    - `grep -q 'GTX_F7_WJOIN: int = 0x03' src/main/python/riscv/gtx/encoding.py` 종료코드 0
    - `test -f src/main/python/riscv/gtx/ops/__init__.py` 종료코드 0
    - `python -c "import ast; ast.parse(open('src/main/python/riscv/gtx/params.py').read()); ast.parse(open('src/main/python/riscv/gtx/encoding.py').read()); ast.parse(open('src/main/python/riscv/gtx/ops/__init__.py').read())"` 종료코드 0
  </acceptance_criteria>
  <done>params.py / encoding.py / ops/__init__.py가 존재하고, 위 import 어서션이 모두 통과 (GTX_NEST_NUM/GTX_SPU_NUM/L1/L2/DDR_DEFAULT/BUS_WORD/SPR base + funct7 상수).</done>
</task>

<task type="auto" tdd="false">
  <name>Task 01-03: tests/gtx 디렉토리 + 라이선스 헤더 마커</name>
  <files>tests/gtx/__init__.py</files>
  <read_first>
    - tests/__init__.py (기존 라이선스 헤더 톤 — 같은 헤더 + 빈 모듈)
    - .planning/phases/01-foundation/01-RESEARCH.md "Don't Hand-Roll" 마지막 줄 "Test discovery for new tests/gtx/ subdir → None — pytest auto-discovers"
    - .planning/phases/01-foundation/01-VALIDATION.md "Wave 0 Requirements" (tests/gtx/__init__.py 필수)
  </read_first>
  <action>
    `tests/gtx/__init__.py` 파일을 새로 만들고 `tests/__init__.py`와 동일한 톤의
    라이선스 헤더만 포함하는 빈 모듈로 작성:

    ```python
    #
    # Copyright 2026 WuXi EsionTech Co., Ltd.
    #
    # Licensed under the Apache License, Version 2.0 (the "License");
    # you may not use this file except in compliance with the License.
    # You may obtain a copy of the License at
    #
    #    http://www.apache.org/licenses/LICENSE-2.0
    #
    # Unless required by applicable law or agreed to in writing, software
    # distributed under the License is distributed on an "AS IS" BASIS,
    # WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
    # See the License for the specific language governing permissions and
    # limitations under the License.
    #
    ```

    중요한 점:
    - 추가 코드 없음 — pytest는 `pyproject.toml [tool.pytest.ini_options].testpaths
      = ["tests"]`를 통해 `tests/gtx/`를 자동 발견.
    - 이 task는 디렉토리 생성 + 라이선스 헤더만 — 실제 테스트는 Wave 1의 별도
      plan(`02-fp`/`03-memory`)이 작성한다.
    - `tests/__init__.py`도 헤더만 있는 빈 모듈 (RESEARCH.md "Open Questions" #2에서
      확인된 기존 컨벤션).
  </action>
  <verify>
    <automated>test -f tests/gtx/__init__.py &amp;&amp; grep -q 'Apache License' tests/gtx/__init__.py &amp;&amp; python -c "import ast; ast.parse(open('tests/gtx/__init__.py').read())"</automated>
  </verify>
  <acceptance_criteria>
    - `test -d tests/gtx` 종료코드 0
    - `test -f tests/gtx/__init__.py` 종료코드 0
    - `grep -q 'Apache License' tests/gtx/__init__.py` 종료코드 0 (라이선스 헤더 존재)
    - `python -c "import ast; ast.parse(open('tests/gtx/__init__.py').read())"` 종료코드 0
    - `wc -l tests/gtx/__init__.py` 출력 라인 수 <= 20 (pure 헤더만; 본문 코드 없음)
  </acceptance_criteria>
  <done>`tests/gtx/__init__.py`가 존재 + 라이선스 헤더만 포함. pytest collection이 이 디렉토리를 인식 (Wave 1의 다른 plan들이 실제 테스트 파일 추가).</done>
</task>

</tasks>

<verification>
**Plan-level verification:**
- 모든 3개 task의 `<acceptance_criteria>` 통과
- `python -c "import ast; ast.parse(open(p).read())"` 5개 파일 모두 syntax valid
- `riscv/__init__.py`는 unchanged (`git diff src/main/python/riscv/__init__.py` 출력 비어있음)
- Wave 1의 다른 plan들(`02-fp`, `03-memory`)이 같은 시점에 실행되어 fp.py/memory.py/ddr.py를 만든 후에야 `python -c "from riscv.gtx import fp, memory, params"` 통과 가능 — 이 plan 단독으로는 import가 깨질 수 있음 (Wave 1 종료 시점에 통합 검증).
</verification>

<success_criteria>
1. `src/main/python/riscv/gtx/__init__.py`, `params.py`, `encoding.py`, `ops/__init__.py`, `tests/gtx/__init__.py` 5개 파일 모두 존재
2. `python -c "from riscv.gtx.params import GTX_NEST_NUM; assert GTX_NEST_NUM == 4"` 종료코드 0
3. `python -c "from riscv.gtx.encoding import GTX_F7_WRSPR; assert GTX_F7_WRSPR == 0x00"` 종료코드 0
4. `python -c "import riscv.gtx.ops"` 종료코드 0
5. `git diff src/main/python/riscv/__init__.py` 출력 비어있음 (RESEARCH.md open-question §3 준수)
</success_criteria>

<output>
After completion, create `.planning/phases/01-foundation/01-skeleton-SUMMARY.md` with:
- 5개 파일의 정확한 라인 수
- params.py / encoding.py에 정의된 상수 목록 (이름 + 값)
- riscv/__init__.py가 변경되지 않았음을 확인
- Wave 1 다른 plan(02-fp, 03-memory)에 대한 import 의존성 노트
</output>
