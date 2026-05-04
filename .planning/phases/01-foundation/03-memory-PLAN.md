---
phase: 01-foundation
plan: 03
type: execute
wave: 1
depends_on: []
files_modified:
  - src/main/python/riscv/gtx/memory.py
  - src/main/python/riscv/gtx/ddr.py
  - tests/gtx/test_memory_layout.py
autonomous: true
requirements:
  - FOUND-02
must_haves:
  truths:
    - "Writing LE bytes [0x00, 0x3C] to mem.l1_byte(0,0)[0:2] reads back as np.float16(1.0) via mem.l1_f16(0,0)[0]"
    - "Writing np.float16(2.0) via mem.l1_f16(0,0)[0] produces LE bytes [0x00, 0x40] in mem.l1_byte(0,0)[0:2]"
    - "Every named accessor (l0_byte/l0_f16/l1_byte/l1_f16/l2_byte/l2_f16) returns a view: arr.base is not None (D-12)"
    - "Slicing a view preserves base — sub = view[100:200]; sub.base is not None"
    - "L1 shape per (nest, spu) == (GTX_L1_SIZE_BYTES,) = (393216,); fp16 view shape == (196608,)"
    - "mem.spr is dict[int,int] (D-11 unified GSPR/NSPR/LSPR routing)"
    - "mem._ddr_bytes is None at construction (D-01 lazy DDR)"
  artifacts:
    - path: "src/main/python/riscv/gtx/memory.py"
      provides: "GtxMemory class — L0/L1/L2 contiguous np.uint8 + named accessors + SPR dict"
      exports: ["GtxMemory"]
    - path: "src/main/python/riscv/gtx/ddr.py"
      provides: "DDR lazy alloc helpers (D-01) + GTX_DDR_SIZE env var parsing (D-02). I/O body deferred to Phase 3"
      exports: ["DEFAULT_DDR_SIZE", "get_ddr_cap", "ensure_ddr"]
    - path: "tests/gtx/test_memory_layout.py"
      provides: "8+ acceptance tests covering FOUND-02 — LE byte order, view invariants, shapes, SPR dict, DDR lazy"
      contains: "test_le_byte_order_via_byte_write"
  key_links:
    - from: "src/main/python/riscv/gtx/memory.py"
      to: "src/main/python/riscv/gtx/params.py"
      via: "from .params import GTX_NEST_NUM, GTX_SPU_NUM, GTX_L0_SIZE_BYTES, GTX_L1_SIZE_BYTES, GTX_L2_SIZE_BYTES"
      pattern: "from \\.params import"
    - from: "src/main/python/riscv/gtx/memory.py"
      to: "numpy ndarray view semantics"
      via: "self._l1_bytes[nest, spu].view(np.float16) — D-12 view-base 보장"
      pattern: "\\.view\\(np\\.float16\\)"
    - from: "tests/gtx/test_memory_layout.py"
      to: "src/main/python/riscv/gtx/memory.py"
      via: "from riscv.gtx.memory import GtxMemory"
      pattern: "from riscv.gtx.memory import GtxMemory"
---

<objective>
NumPy 백엔드 메모리 레이어 — `GtxMemory` 클래스(L0/L1/L2 ndarray + named accessor +
SPR unified dict + DDR lazy alloc) 와 DDR 환경변수 파싱 헬퍼를 작성하고, FOUND-02의
모든 acceptance 조건(LE byte order, view-base 불변량, shape, SPR dict, DDR lazy)을
다루는 8+ pytest 함수를 `tests/gtx/test_memory_layout.py`에 작성한다.

Purpose: D-10 (layered API), D-11 (SPR unified dict), D-12 (view-base invariant),
D-01 (lazy DDR), D-02 (GTX_DDR_SIZE env var) 모두 lock-in. Phase 2~5 op 핸들러가
`mem.l1_f16(nest, spu)[off]`로 in-place FP16 쓰기를 안정적으로 할 수 있는 기반.
PROJECT.md PITFALL #1 (verify.py BE vs L1/L0 LE byte order)을 직접 방어한다.

Output: 3개 파일. RESEARCH.md Pattern 1 (memory) + Pattern 3 (DDR) + Example 5 (tests)
의 정확한 코드 사용 — 2026-05-04 NumPy 2.2.6 cp310 venv에서 검증됨.
</objective>

<execution_context>
@$HOME/.claude/get-shit-done/workflows/execute-plan.md
@$HOME/.claude/get-shit-done/templates/summary.md
</execution_context>

<context>
@.planning/PROJECT.md
@.planning/ROADMAP.md
@.planning/REQUIREMENTS.md
@.planning/phases/01-foundation/01-CONTEXT.md
@.planning/phases/01-foundation/01-RESEARCH.md
@.planning/phases/01-foundation/01-VALIDATION.md
@CLAUDE.md
</context>

<interfaces>
<!-- 이 plan이 만드는 contract — Phase 2 SPR + Phase 3 DMA + Phase 4/5 op 핸들러가 사용 -->

```python
# src/main/python/riscv/gtx/memory.py exports:
class GtxMemory:
    def __init__(self) -> None: ...

    # Raw byte view (D-10 low-level)
    def l0_byte(self, nest: int, spu: int) -> np.ndarray: ...   # uint8, shape=(GTX_L0_SIZE_BYTES,)
    def l1_byte(self, nest: int, spu: int) -> np.ndarray: ...   # uint8, shape=(GTX_L1_SIZE_BYTES,)
    def l2_byte(self, nest: int) -> np.ndarray: ...             # uint8, shape=(GTX_L2_SIZE_BYTES,)

    # Halfword fp16 view (D-10 named accessor, D-12 view guarantee)
    def l0_f16(self, nest: int, spu: int) -> np.ndarray: ...    # float16, shape=(GTX_L0_SIZE_BYTES//2,)
    def l1_f16(self, nest: int, spu: int) -> np.ndarray: ...    # float16, shape=(GTX_L1_SIZE_BYTES//2,)
    def l2_f16(self, nest: int) -> np.ndarray: ...              # float16, shape=(GTX_L2_SIZE_BYTES//2,)

    # uint16 view (rare; pattern testing)
    def l1_u16(self, nest: int, spu: int) -> np.ndarray: ...

    spr: dict[int, int]                       # D-11: unified GSPR/NSPR/LSPR routing
    _ddr_bytes: np.ndarray | None             # D-01: None at construction
    _l0_bytes: np.ndarray                     # shape (GTX_NEST_NUM, GTX_SPU_NUM, GTX_L0_SIZE_BYTES)
    _l1_bytes: np.ndarray                     # shape (GTX_NEST_NUM, GTX_SPU_NUM, GTX_L1_SIZE_BYTES)
    _l2_bytes: np.ndarray                     # shape (GTX_NEST_NUM, GTX_L2_SIZE_BYTES)

# src/main/python/riscv/gtx/ddr.py exports:
DEFAULT_DDR_SIZE: int = 4 * 1024 ** 3   # 4 GiB
def get_ddr_cap() -> int: ...           # reads GTX_DDR_SIZE env var ("4G"/"64M"/"1024K"/raw int)
def ensure_ddr(mem: GtxMemory, end_offset: int) -> np.ndarray: ...   # Phase 1 stub: raises ValueError on > cap
```

D-03 reminder: GTX_DDR_REVERSED는 I/O 경계에서만 적용. 내부 DDR 버퍼는 항상 LE — Phase 3가
I/O 본체 채움. Phase 1은 lazy alloc + cap 파싱만.
</interfaces>

<tasks>

<task type="auto" tdd="true">
  <name>Task 03-01: tests/gtx/test_memory_layout.py — RED phase</name>
  <files>tests/gtx/test_memory_layout.py</files>
  <read_first>
    - .planning/phases/01-foundation/01-CONTEXT.md (D-10/D-11/D-12/D-17 ALL critical)
    - .planning/phases/01-foundation/01-RESEARCH.md "Example 5: Memory layout test" (정확한 8개 테스트)
    - .planning/phases/01-foundation/01-VALIDATION.md "Per-Task Verification Map" (03-memory 7개 verify 명령)
    - tests/test_extension.py (기존 pytest 컨벤션)
  </read_first>
  <behavior>
    - test_le_byte_order_via_byte_write: l1_byte(0,0)[0:2] = [0x00,0x3C] → l1_f16(0,0)[0] == np.float16(1.0)
    - test_le_byte_order_via_fp16_write: l1_f16(0,0)[0] = np.float16(2.0) → l1_byte(0,0)[0:2] == [0x00,0x40]
    - test_l1_f16_view_invariant: l1_f16().base is not None, shape == (GTX_L1_SIZE_BYTES//2,), dtype == np.float16
    - test_l0_f16_view_invariant: l0_f16().base is not None, shape == (GTX_L0_SIZE_BYTES//2,)
    - test_slice_preserves_base: view = l1_f16(0,0); sub = view[100:200]; sub.base is not None
    - test_l1_shape: l1_byte(n,s).shape == (GTX_L1_SIZE_BYTES,) for all (n,s) in NEST*SPU
    - test_spr_dict: mem.spr is dict, empty initially, supports 0x100/0x500/0x900 routing
    - test_ddr_lazy_allocation: mem._ddr_bytes is None
  </behavior>
  <action>
    `tests/gtx/test_memory_layout.py` 파일을 새로 만들고 RESEARCH.md Example 5의
    정확한 코드를 사용:

    ```python
    #
    # Copyright 2026 WuXi EsionTech Co., Ltd.
    # ... (Apache 2.0 헤더 — tests/__init__.py 톤과 동일)
    #
    """Phase 1 acceptance: GtxMemory layout invariants.

    D-17: writing 0x3C00 to halfword view at L1[nest=0, spu=0, off=0] produces bytes
          [0x00, 0x3C] LE.
    D-12: every named accessor returns a non-copying view (arr.base is not None).
    D-11: mem.spr is a unified dict[int, int].
    D-01: DDR is None at construction (lazy alloc).
    """
    import numpy as np
    import pytest

    from riscv.gtx.memory import GtxMemory
    from riscv.gtx.params import (
        GTX_NEST_NUM,
        GTX_SPU_NUM,
        GTX_L0_SIZE_BYTES,
        GTX_L1_SIZE_BYTES,
    )


    @pytest.fixture
    def mem():
        return GtxMemory()


    def test_le_byte_order_via_byte_write(mem):
        """Writing LE bytes [0x00, 0x3C] to L1 byte view appears as np.float16(1.0) in fp16 view."""
        mem.l1_byte(0, 0)[0] = 0x00
        mem.l1_byte(0, 0)[1] = 0x3C
        assert mem.l1_f16(0, 0)[0] == np.float16(1.0)


    def test_le_byte_order_via_fp16_write(mem):
        """Writing np.float16(2.0) to L1 fp16 view produces LE bytes [0x00, 0x40]."""
        mem.l1_f16(0, 0)[0] = np.float16(2.0)
        assert mem.l1_byte(0, 0)[0] == 0x00
        assert mem.l1_byte(0, 0)[1] == 0x40


    def test_l1_f16_view_invariant(mem):
        """D-12: l1_f16 returns a view, not a copy."""
        view = mem.l1_f16(0, 0)
        assert view.base is not None, "l1_f16 must return a view (D-12)"
        assert view.shape == (GTX_L1_SIZE_BYTES // 2,)
        assert view.dtype == np.float16


    def test_l0_f16_view_invariant(mem):
        """D-12: l0_f16 returns a view, not a copy."""
        view = mem.l0_f16(0, 0)
        assert view.base is not None
        assert view.shape == (GTX_L0_SIZE_BYTES // 2,)


    def test_slice_preserves_base(mem):
        """Slicing an fp16 view preserves base (no copy on slice)."""
        view = mem.l1_f16(0, 0)
        sub = view[100:200]
        assert sub.base is not None, "slice of view must remain a view"


    def test_l1_shape(mem):
        """L1 dimensions match HW parameters."""
        assert mem.l1_byte(0, 0).shape == (GTX_L1_SIZE_BYTES,)
        assert mem.l1_byte(0, 0).dtype == np.uint8
        for n in range(GTX_NEST_NUM):
            for s in range(GTX_SPU_NUM):
                assert mem.l1_byte(n, s).shape == (GTX_L1_SIZE_BYTES,)


    def test_spr_dict(mem):
        """D-11: mem.spr is a unified dict[int, int]."""
        assert isinstance(mem.spr, dict)
        assert len(mem.spr) == 0
        mem.spr[0x100] = 0xCAFE     # GSPR range
        mem.spr[0x500] = 0xBABE     # NSPR range
        mem.spr[0x900] = 0xF00D     # LSPR range
        assert mem.spr[0x100] == 0xCAFE
        assert mem.spr[0x500] == 0xBABE
        assert mem.spr[0x900] == 0xF00D


    def test_ddr_lazy_allocation(mem):
        """D-01: DDR is None at construction."""
        assert mem._ddr_bytes is None
    ```

    중요한 점:
    - TDD RED: `riscv.gtx.memory.GtxMemory`가 없어서 import 실패가 정상. Task 03-02가 GREEN.
    - `mem._ddr_bytes` private 접근은 D-01 검증에 필수 — pytest 단위 테스트에서 허용.
  </behavior>
  <verify>
    <automated>test -f tests/gtx/test_memory_layout.py &amp;&amp; grep -q 'def test_le_byte_order_via_byte_write' tests/gtx/test_memory_layout.py &amp;&amp; grep -q 'def test_le_byte_order_via_fp16_write' tests/gtx/test_memory_layout.py &amp;&amp; grep -q 'def test_l1_f16_view_invariant' tests/gtx/test_memory_layout.py &amp;&amp; grep -q 'def test_slice_preserves_base' tests/gtx/test_memory_layout.py &amp;&amp; grep -q 'def test_spr_dict' tests/gtx/test_memory_layout.py &amp;&amp; grep -q 'def test_ddr_lazy_allocation' tests/gtx/test_memory_layout.py &amp;&amp; python -c "import ast; ast.parse(open('tests/gtx/test_memory_layout.py').read())"</automated>
  </verify>
  <acceptance_criteria>
    - `grep -c 'def test_' tests/gtx/test_memory_layout.py` 출력 >= 8
    - 7개 핵심 함수 모두 grep 매칭 (le_byte_order_via_byte_write/le_byte_order_via_fp16_write/l1_f16_view_invariant/l0_f16_view_invariant/slice_preserves_base/l1_shape/spr_dict/ddr_lazy_allocation)
    - `grep -q 'from riscv.gtx.memory import GtxMemory' tests/gtx/test_memory_layout.py` 종료코드 0
    - `grep -q 'from riscv.gtx.params import' tests/gtx/test_memory_layout.py` 종료코드 0
    - `grep -q 'is not None' tests/gtx/test_memory_layout.py` 종료코드 0 (D-12 검증)
    - `grep -q '0x3C' tests/gtx/test_memory_layout.py` 종료코드 0 (D-17 검증)
    - `python -c "import ast; ast.parse(open('tests/gtx/test_memory_layout.py').read())"` 종료코드 0
  </acceptance_criteria>
  <done>8+ 테스트 함수 작성됨; FOUND-02의 모든 acceptance 조건이 테스트로 표현됨; syntax valid; RED state.</done>
</task>

<task type="auto" tdd="true">
  <name>Task 03-02: src/main/python/riscv/gtx/memory.py — GREEN phase</name>
  <files>src/main/python/riscv/gtx/memory.py</files>
  <read_first>
    - .planning/phases/01-foundation/01-CONTEXT.md (D-10/D-11/D-12 lock; "anti-pattern: arr.view 직접 호출 host endian")
    - .planning/phases/01-foundation/01-RESEARCH.md "Pattern 1: NumPy-backed memory with halfword view" (정확한 코드)
    - .planning/phases/01-foundation/01-RESEARCH.md "Anti-Patterns to Avoid" (eager DDR alloc 금지)
    - tests/gtx/test_memory_layout.py (Task 03-01 출력 — 메서드 시그니처 contract)
    - src/main/python/riscv/gtx/params.py (Task 01-02 출력 — 사용할 상수)
  </read_first>
  <behavior>
    - GtxMemory().__init__: L0/L1/L2 contiguous np.uint8 zeros, spr={}, _ddr_bytes=None
    - L0 shape: (GTX_NEST_NUM, GTX_SPU_NUM, GTX_L0_SIZE_BYTES) = (4, 16, 1024)
    - L1 shape: (GTX_NEST_NUM, GTX_SPU_NUM, GTX_L1_SIZE_BYTES) = (4, 16, 393216)
    - L2 shape: (GTX_NEST_NUM, GTX_L2_SIZE_BYTES) = (4, 16777216)
    - l0_byte/l1_byte/l2_byte → uint8 view (slice indexing)
    - l0_f16/l1_f16/l2_f16 → float16 view (.view(np.float16))
    - l1_u16 → uint16 view
    - 모든 named accessor의 반환값에 D-12: `arr.base is not None`
  </behavior>
  <action>
    `src/main/python/riscv/gtx/memory.py` 파일을 새로 만들고 RESEARCH.md Pattern 1
    그대로 작성:

    ```python
    #
    # Copyright 2026 WuXi EsionTech Co., Ltd.
    # ... (Apache 2.0 헤더)
    #
    """NumPy-backed memory layer for GTX NPU.

    D-10: Layered API. Both raw byte views and named halfword accessors.
    D-11: SPR unified dict[int, int]. GSPR/NSPR/LSPR routing by address (P2 SPR-01).
    D-12: Every named accessor returns a non-copying view (arr.base is not None).
    D-01: DDR is lazily allocated (see ddr.py:ensure_ddr); _ddr_bytes is None initially.
    """
    from typing import Optional

    import numpy as np

    from .params import (
        GTX_L0_SIZE_BYTES,
        GTX_L1_SIZE_BYTES,
        GTX_L2_SIZE_BYTES,
        GTX_NEST_NUM,
        GTX_SPU_NUM,
    )


    class GtxMemory:
        """GTX NPU memory layer — L0/L1/L2 ndarray + DDR lazy alloc + SPR dict."""

        def __init__(self) -> None:
            self._l0_bytes: np.ndarray = np.zeros(
                (GTX_NEST_NUM, GTX_SPU_NUM, GTX_L0_SIZE_BYTES), dtype=np.uint8
            )
            self._l1_bytes: np.ndarray = np.zeros(
                (GTX_NEST_NUM, GTX_SPU_NUM, GTX_L1_SIZE_BYTES), dtype=np.uint8
            )
            self._l2_bytes: np.ndarray = np.zeros(
                (GTX_NEST_NUM, GTX_L2_SIZE_BYTES), dtype=np.uint8
            )
            self.spr: dict[int, int] = {}
            self._ddr_bytes: Optional[np.ndarray] = None

        # ----- Raw byte views (D-10 low-level) -----

        def l0_byte(self, nest: int, spu: int) -> np.ndarray:
            return self._l0_bytes[nest, spu]

        def l1_byte(self, nest: int, spu: int) -> np.ndarray:
            return self._l1_bytes[nest, spu]

        def l2_byte(self, nest: int) -> np.ndarray:
            return self._l2_bytes[nest]

        # ----- Halfword fp16 views (D-10 named, D-12 view guarantee) -----

        def l0_f16(self, nest: int, spu: int) -> np.ndarray:
            view = self._l0_bytes[nest, spu].view(np.float16)
            assert view.base is not None  # D-12 tripwire
            return view

        def l1_f16(self, nest: int, spu: int) -> np.ndarray:
            view = self._l1_bytes[nest, spu].view(np.float16)
            assert view.base is not None
            return view

        def l2_f16(self, nest: int) -> np.ndarray:
            view = self._l2_bytes[nest].view(np.float16)
            assert view.base is not None
            return view

        # ----- Halfword uint16 view (rare) -----

        def l1_u16(self, nest: int, spu: int) -> np.ndarray:
            view = self._l1_bytes[nest, spu].view(np.uint16)
            assert view.base is not None
            return view
    ```

    중요한 점:
    - L0/L1/L2은 eager `np.zeros` (총 ~88MB. CI 메모리 압박 없음). DDR만 lazy (D-01).
    - `.view(np.float16)`은 host endian (RESEARCH.md Anti-Pattern #1) — `riscv/gtx/__init__.py`의
      LE tripwire (Task 01-01)가 방어. `np.frombuffer(buf, dtype='<f2')`는 view를 얻지
      못해 D-12 위반이므로 사용 안 함.
    - `assert view.base is not None`을 helper 안에 D-12 tripwire로 박는다 (런타임 +
      pytest 이중 안전망).
    - `dict[int, int]` annotation은 cp310+ (D-08, PEP 585).
    - `Optional[np.ndarray]` typing.Optional 사용 (PEP 604 `|`도 가능; Optional 유지).
    - reset() 메서드는 P2 (CORE-02). 여기 Phase 1 스코프 밖.
    - WRSPR/RDSPR 비즈니스 로직은 P2 SPR-01/02. 여기는 spr dict만 노출.
  </action>
  <verify>
    <automated>test -f src/main/python/riscv/gtx/memory.py &amp;&amp; PYTHONPATH=src/main/python python -c "from riscv.gtx.memory import GtxMemory; from riscv.gtx.params import GTX_L1_SIZE_BYTES; import numpy as np; m=GtxMemory(); v=m.l1_f16(0,0); assert v.base is not None; assert v.shape == (GTX_L1_SIZE_BYTES//2,); assert v.dtype == np.float16; m.l1_byte(0,0)[0]=0x00; m.l1_byte(0,0)[1]=0x3C; assert m.l1_f16(0,0)[0] == np.float16(1.0); m.l1_f16(0,0)[10] = np.float16(2.0); assert m.l1_byte(0,0)[20]==0x00 and m.l1_byte(0,0)[21]==0x40; assert isinstance(m.spr, dict); assert m._ddr_bytes is None; print('OK')" &amp;&amp; PYTHONPATH=src/main/python pytest tests/gtx/test_memory_layout.py -x -p no:pylint -p no:mypy --no-header -q</automated>
  </verify>
  <acceptance_criteria>
    - `test -f src/main/python/riscv/gtx/memory.py` 종료코드 0
    - `grep -q 'class GtxMemory' src/main/python/riscv/gtx/memory.py` 종료코드 0
    - `grep -c 'def l[012]_byte\|def l[012]_f16' src/main/python/riscv/gtx/memory.py` 출력 >= 6 (l0/l1/l2 byte+f16 6개)
    - `grep -q '\.view(np\.float16)' src/main/python/riscv/gtx/memory.py` 종료코드 0
    - `grep -q 'self.spr: dict\[int, int\] = {}' src/main/python/riscv/gtx/memory.py` 종료코드 0
    - `grep -q 'self._ddr_bytes' src/main/python/riscv/gtx/memory.py` 종료코드 0
    - `grep -q 'assert view.base is not None' src/main/python/riscv/gtx/memory.py` 종료코드 0 (D-12 tripwire 런타임 검증)
    - 위 `python -c "from riscv.gtx.memory import GtxMemory"` 명령이 "OK" 출력 + 종료코드 0
    - `PYTHONPATH=src/main/python pytest tests/gtx/test_memory_layout.py -p no:pylint -p no:mypy --no-header -q` 8/8 PASS (또는 그 이상)
  </acceptance_criteria>
  <done>memory.py + GtxMemory 존재. test_memory_layout.py 8/8 GREEN. l1_byte/l1_f16 LE byte order, view-base 불변량, shape, SPR dict, DDR lazy 모두 검증됨.</done>
</task>

<task type="auto" tdd="false">
  <name>Task 03-03: src/main/python/riscv/gtx/ddr.py — DDR lazy alloc 헬퍼 (Phase 1 stub)</name>
  <files>src/main/python/riscv/gtx/ddr.py</files>
  <read_first>
    - .planning/phases/01-foundation/01-CONTEXT.md (D-01 lazy ensure_ddr; D-02 GTX_DDR_SIZE env var)
    - .planning/phases/01-foundation/01-RESEARCH.md "Pattern 3: Lazy DDR allocation" (정확한 코드)
    - src/main/python/riscv/gtx/memory.py (Task 03-02 출력 — `mem._ddr_bytes` private attr 사용)
  </read_first>
  <action>
    `src/main/python/riscv/gtx/ddr.py` 파일을 새로 만들고 RESEARCH.md Pattern 3을 그대로
    작성. Phase 1 scope: lazy alloc + env var 파싱 stub. I/O 본체 (ddr_init_from_file,
    ddr_dump_to_file)는 Phase 3.

    ```python
    #
    # Copyright 2026 WuXi EsionTech Co., Ltd.
    # ... (Apache 2.0 헤더)
    #
    """DDR backing store — lazy allocation (D-01) + GTX_DDR_SIZE env var parsing (D-02).

    Phase 1 scope: ensure_ddr() lazy growth + cap parsing.
    Phase 3 fills: ddr_init_from_file / ddr_dump_to_file (with GTX_DDR_REVERSED I/O — D-03).
    """
    from __future__ import annotations
    import os
    from typing import TYPE_CHECKING

    import numpy as np

    if TYPE_CHECKING:
        from .memory import GtxMemory   # avoid circular import at runtime

    # D-02 default: 4 GiB
    DEFAULT_DDR_SIZE: int = 4 * 1024 * 1024 * 1024


    def get_ddr_cap() -> int:
        """Read GTX_DDR_SIZE env var; default 4GB. Supports 'G'/'M'/'K' suffixes.

        Examples: '4G' -> 4*1024**3, '64M' -> 64*1024**2, '1024K' -> 1024*1024.
        """
        val = os.environ.get("GTX_DDR_SIZE")
        if val is None:
            return DEFAULT_DDR_SIZE
        val = val.strip().upper()
        if val.endswith("G"):
            return int(val[:-1]) * 1024 ** 3
        if val.endswith("M"):
            return int(val[:-1]) * 1024 ** 2
        if val.endswith("K"):
            return int(val[:-1]) * 1024
        return int(val)


    def ensure_ddr(mem: "GtxMemory", end_offset: int) -> np.ndarray:
        """Lazy DDR alloc. Phase 1 stub: allocates exactly end_offset (no doubling).

        D-01: DDR not pre-allocated at GtxMemory construction; first ensure_ddr() call
        materializes it.
        D-02: end_offset > GTX_DDR_SIZE cap -> ValueError (explicit, not silent truncation).

        Phase 3 will replace this with the C++ doubling-grow strategy matching
        gtx_npu_t::ensure_ddr.
        """
        cap = get_ddr_cap()
        if end_offset > cap:
            raise ValueError(
                f"DDR access {end_offset:#x} exceeds cap {cap:#x} "
                f"(set GTX_DDR_SIZE env var to raise)"
            )
        if mem._ddr_bytes is None or end_offset > mem._ddr_bytes.size:
            new_size = max(
                end_offset,
                mem._ddr_bytes.size if mem._ddr_bytes is not None else 0,
            )
            new_arr = np.zeros(new_size, dtype=np.uint8)
            if mem._ddr_bytes is not None:
                new_arr[:mem._ddr_bytes.size] = mem._ddr_bytes
            mem._ddr_bytes = new_arr
        return mem._ddr_bytes
    ```

    중요한 점:
    - `from __future__ import annotations` + `TYPE_CHECKING` import — `ddr.py`가
      `memory.py`를 import 하면 순환 import 위험. cp310+에서는 annotations는 자동
      lazy 평가되지만, 명시적 `__future__` 보장 + `TYPE_CHECKING`으로 런타임에는
      import 안 함. RESEARCH.md "Don't Hand-Roll" PEP 585 패턴.
    - DDR_DEFAULT 4GB는 D-02 명시. CI 압박 시 `GTX_DDR_SIZE=64M` 등으로 다운사이즈.
    - Phase 1은 end_offset 정확히 alloc (no doubling). Phase 3가 doubling-grow strategy로
      교체.
    - I/O 함수 (`ddr_init_from_file` / `ddr_dump_to_file`)는 이 plan에서 작성 안 함 — DMA-04
      (Phase 3) 책임.
  </action>
  <verify>
    <automated>test -f src/main/python/riscv/gtx/ddr.py &amp;&amp; PYTHONPATH=src/main/python python -c "from riscv.gtx.ddr import DEFAULT_DDR_SIZE, get_ddr_cap, ensure_ddr; from riscv.gtx.memory import GtxMemory; import numpy as np, os; assert DEFAULT_DDR_SIZE == 4*1024**3; assert get_ddr_cap() == 4*1024**3; m = GtxMemory(); assert m._ddr_bytes is None; arr = ensure_ddr(m, 1024); assert m._ddr_bytes is not None; assert arr.size >= 1024; os.environ['GTX_DDR_SIZE'] = '64M'; assert get_ddr_cap() == 64*1024*1024; del os.environ['GTX_DDR_SIZE']; print('OK')"</automated>
  </verify>
  <acceptance_criteria>
    - `test -f src/main/python/riscv/gtx/ddr.py` 종료코드 0
    - `grep -q 'DEFAULT_DDR_SIZE: int = 4 \* 1024 \* 1024 \* 1024' src/main/python/riscv/gtx/ddr.py` 종료코드 0 (D-02 default)
    - `grep -q 'def get_ddr_cap' src/main/python/riscv/gtx/ddr.py` 종료코드 0
    - `grep -q 'def ensure_ddr' src/main/python/riscv/gtx/ddr.py` 종료코드 0
    - `grep -q 'GTX_DDR_SIZE' src/main/python/riscv/gtx/ddr.py` 종료코드 0 (D-02 env var)
    - `grep -q 'os.environ.get' src/main/python/riscv/gtx/ddr.py` 종료코드 0
    - `grep -q 'raise ValueError' src/main/python/riscv/gtx/ddr.py` 종료코드 0 (cap 초과 시 명시적 에러)
    - `python -c "import ast; ast.parse(open('src/main/python/riscv/gtx/ddr.py').read())"` 종료코드 0
    - 위 `python -c "from riscv.gtx.ddr import …"` 명령이 "OK" + 종료코드 0
    - cap 초과 검증: `PYTHONPATH=src/main/python python -c "from riscv.gtx.ddr import ensure_ddr; from riscv.gtx.memory import GtxMemory; m = GtxMemory(); ensure_ddr(m, 5*1024**3)"` 종료코드 != 0 (4GB cap 초과로 ValueError)
  </acceptance_criteria>
  <done>ddr.py 존재 + DEFAULT_DDR_SIZE/get_ddr_cap/ensure_ddr 3개 export. 4GB default + cap 초과 ValueError + lazy alloc 모두 동작.</done>
</task>

</tasks>

<verification>
**Plan-level verification:**
- `PYTHONPATH=src/main/python pytest tests/gtx/test_memory_layout.py -v` → 8/8 PASS
- 3개 파일(memory.py, ddr.py, test_memory_layout.py) 모두 syntax valid
- `python -c "from riscv.gtx.memory import GtxMemory; from riscv.gtx.ddr import ensure_ddr"` 종료코드 0
- D-12 view-base 보장: 6개 named accessor (l0/l1/l2 byte+f16) 모두 `view.base is not None`
</verification>

<success_criteria>
1. `memory.py`에 `GtxMemory` 클래스 (L0/L1/L2 + spr dict + _ddr_bytes private attr)
2. `ddr.py`에 `DEFAULT_DDR_SIZE`, `get_ddr_cap`, `ensure_ddr` 3 export (Phase 3가 I/O 본체 채움)
3. `test_memory_layout.py` 8/8 PASS — D-17 LE byte order + D-12 view-base + D-11 SPR dict + D-01 DDR lazy 모두 검증
4. eager DDR alloc 코드 없음 (Anti-Pattern #2 방어)
5. `arr.copy()`로 view 깨는 코드 없음 (D-12 위반 방어)
</success_criteria>

<output>
After completion, create `.planning/phases/01-foundation/03-memory-SUMMARY.md` with:
- memory.py / ddr.py 정확한 LOC + exported 심볼 목록
- 8개 테스트 결과 (PASS/FAIL count + 타이밍)
- 메모리 사용량 측정: GtxMemory().__init__ 후 RSS 증가량 (예상 ~88MB; CI 압박 없음 확인)
- D-12 view-base tripwire가 모든 named accessor에 박힘 — Phase 4/5 op 핸들러가 in-place 쓰기 시 안전
</output>
