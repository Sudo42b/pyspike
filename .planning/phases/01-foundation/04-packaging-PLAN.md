---
phase: 01-foundation
plan: 04
type: execute
wave: 2
depends_on:
  - 01-foundation/01
  - 01-foundation/02
  - 01-foundation/03
  - 01-foundation/05
files_modified:
  - pyproject.toml
autonomous: true
requirements:
  - PKG-02
  - FOUND-03
must_haves:
  truths:
    - "`pyproject.toml [project].dependencies` contains `numpy>=2.0,<3` (D-07)"
    - "`pyproject.toml [project].requires-python == '>=3.10'` (D-08)"
    - "`pyproject.toml [tool.cibuildwheel].build` contains only cp310/cp311/cp312 (cp38/cp39 lines removed)"
    - "`pyproject.toml [project].classifiers` lists Python 3.10/3.11/3.12 only (3.8/3.9 removed)"
    - "`pyproject.toml [tool.setuptools.packages.find].include` is `['riscv', 'riscv.*']` (RESEARCH.md §2 critical fix; was `['riscv']`)"
    - "`pyproject.toml [tool.cibuildwheel.linux].before-all` runs `git submodule update --init --recursive` chained with `yum install -y dtc`"
    - "`pip wheel . -w /tmp/wheel-test/ ` produces a manylinux2014_x86_64 wheel that contains `riscv/gtx/__init__.py`"
    - "Building wheel and installing into a clean cp310 venv: `python -c 'from riscv.gtx import fp'` succeeds"
  artifacts:
    - path: "pyproject.toml"
      provides: "PEP 621 packaging — numpy dep, cp310+ baseline, cibuildwheel cp310-cp312, packages.find glob fix"
      contains: "numpy>=2.0,<3"
  key_links:
    - from: "pyproject.toml [tool.setuptools.packages.find].include"
      to: "src/main/python/riscv/gtx/"
      via: "['riscv', 'riscv.*'] glob discovers riscv.gtx subpackage in wheel"
      pattern: "include = \\[\\s*\"riscv\",\\s*\"riscv\\.\\*\""
    - from: "pyproject.toml [project].dependencies"
      to: "numpy 2.x runtime"
      via: "pip install pulls numpy>=2.0,<3"
      pattern: "numpy>=2\\.0,<3"
    - from: "pyproject.toml [tool.cibuildwheel.linux].before-all"
      to: "vendor/spike + vendor/gtx_cpp_reference submodules"
      via: "git submodule update --init --recursive"
      pattern: "git submodule update --init --recursive"
---

<objective>
`pyproject.toml`을 5곳 패치해서 PKG-02 + FOUND-03 (wheel discovery) 충족: NumPy 2.x
runtime dep, Python 3.10+ baseline, cibuildwheel matrix cp310-cp312로 축소,
classifiers에서 3.8/3.9 제거, **`packages.find.include` glob을 `['riscv', 'riscv.*']`
로 수정**(RESEARCH.md §"Critical Finding" — 이 한 줄 없으면 wheel에 `riscv.gtx`가 안
실린다), 그리고 `before-all`에 `git submodule update --init --recursive` 체이닝.

Purpose: D-07/D-08/D-09 NumPy/cp310 pivot을 빌드 시스템에 lock-in. RESEARCH.md
"Critical Finding" (`include = ["riscv"]`은 자동 recursive discovery 안 함 — empirically
verified on setuptools 80.9.0)을 직접 수정. Wave 1의 모든 모듈(`riscv/gtx/{__init__,
params, encoding, fp, memory, ddr}.py` + `ops/__init__.py`)이 wheel에 포함되어야 하므로
이 plan은 Wave 2이며 Wave 1 + Plan 05(submodule registration)에 의존.

Output: 1개 파일 (pyproject.toml) 5곳 패치. wheel 빌드 검증 명령 실행.
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
@pyproject.toml
@.planning/phases/01-foundation/01-skeleton-SUMMARY.md
@.planning/phases/01-foundation/02-fp-SUMMARY.md
@.planning/phases/01-foundation/03-memory-SUMMARY.md
@.planning/phases/01-foundation/05-submodule-SUMMARY.md
</context>

<tasks>

<task type="auto" tdd="false">
  <name>Task 04-01: pyproject.toml 5-stanza 패치 (NumPy 2.x / cp310 / packages.find glob fix)</name>
  <files>pyproject.toml</files>
  <read_first>
    - pyproject.toml (현재 상태 — 5곳 패치 적용 위치 확인. RESEARCH.md snapshot이 아닌 디스크 현재 상태가 진실)
    - .planning/phases/01-foundation/01-CONTEXT.md (D-07/D-08/D-09 lock-in 사항)
    - .planning/phases/01-foundation/01-RESEARCH.md "Example 3: Full pyproject.toml deltas" + "Critical Finding" (`include` glob 정확한 패턴)
    - .planning/phases/01-foundation/01-VALIDATION.md "Per-Task Verification Map" (04-packaging 5개 verify 명령)
  </read_first>
  <action>
    `pyproject.toml`에 다음 5개 변경을 정확히 적용. 각 변경은 RESEARCH.md "Example 3"
    diff와 일치해야 한다.

    **Edit 1: `[tool.cibuildwheel].build` — cp38/cp39 라인 제거**
    현재:
    ```toml
    [tool.cibuildwheel]
    build = [
      "cp38-manylinux_x86_64",
      "cp39-manylinux_x86_64",
      "cp310-manylinux_x86_64",
      "cp311-manylinux_x86_64",
      "cp312-manylinux_x86_64"
    ]
    ```
    변경 후:
    ```toml
    [tool.cibuildwheel]
    build = [
      "cp310-manylinux_x86_64",
      "cp311-manylinux_x86_64",
      "cp312-manylinux_x86_64"
    ]
    ```

    **Edit 2: `[tool.cibuildwheel.linux].before-all` — git submodule init 체이닝**
    현재:
    ```toml
    [tool.cibuildwheel.linux]
    before-all = "yum install -y dtc"
    ```
    변경 후:
    ```toml
    [tool.cibuildwheel.linux]
    before-all = "yum install -y dtc && git submodule update --init --recursive"
    ```

    **Edit 3: `[project].classifiers` — Python 3.8 / 3.9 줄 제거**
    현재 (잘 살펴 보고 정확히 두 줄 제거):
    ```toml
      "Programming Language :: Python :: 3",
      "Programming Language :: Python :: 3.8",
      "Programming Language :: Python :: 3.9",
      "Programming Language :: Python :: 3.10",
    ```
    변경 후:
    ```toml
      "Programming Language :: Python :: 3",
      "Programming Language :: Python :: 3.10",
    ```

    **Edit 4: `[project].requires-python` + 신규 `[project].dependencies`** (D-07/D-08)
    현재:
    ```toml
    requires-python = ">=3.8"

    [project.urls]
    ```
    변경 후:
    ```toml
    requires-python = ">=3.10"

    dependencies = [
      "numpy>=2.0,<3",
    ]

    [project.urls]
    ```
    `dependencies`는 `requires-python` 바로 다음, `[project.urls]` 직전에 빈 줄 한 줄로
    구분 (PEP 621 권장 ordering — RESEARCH.md "Open Questions" #1).

    **Edit 5: `[tool.setuptools.packages.find].include` — riscv.gtx 발견을 위한 glob 수정** (★ CRITICAL)
    현재:
    ```toml
    [tool.setuptools.packages.find]
    where = [
      "src/main/python"
    ]
    include = [
      "riscv"
    ]
    ```
    변경 후:
    ```toml
    [tool.setuptools.packages.find]
    where = [
      "src/main/python"
    ]
    include = [
      "riscv",
      "riscv.*"
    ]
    ```

    중요한 점:
    - **Edit 5가 Phase 1 전체에서 가장 위험한 단일 라인**. 이 한 줄을 빠뜨리면 wheel에
      `riscv.gtx` 서브패키지가 포함되지 않아 `pip install spike`한 사용자가
      `from riscv.gtx import fp`로 ModuleNotFoundError를 만난다 — Phase 1 success
      criterion 3 silent fail (RESEARCH.md "Critical Finding" 참조). CONTEXT.md
      `code_context`의 "자동으로 riscv.gtx 발견" 주장은 잘못됨.
    - cibuildwheel `before-all`은 `git submodule update --init --recursive`를 추가해야
      `vendor/spike` (기존) + `vendor/gtx_cpp_reference` (Plan 05에서 등록) 모두 manylinux
      컨테이너 안에서 초기화된다. 단순 `&&` 체이닝 — 추가 wrapper 스크립트 불필요.
    - `dependencies`는 NEW key (현재 pyproject.toml에는 없음). 위치는 `requires-python`
      직후로 lock — RESEARCH.md "Open Questions" #1에서 권장.
    - `cp38`/`cp39` 라인을 cibuildwheel `build` array와 classifiers 양쪽 모두에서 제거 —
      한쪽만 제거하면 cibuildwheel이 cp38 wheel을 build 시도하다 NumPy 2.x 미지원으로
      실패한다.
    - 다른 stanza (`[build-system]`, `[tool.pytest.ini_options]`, `[tool.mypy]`,
      `[tool.pylint.*]`, `[tool.coverage.*]`, `[tool.setuptools.package-data]`,
      `[tool.setuptools_scm]`, `[project.optional-dependencies]`, `[project.urls]`)는
      변경하지 않음 — RESEARCH.md "Example 3" diff에 포함되지 않음.
  </action>
  <verify>
    <automated>python -c "import tomllib; t = tomllib.load(open('pyproject.toml','rb')); deps = t['project']['dependencies']; assert any('numpy>=2.0' in d and '<3' in d for d in deps), f'numpy>=2.0,<3 missing from {deps}'; assert t['project']['requires-python'] == '>=3.10', f\"requires-python={t['project']['requires-python']}\"; b = t['tool']['cibuildwheel']['build']; assert all(('cp310' in x or 'cp311' in x or 'cp312' in x) for x in b), f'cibuildwheel build={b}'; assert not any('cp38' in x or 'cp39' in x for x in b), f'cibuildwheel still has cp38/cp39: {b}'; cls = t['project']['classifiers']; assert any('3.10' in c for c in cls); assert not any('3.8' in c or '3.9' in c for c in cls if 'Python' in c), f'classifiers still have 3.8/3.9: {[c for c in cls if \"Python\" in c]}'; inc = t['tool']['setuptools']['packages']['find']['include']; assert 'riscv.*' in inc or any('riscv*' == x for x in inc), f'include glob missing recursive pattern: {inc}'; ba = t['tool']['cibuildwheel']['linux']['before-all']; assert 'git submodule update --init --recursive' in ba, f'before-all missing submodule init: {ba}'; print('OK pyproject.toml validation passed')"</automated>
  </verify>
  <acceptance_criteria>
    - 위 `python -c "import tomllib"` 검증 명령이 "OK pyproject.toml validation passed" 출력 + 종료코드 0
    - `python -c "import tomllib; t=tomllib.load(open('pyproject.toml','rb')); assert any('numpy>=2.0' in d for d in t['project']['dependencies'])"` 종료코드 0
    - `python -c "import tomllib; t=tomllib.load(open('pyproject.toml','rb')); assert t['project']['requires-python'] == '>=3.10'"` 종료코드 0
    - `python -c "import tomllib; t=tomllib.load(open('pyproject.toml','rb')); b=t['tool']['cibuildwheel']['build']; assert all('cp31' in x for x in b) and not any('cp38' in x or 'cp39' in x for x in b)"` 종료코드 0
    - `grep -F '"riscv.*"' pyproject.toml` 종료코드 0 (Edit 5 적용됨)
    - `grep -F 'git submodule update --init --recursive' pyproject.toml` 종료코드 0 (Edit 2 적용됨)
    - `grep -F '"cp38-manylinux_x86_64"' pyproject.toml` 종료코드 1 (cp38 라인 제거 확인)
    - `grep -F '"cp39-manylinux_x86_64"' pyproject.toml` 종료코드 1 (cp39 라인 제거 확인)
    - `grep -F 'Python :: 3.8' pyproject.toml` 종료코드 1 (classifier 3.8 제거 확인)
    - `grep -F 'Python :: 3.9' pyproject.toml` 종료코드 1 (classifier 3.9 제거 확인)
    - `python -c "import tomllib; tomllib.load(open('pyproject.toml','rb'))"` 종료코드 0 (TOML syntax valid)
  </acceptance_criteria>
  <done>pyproject.toml 5곳 패치 완료. tomllib 검증 명령 5개 모두 통과. cp38/cp39 흔적 없음. `riscv.*` glob 존재. `git submodule update --init --recursive` 체이닝 존재.</done>
</task>

<task type="auto" tdd="false">
  <name>Task 04-02: wheel 빌드 + riscv.gtx 포함 검증 (clean cp310 venv install + import)</name>
  <files></files>
  <read_first>
    - pyproject.toml (Task 04-01 완료 후 패치된 상태)
    - src/main/python/riscv/gtx/__init__.py (Wave 1 Plan 01 출력 — 이 파일이 wheel 안에 들어가야 함)
    - src/main/python/riscv/gtx/fp.py (Wave 1 Plan 02 출력)
    - src/main/python/riscv/gtx/memory.py (Wave 1 Plan 03 출력)
    - .planning/phases/01-foundation/05-submodule-SUMMARY.md (Plan 05 출력 — vendor/gtx_cpp_reference이 wheel에 포함되지 않아야 함)
    - .planning/phases/01-foundation/01-VALIDATION.md "Per-Task Verification Map" + "Manual-Only Verifications" (cibuildwheel CI는 manual)
  </read_first>
  <action>
    Wave 1의 모든 모듈 + Plan 05의 MANIFEST.in 패치가 적용된 상태에서 wheel 빌드를
    실행하고 `riscv.gtx`가 포함됨을 검증한다. 이 task는 새 파일 생성 없이 검증 명령만
    실행한다.

    **Step 1**: 빌드 환경 정리 후 wheel build:
    ```bash
    rm -rf /tmp/wheel-test/ build/ dist/ src/main/python/riscv/_version.py
    pip wheel . -w /tmp/wheel-test/ --no-deps -v 2>&1 | tail -50
    ```
    `--no-deps`로 numpy 의존성 해결을 skip (build-time에는 numpy 불필요; install 시점에
    pip이 알아서 처리). `-v`로 build 로그 확인 (riscv.gtx 미포함 시 build 로그에서
    경고 예상).

    **Step 2**: 빌드된 wheel에 `riscv/gtx/__init__.py`가 들어 있음을 확인:
    ```bash
    unzip -l /tmp/wheel-test/spike-*.whl | grep -E 'riscv/gtx/(__init__|params|encoding|fp|memory|ddr)\.py'
    ```
    예상 출력 (각 파일이 한 줄씩):
    ```
    ...riscv/gtx/__init__.py
    ...riscv/gtx/params.py
    ...riscv/gtx/encoding.py
    ...riscv/gtx/fp.py
    ...riscv/gtx/memory.py
    ...riscv/gtx/ddr.py
    ...riscv/gtx/ops/__init__.py
    ```
    **만약 위 6+1개 파일이 wheel에 안 보이면 Edit 5(`include = ["riscv", "riscv.*"]`)
    누락이 의심됨 — Task 04-01 acceptance criteria 다시 확인.**

    **Step 3**: vendor/gtx_cpp_reference이 wheel에 포함되지 않음 확인 (D-06):
    ```bash
    unzip -l /tmp/wheel-test/spike-*.whl | grep -c gtx_cpp_reference
    ```
    예상: 출력 `0` — wheel에 vendor 디렉토리 미포함.

    **Step 4**: Clean cp310 venv에서 install + import test:
    ```bash
    python3.10 -m venv /tmp/p1venv
    /tmp/p1venv/bin/pip install /tmp/wheel-test/spike-*.whl
    /tmp/p1venv/bin/python -c "from riscv.gtx import fp, memory, params; from riscv.gtx.params import GTX_NEST_NUM; assert GTX_NEST_NUM == 4; print('Phase 1 wheel import OK')"
    ```
    Python 3.10이 시스템에 없으면 `python3 -m venv /tmp/p1venv` (현재 Python)로 fallback,
    단 결과는 cp310 wheel과 다를 수 있음을 노트에 기록. Manual cibuildwheel 검증은 CI
    런에서 (`01-VALIDATION.md` "Manual-Only Verifications" 참조).

    **Step 5**: auditwheel manylinux2014 호환성 검증 (선택적이지만 권장):
    ```bash
    auditwheel show /tmp/wheel-test/spike-*.whl | grep manylinux2014_x86_64 || echo "auditwheel result above"
    ```

    중요한 점:
    - 이 task는 **검증 task** — 새 파일 생성 없음. 모든 부작용은 `/tmp/wheel-test/`,
      `/tmp/p1venv/`, `dist/`, `build/`로 격리.
    - `pip wheel .`이 실패하면 Task 04-01의 패치가 잘못된 것 — `pyproject.toml`을 다시
      검사.
    - `setuptools_scm`이 git에서 version 추론 — 이 task 시점에는 `git status` 깨끗할
      필요 없으나 dirty면 wheel 이름이 `+dirty` suffix가 붙음 (`local_scheme =
      "no-local-version"` 설정으로 회피되어 있음).
  </action>
  <verify>
    <automated>cd /mnt/e/14_NIGHTLY/pyspike &amp;&amp; rm -rf /tmp/wheel-test/ &amp;&amp; pip wheel . -w /tmp/wheel-test/ --no-deps -q 2>&amp;1 | tail -20 &amp;&amp; ls /tmp/wheel-test/spike-*.whl &amp;&amp; unzip -l /tmp/wheel-test/spike-*.whl | grep -q 'riscv/gtx/__init__.py' &amp;&amp; unzip -l /tmp/wheel-test/spike-*.whl | grep -q 'riscv/gtx/fp.py' &amp;&amp; unzip -l /tmp/wheel-test/spike-*.whl | grep -q 'riscv/gtx/memory.py' &amp;&amp; unzip -l /tmp/wheel-test/spike-*.whl | grep -q 'riscv/gtx/params.py' &amp;&amp; unzip -l /tmp/wheel-test/spike-*.whl | grep -q 'riscv/gtx/ddr.py' &amp;&amp; unzip -l /tmp/wheel-test/spike-*.whl | grep -q 'riscv/gtx/encoding.py' &amp;&amp; unzip -l /tmp/wheel-test/spike-*.whl | grep -q 'riscv/gtx/ops/__init__.py' &amp;&amp; test "$(unzip -l /tmp/wheel-test/spike-*.whl | grep -c gtx_cpp_reference)" = "0" &amp;&amp; echo OK</automated>
  </verify>
  <acceptance_criteria>
    - `pip wheel . -w /tmp/wheel-test/ --no-deps` 종료코드 0 (wheel build 성공)
    - `ls /tmp/wheel-test/spike-*.whl` 적어도 1개 파일 매칭
    - `unzip -l /tmp/wheel-test/spike-*.whl | grep -q 'riscv/gtx/__init__.py'` 종료코드 0 (★ FOUND-03 핵심 — wheel에 패키지 포함)
    - `unzip -l /tmp/wheel-test/spike-*.whl | grep -q 'riscv/gtx/fp.py'` 종료코드 0
    - `unzip -l /tmp/wheel-test/spike-*.whl | grep -q 'riscv/gtx/memory.py'` 종료코드 0
    - `unzip -l /tmp/wheel-test/spike-*.whl | grep -q 'riscv/gtx/params.py'` 종료코드 0
    - `unzip -l /tmp/wheel-test/spike-*.whl | grep -q 'riscv/gtx/ddr.py'` 종료코드 0
    - `unzip -l /tmp/wheel-test/spike-*.whl | grep -q 'riscv/gtx/encoding.py'` 종료코드 0
    - `unzip -l /tmp/wheel-test/spike-*.whl | grep -q 'riscv/gtx/ops/__init__.py'` 종료코드 0
    - `unzip -l /tmp/wheel-test/spike-*.whl | grep -c gtx_cpp_reference` 출력 == 0 (D-06: vendor 미포함)
    - (manual / 권장) `auditwheel show /tmp/wheel-test/spike-*.whl` 출력에 `manylinux2014_x86_64` 표시
    - (manual / 권장) `python3.10` 사용 가능 시: `python3.10 -m venv /tmp/p1venv && /tmp/p1venv/bin/pip install /tmp/wheel-test/spike-*.whl && /tmp/p1venv/bin/python -c "from riscv.gtx import fp"` 종료코드 0 — 시스템에 cp310 없으면 manual 단계로 record (01-VALIDATION.md "Manual-Only Verifications" 참조)
  </acceptance_criteria>
  <done>pip wheel 빌드 성공. wheel 안에 7개 `riscv.gtx` 파일 모두 포함. vendor/gtx_cpp_reference은 미포함 (count 0). FOUND-03(wheel 동봉) acceptance 충족.</done>
</task>

</tasks>

<verification>
**Plan-level verification:**
- Task 04-01 검증: `python -c "import tomllib; ..."` 5개 어서션 모두 통과 (numpy>=2.0, requires-python>=3.10, cibuildwheel cp310-cp312, classifiers 3.10/3.11/3.12, packages.find include glob, before-all 체이닝)
- Task 04-02 검증: wheel build 성공 + `riscv.gtx` 7개 파일 wheel 동봉 + `gtx_cpp_reference` count 0
- 통합: Phase 1 ROADMAP 성공 기준 5 (`pyproject.toml` 선언 + 유효한 manylinux2014_x86_64 wheel) 충족

**의존성 확인:** 이 plan은 Wave 1 모든 plan + Plan 05(submodule registration → MANIFEST.in
prune)에 의존. Plan 05의 MANIFEST.in 변경 없이 Task 04-02 wheel 빌드를 하면 sdist에
vendor/gtx_cpp_reference이 포함되지만 wheel에는 영향 없음 (wheel은 [tool.setuptools.package-data]
기준으로 포함 여부 결정 — vendor는 거기 선언 안 됨). 따라서 Task 04-02 자체는 Plan 05 완료
없이도 통과 가능 — 다만 "통합 acceptance"를 위해 Plan 05도 같은 Wave 2에 둠.
</verification>

<success_criteria>
1. `pyproject.toml` 5곳 패치: cibuildwheel cp310-cp312 only, before-all submodule 체이닝, classifiers 3.10/11/12 only, requires-python >= 3.10, dependencies = ["numpy>=2.0,<3"], packages.find.include = ["riscv", "riscv.*"]
2. `pip wheel .` 종료코드 0
3. `unzip -l dist/...whl | grep riscv/gtx/__init__.py` 매칭 (FOUND-03 wheel 동봉 확인)
4. `unzip -l dist/...whl | grep gtx_cpp_reference` count 0 (D-06 wheel 미포함 확인)
5. tomllib 어서션 5/5 통과 (PKG-02 모든 구성 요소)
</success_criteria>

<output>
After completion, create `.planning/phases/01-foundation/04-packaging-SUMMARY.md` with:
- 5개 stanza 패치의 정확한 변경 라인 (before/after diff)
- 빌드된 wheel 파일명 + 사이즈
- wheel 안의 `riscv.gtx` 파일 목록 (unzip -l 출력)
- vendor/gtx_cpp_reference 미포함 확인 (count 0)
- Manual cibuildwheel CI 검증은 deferred (다음 PR push 시 GitHub Actions 결과 확인) — 01-VALIDATION.md "Manual-Only Verifications" 참조
</output>
