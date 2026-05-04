---
phase: 01-foundation
plan: 05
type: execute
wave: 1
depends_on: []
files_modified:
  - .gitmodules
  - MANIFEST.in
autonomous: true
requirements:
  - FOUND-04
must_haves:
  truths:
    - "`vendor/gtx_cpp_reference/` is registered as a git submodule pointing to `https://github.com/Sudo42b/gtx_spike(.git)?` (D-04)"
    - "`git submodule status` lists `vendor/gtx_cpp_reference` with a SHA and path"
    - "`.gitmodules` contains both `submodule \"vendor/spike\"` (pre-existing) and `submodule \"vendor/gtx_cpp_reference\"` (new)"
    - "`MANIFEST.in` contains a line `prune vendor/gtx_cpp_reference` AFTER the existing `recursive-include vendor *` (D-06)"
    - "`python -m build --sdist` produces a tarball that does NOT contain any `vendor/gtx_cpp_reference/*` entry"
  artifacts:
    - path: ".gitmodules"
      provides: "Submodule registration metadata (path + url for vendor/gtx_cpp_reference)"
      contains: "vendor/gtx_cpp_reference"
    - path: "MANIFEST.in"
      provides: "sdist exclusion of vendor/gtx_cpp_reference (D-06)"
      contains: "prune vendor/gtx_cpp_reference"
    - path: "vendor/gtx_cpp_reference/"
      provides: "C++ ground-truth submodule (D-04/D-05) — gtx/ + spike patches"
  key_links:
    - from: ".gitmodules"
      to: "https://github.com/Sudo42b/gtx_spike"
      via: "url field of submodule.vendor/gtx_cpp_reference"
      pattern: "url = https://github.com/Sudo42b/gtx_spike"
    - from: "MANIFEST.in"
      to: "vendor/gtx_cpp_reference/"
      via: "prune directive (sdist exclude per D-06)"
      pattern: "^prune vendor/gtx_cpp_reference"
---

<objective>
C++ ground-truth(`gtx/` 디렉토리 + spike 패치)를 git submodule로 `vendor/gtx_cpp_reference/`에
등록하고, sdist에서 해당 디렉토리를 prune하도록 `MANIFEST.in`을 패치한다. 이 plan은
디스크에 새 파일을 작성하지 않고 (`git submodule add`가 디렉토리를 자동 생성), 기존
`MANIFEST.in`에 단 한 줄(`prune vendor/gtx_cpp_reference`)을 추가한다.

Purpose: D-04 (submodule URL = https://github.com/Sudo42b/gtx_spike), D-05 (scope =
gtx/ + spike patches), D-06 (wheel/sdist 미포함) lock-in. P4/P5 strict-mode 측정 시
C++ 동작 비교 빌드를 위한 ground-truth 확보 (개발자 환경에서만; 사용자가 `pip install
spike` 받을 때는 미포함).

Output: `.gitmodules` 갱신 + `vendor/gtx_cpp_reference/` 디렉토리 등록 (자동, 본 plan은
콘텐츠를 작성하지 않음) + `MANIFEST.in` 한 줄 추가.

Wave 2지만 Plan 04와 병렬 실행 가능 (다른 파일 — pyproject.toml vs MANIFEST.in/.gitmodules).
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
@MANIFEST.in
</context>

<tasks>

<task type="auto" tdd="false">
  <name>Task 05-01: git submodule add — vendor/gtx_cpp_reference 등록</name>
  <files>.gitmodules</files>
  <read_first>
    - .planning/phases/01-foundation/01-CONTEXT.md (D-04 URL = https://github.com/Sudo42b/gtx_spike, 공개 레포)
    - .planning/phases/01-foundation/01-RESEARCH.md "Example 1: Adding the C++ reference submodule" (정확한 명령 + 검증)
    - .planning/phases/01-foundation/01-RESEARCH.md "Subtle but important caveat" (cibuildwheel + sdist 상호작용)
    - .planning/phases/01-foundation/01-VALIDATION.md "Manual-Only Verifications" (URL pin 검증은 manual)
    - .gitmodules (현재 상태 — vendor/spike 항목 존재 확인)
  </read_first>
  <action>
    프로젝트 루트(`/mnt/e/14_NIGHTLY/pyspike`)에서 다음 단일 명령으로 submodule 등록:

    ```bash
    git submodule add https://github.com/Sudo42b/gtx_spike vendor/gtx_cpp_reference
    ```

    이 명령은:
    1. `.gitmodules` 파일에 `[submodule "vendor/gtx_cpp_reference"]` stanza 추가
       (path = vendor/gtx_cpp_reference, url = https://github.com/Sudo42b/gtx_spike)
    2. `vendor/gtx_cpp_reference/` 디렉토리에 원격 레포 clone (default branch HEAD)
    3. git index에 submodule entry 추가 (parent 레포가 SHA를 추적)

    **중요한 점:**
    - URL은 D-04에서 명시된 정확한 값: `https://github.com/Sudo42b/gtx_spike` — 공개
      레포라 익명 clone 가능 (CI에서도 동작; SSH key/PAT 불필요).
    - `-b` (branch tracking) 플래그는 사용하지 않음 — RESEARCH.md "Note on `--branch`"
      참조: ground-truth는 moving target이 아닌 commit-pinned가 적절. parent 레포가
      커밋한 SHA만 `git submodule update --init`으로 복원됨.
    - Phase 1 시점에는 default branch HEAD를 그대로 사용. 향후 `cd
      vendor/gtx_cpp_reference; git checkout <sha>; cd -; git add vendor/gtx_cpp_reference`로
      특정 커밋 pin 가능하지만 이 task에서는 안 함 (별도 chore 커밋이 자연스러움 —
      RESEARCH.md "Example 1" 마지막 코멘트).
    - `git submodule add`가 실패할 경우 (네트워크 / 권한 / 이미 존재):
      - 이미 등록되어 있으면 `git submodule status | grep gtx_cpp_reference` 결과로 확인 후 skip
      - 네트워크 실패 시 명시적 에러로 중단 — 우회 stub 만들지 않음
      - 자격 증명 문제는 D-04 "공개 레포" 가정 위반 → user에게 escalation
    - `vendor/spike` (기존 submodule)는 변경하지 않음 — `.gitmodules`에 새 stanza만 추가.

    **검증 step (이 task 수행 후 즉시 확인):**
    ```bash
    # 1. .gitmodules에 새 항목 등록 확인
    git config -f .gitmodules submodule.vendor/gtx_cpp_reference.url

    # 2. submodule status 확인
    git submodule status | grep gtx_cpp_reference

    # 3. 디렉토리에 콘텐츠 존재 확인
    ls vendor/gtx_cpp_reference/.git
    ```

    예상 출력:
    1. `https://github.com/Sudo42b/gtx_spike` (정확히 그 URL)
    2. `<SHA> vendor/gtx_cpp_reference (heads/main 또는 tag)` (앞 +/- 또는 공백)
    3. 파일 (`.git`은 file이며 `gitdir: ...`을 가리킴 — git submodule의 표준)

    **CI 영향:** `[tool.cibuildwheel.linux].before-all`이 Plan 04에서 `git submodule
    update --init --recursive`로 패치됨 — 이 submodule도 함께 init됨. 따라서 manylinux
    컨테이너 안에서도 `vendor/gtx_cpp_reference/`가 존재. 하지만 wheel에는 미포함
    (`[tool.setuptools.package-data]`에 미선언, MANIFEST.in `prune` — Task 05-02).
  </action>
  <verify>
    <automated>git submodule status | grep -q gtx_cpp_reference &amp;&amp; git config -f .gitmodules submodule.vendor/gtx_cpp_reference.url | grep -qE '^https://github\.com/Sudo42b/gtx_spike(\.git)?$' &amp;&amp; git config -f .gitmodules submodule.vendor/gtx_cpp_reference.path | grep -q '^vendor/gtx_cpp_reference$' &amp;&amp; test -e vendor/gtx_cpp_reference/.git &amp;&amp; echo OK</automated>
  </verify>
  <acceptance_criteria>
    - `git submodule status | grep -q gtx_cpp_reference` 종료코드 0
    - `git config -f .gitmodules submodule.vendor/gtx_cpp_reference.url` 출력이 `https://github.com/Sudo42b/gtx_spike` 또는 `https://github.com/Sudo42b/gtx_spike.git` (D-04 URL 정확히 일치)
    - `git config -f .gitmodules submodule.vendor/gtx_cpp_reference.path` 출력 == `vendor/gtx_cpp_reference`
    - `test -e vendor/gtx_cpp_reference/.git` 종료코드 0 (submodule 콘텐츠 mounted)
    - `git config -f .gitmodules submodule.vendor/spike.url` 종료코드 0 (기존 submodule unchanged)
    - `grep -c '\[submodule' .gitmodules` 출력 >= 2 (vendor/spike + vendor/gtx_cpp_reference)
  </acceptance_criteria>
  <done>vendor/gtx_cpp_reference이 git submodule로 등록됨; URL이 https://github.com/Sudo42b/gtx_spike(.git)?과 정확히 일치; 디렉토리 mounted; 기존 vendor/spike submodule은 unchanged.</done>
</task>

<task type="auto" tdd="false">
  <name>Task 05-02: MANIFEST.in 패치 — vendor/gtx_cpp_reference sdist exclusion (D-06)</name>
  <files>MANIFEST.in</files>
  <read_first>
    - MANIFEST.in (현재 상태 — 16라인, line 13에 `recursive-include vendor *` 존재)
    - .planning/phases/01-foundation/01-CONTEXT.md (D-06: vendor/gtx_cpp_reference은 wheel에 미포함)
    - .planning/phases/01-foundation/01-RESEARCH.md "Pitfall (Phase 1-specific): MANIFEST.in recursive-include vendor *" (정확한 패치 위치 + prune 사용 이유)
    - .planning/phases/01-foundation/01-VALIDATION.md "05-submodule" 행 (sdist exclusion 검증)
  </read_first>
  <action>
    `MANIFEST.in`에 다음 단일 라인을 추가. 정확한 위치는 `recursive-include vendor *`
    (line 13) **다음**, `recursive-exclude . __pycache__ ...` (line 15) **이전**:

    추가할 라인:
    ```
    prune vendor/gtx_cpp_reference
    ```

    변경 후 MANIFEST.in 전체:
    ```
    include LICENSE
    include MANIFEST.in
    include pyproject.toml
    include README.md
    include riscv.pth
    include setup.py
    recursive-include docs *.puml *.md
    recursive-include src/main/cpp *.h *.cc
    recursive-include src/main/python *.py *.pyi py.typed
    recursive-include tests *.py *.pyi
    recursive-include examples *
    recursive-include tests/data *.py *.elf
    recursive-include vendor *
    recursive-exclude src/main/python/riscv/data *
    prune vendor/gtx_cpp_reference
    recursive-exclude . __pycache__ *.pyc *.pyo .gitignore .DS_Store .coverage .mypy_cache .tox .pytest_cache *.egg-info
    ```

    **중요한 점:**
    - `prune` directive는 setuptools MANIFEST.in 의 정식 디렉토리 트리 제외 메커니즘
      이며 `recursive-include vendor *`보다 나중에 와야 효과적임 (RESEARCH.md "Pitfall
      Phase-1 specific"). `recursive-exclude vendor/gtx_cpp_reference *`로 등가 표현
      가능하지만 `prune`이 canonical (setuptools 공식 docs).
    - `prune`은 sdist에만 영향. wheel은 `[tool.setuptools.package-data]`에 의해 결정
      (vendor/gtx_cpp_reference가 거기 선언되지 않으므로 wheel에도 자동 미포함).
      따라서 sdist + wheel 양쪽에서 D-06 충족.
    - `vendor/spike` (기존 submodule)는 영향 받지 않음 — `prune`은 정확히 지정된 경로만
      제외.
    - `recursive-include tests/data *.py *.elf` 항목은 P3+에서 elf 파일 추가 시 사용 —
      Phase 1은 건드리지 않음.
    - 라인 추가는 한 줄만 — 다른 라인은 unchanged.

    **검증 step (이 task 수행 후 즉시):**
    ```bash
    # 1. prune 라인 존재 확인
    grep -q '^prune vendor/gtx_cpp_reference$' MANIFEST.in

    # 2. 위치 확인 — recursive-include vendor 다음에
    awk '/^recursive-include vendor \*/{found=1} /^prune vendor\/gtx_cpp_reference/{if(found) print "OK position"; else print "FAIL position"; exit}' MANIFEST.in

    # 3. sdist 빌드 후 검증 (선택적 — 실제 sdist 생성)
    python -m build --sdist 2>&1 | tail -5
    tar tzf dist/spike-*.tar.gz | grep -c gtx_cpp_reference
    # 예상: 0
    ```
  </action>
  <verify>
    <automated>grep -q '^prune vendor/gtx_cpp_reference$' MANIFEST.in &amp;&amp; awk '/^recursive-include vendor \*/{found=1} /^prune vendor\/gtx_cpp_reference/{if(found) {print "OK"; exit 0} else {print "FAIL_ORDER"; exit 1}}' MANIFEST.in</automated>
  </verify>
  <acceptance_criteria>
    - `grep -q '^prune vendor/gtx_cpp_reference$' MANIFEST.in` 종료코드 0 (정확히 한 줄, 시작 위치 lock)
    - `awk` 명령으로 `prune vendor/gtx_cpp_reference`가 `recursive-include vendor *` **다음**에 위치 — 위 verify 명령에서 "OK" 출력 + 종료코드 0
    - `wc -l MANIFEST.in` 출력 == 17 (기존 16 + 신규 1)
    - 기존 라인들 변경 없음:
      - `grep -q '^include LICENSE$' MANIFEST.in` 종료코드 0
      - `grep -q '^recursive-include vendor \*$' MANIFEST.in` 종료코드 0
      - `grep -q '^recursive-exclude src/main/python/riscv/data \*$' MANIFEST.in` 종료코드 0
    - (manual / 권장) sdist 빌드 후 `tar tzf dist/spike-*.tar.gz | grep -c gtx_cpp_reference` 출력 == 0 (실제 sdist exclusion 검증)
  </acceptance_criteria>
  <done>MANIFEST.in에 `prune vendor/gtx_cpp_reference` 한 줄 추가됨 (recursive-include vendor * 다음 위치). 기존 라인 모두 unchanged. sdist 빌드 시 vendor/gtx_cpp_reference이 미포함됨.</done>
</task>

</tasks>

<verification>
**Plan-level verification:**
- `git submodule status | grep gtx_cpp_reference` 매칭 (Task 05-01)
- `.gitmodules` URL이 `https://github.com/Sudo42b/gtx_spike(.git)?`와 정확히 일치 (Task 05-01)
- `grep -q '^prune vendor/gtx_cpp_reference$' MANIFEST.in` 매칭 (Task 05-02)
- `awk` order-check 명령이 "OK" 출력 (Task 05-02)
- (manual) `python -m build --sdist && tar tzf dist/spike-*.tar.gz | grep -c gtx_cpp_reference` 출력 == 0 (D-06 sdist exclusion 검증)
- (manual) Plan 04의 wheel 빌드 후 `unzip -l ... | grep -c gtx_cpp_reference` 출력 == 0 (D-06 wheel exclusion — Plan 04 Task 04-02에서 검증됨)
</verification>

<success_criteria>
1. vendor/gtx_cpp_reference이 git submodule로 등록 (URL 정확히 D-04 값)
2. .gitmodules에 vendor/spike + vendor/gtx_cpp_reference 두 항목 존재
3. MANIFEST.in에 `prune vendor/gtx_cpp_reference` 한 줄 추가됨 (정확한 위치)
4. sdist 빌드 시 vendor/gtx_cpp_reference 콘텐츠 미포함 (count 0)
5. (Plan 04와 결합) wheel에도 미포함 — D-06 충족
</success_criteria>

<output>
After completion, create `.planning/phases/01-foundation/05-submodule-SUMMARY.md` with:
- `git submodule status` 출력 (vendor/spike + vendor/gtx_cpp_reference 두 줄)
- `git config -f .gitmodules submodule.vendor/gtx_cpp_reference.url` 출력 (D-04 URL 검증)
- 등록된 submodule HEAD SHA (chore 커밋으로 향후 pin 시 참조)
- MANIFEST.in 변경 diff (한 줄 추가)
- (선택적) sdist 빌드 결과: tarball 사이즈 + gtx_cpp_reference 항목 count 0 확인
</output>
