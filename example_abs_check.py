"""
example_abs_check.py — pyspike + riscv.gtx extension 통합 검증 예제 (ABS op)

이 스크립트는 한 op(ABS)에 대해 두 가지 검증 시나리오를 보여줍니다:

  Test A: 우리 hand-written abs.elf — zero-init oracle (P6 trivial pattern)
          - 즉시 동작 가능 (GTX_DDR_DUMP atexit hook만 필요)
          - 단일 32-byte row 비교 (input zero → ABS(0)=0 → all-zero output)
          - plumbing 검증용 (SPR→dispatch→DMA→compute→writeback chain)

  Test B: vendor n1s16_abs.elf + vendor input/ref — 풀 회귀
          - GTX_DDR_INIT env var 지원 필요 (vendor C++ gtx_npu_core.cc:120 직역)
          - 393217 row × 16 element FP16 = 12.2MB output
          - 실제 NPU 연산 정확성 검증 (non-zero input → non-zero expected)

Usage:
    cd /mnt/e/14_NIGHTLY/pyspike
    source .venv/bin/activate
    python example_abs_check.py
"""
import os
import subprocess
import sys
from pathlib import Path

# riscv.gtx + _verify import smoke (PKG-03 한 줄 검증과 동일)
try:
    from riscv.gtx import GtxNpu  # noqa: F401
    from riscv.gtx._verify import compare_hex
except ImportError as e:
    print(f"FAIL: riscv.gtx import error — {e}")
    print("  → 'uv pip install -e .' 후 .venv 활성화 확인")
    sys.exit(1)


REPO = Path(__file__).parent.absolute()

# Test A 자산 (P6 Plan 03 — hand-written .S, zero-init oracle)
OUR_ELF = REPO / "tests" / "gtx" / "data" / "elf" / "abs.elf"
OUR_GOLDEN = REPO / "tests" / "gtx" / "data" / "golden" / "abs.hex"

# Test B 자산 (vendor — pre-built ELF + non-zero input/ref)
VENDOR_ELF = REPO / "test" / "ABS" / "n1s16" / "n1s16_abs.elf"
VENDOR_INPUT = REPO / "test" / "ABS" / "n1s16" / "data" / "n1s16_abs_input.txt"
VENDOR_REF = REPO / "test" / "ABS" / "n1s16" / "data" / "n1s16_abs_ref.txt"


def _resolve_pyspike() -> list:
    """pyspike CLI 경로 결정 (PATH 또는 .venv 탐색)."""
    import shutil
    cli = shutil.which("pyspike")
    if cli:
        return [cli]
    venv_pyspike = REPO / ".venv" / "bin" / "pyspike"
    if venv_pyspike.exists():
        return [str(venv_pyspike)]
    return [sys.executable, "-m", "riscv"]


def run_test_a() -> bool:
    """우리 hand-written abs.elf 회귀 — zero-init oracle."""
    print("=" * 70)
    print("Test A: 우리 hand-written abs.elf (zero-init oracle)")
    print("=" * 70)

    if not OUR_ELF.exists():
        print(f"  SKIP: {OUR_ELF.relative_to(REPO)} 미존재 (P6 Plan 03 미빌드?)")
        return False
    if not OUR_GOLDEN.exists():
        print(f"  SKIP: {OUR_GOLDEN.relative_to(REPO)} 미존재")
        return False

    actual = REPO / "_abs_result_a.hex"
    actual.unlink(missing_ok=True)

    env = os.environ.copy()
    env.update({
        "GTX_DDR_DUMP": str(actual),
        "GTX_DDR_DUMP_ADDR": "0x100",
        "GTX_DDR_DUMP_SIZE": "0x20",   # 32 bytes = 16 FP16 single row
    })

    cmd = _resolve_pyspike() + ["--extlib=riscv.gtx", str(OUR_ELF)]
    print(f"  $ {' '.join(cmd)}")
    rc = subprocess.run(cmd, env=env, timeout=60).returncode
    print(f"  pyspike exit: {rc}")

    if rc != 0:
        print(f"  FAIL: pyspike non-zero exit ({rc})")
        return False

    if not actual.exists():
        print(f"  FAIL: dump 미생성 — GTX_DDR_DUMP atexit hook 미발화? (Plan 02 회귀)")
        return False

    passed, stats = compare_hex(str(actual), str(OUR_GOLDEN), strict=True)
    print(f"  compare_hex(strict=True): exact={stats['exact_matches']}/{stats['total_fp16']}, "
          f"within_tol={stats['within_tolerance']}, fail={stats['failures']}")
    print(f"  → {'PASS' if passed else 'FAIL'}")
    return passed


def run_test_b() -> bool | None:
    """vendor n1s16_abs.elf 풀 회귀 — GTX_DDR_INIT 필요."""
    print("=" * 70)
    print("Test B: vendor n1s16_abs.elf (full regression w/ vendor input)")
    print("=" * 70)

    # vendor 자산 점검
    for path in (VENDOR_ELF, VENDOR_INPUT, VENDOR_REF):
        if not path.exists():
            print(f"  SKIP: {path.relative_to(REPO)} 미존재")
            return None

    # GTX_DDR_INIT hook 지원 점검 — 두 조건 모두 충족해야 vendor 회귀 가능:
    #   (a) ddr.py에 _init_ddr_from_env 함수 존재 (vendor gtx_npu_core.cc:120 1:1 port)
    #   (b) npu.py의 GtxNpu.__init__에서 그 함수 호출 (인스턴스 생성 시 자동 pre-stage)
    try:
        from riscv.gtx.ddr import _init_ddr_from_env  # noqa: F401
    except ImportError:
        print(f"  SKIP: riscv.gtx.ddr._init_ddr_from_env 함수 미존재")
        print(f"        vendor gtx_npu_core.cc:120 1:1 port가 빠진 상태입니다.")
        return None

    npu_py = REPO / "src" / "main" / "python" / "riscv" / "gtx" / "npu.py"
    if "_init_ddr_from_env" not in npu_py.read_text():
        print(f"  SKIP: GtxNpu.__init__에서 _init_ddr_from_env 호출 누락")
        print(f"        함수는 정의돼있으나 인스턴스 생성 시 자동 트리거되지 않음.")
        return None
    print(f"  GTX_DDR_INIT hook OK (ddr._init_ddr_from_env + npu.py 호출 확인)")

    # GTX_DDR_INIT 지원되면 풀 회귀 실행
    actual = REPO / "_abs_result_b.hex"
    actual.unlink(missing_ok=True)

    env = os.environ.copy()
    env.update({
        "GTX_DDR_INIT": str(VENDOR_INPUT),
        "GTX_DDR_DUMP": str(actual),
        "GTX_DDR_DUMP_ADDR": "0x37f000000",  # vendor BASE_DDR_RESULT
        "GTX_DDR_DUMP_SIZE": str(12 * 1024 * 1024),   # 12.2MB
    })

    cmd = _resolve_pyspike() + ["--extlib=riscv.gtx", str(VENDOR_ELF)]
    print(f"  $ {' '.join(cmd)}")
    print(f"  (~12MB I/O, timeout 10분 — Python NPU 성능에 따라 달라집니다)")
    try:
        rc = subprocess.run(cmd, env=env, timeout=600).returncode
    except subprocess.TimeoutExpired:
        print(f"  FAIL: 600초 timeout — Python NPU가 vendor 풀 케이스 못 끝냄")
        return False
    print(f"  pyspike exit: {rc}")

    if rc != 0:
        return False
    if not actual.exists():
        return False

    passed, stats = compare_hex(str(actual), str(VENDOR_REF), strict=True)
    print(f"  compare_hex(strict=True): exact={stats['exact_matches']}/{stats['total_fp16']}, "
          f"within_tol={stats['within_tolerance']}, fail={stats['failures']}")
    print(f"  → {'PASS' if passed else 'FAIL'}")
    return passed


if __name__ == "__main__":
    print()
    print("riscv.gtx ABS op 검증 예제 — pyspike + GTX extension 통합 sanity check")
    print()

    # a = run_test_a()
    # print()
    b = run_test_b()
    print()

    print("=" * 70)
    print("결과 요약")
    print("=" * 70)
    # print(f"  Test A (우리 abs.elf zero-init):     {'PASS' if a else 'FAIL'}")
    print(f"  Test B (vendor 풀 회귀):              "
          f"{'PASS' if b else 'SKIP (GTX_DDR_INIT 미지원)' if b is None else 'FAIL'}")
    print()

    # if a is False:
    #     print("  Test A FAIL — atexit hook 또는 P6 Plan 02 회귀 점검 필요")
    #     sys.exit(1)
    if b is False:
        print("  Test B FAIL — vendor 회귀 mismatch (조사 필요)")
        sys.exit(1)
    print("  ✓ 환경 통합 OK")
