"""GtxNpu xp-port invariants (Phase 9 Plan 09-01b Task 2).

Pins the Wave 1b npu.py contract: GtxNpu allocates `_mxe_accum`,
`_credit_ld`, `_credit_st` as `xp.ndarray`s (numpy default, cupy under
GTX_USE_CUDA=1). RegisterFile instantiation no longer passes
`device=DEVICE`. Line 354's `.cpu()` chain is replaced with `to_host()`.

These tests are written RED first — they will fail against the
torch-backed npu.py and pass after the port.
"""
from __future__ import annotations

from pathlib import Path

from riscv.gtx.config_params import xp


def _read_npu_source() -> str:
    path = Path(__file__).resolve().parents[2] / "src/main/python/riscv/gtx/npu.py"
    return path.read_text()


def test_no_torch_in_npu_source():
    """npu.py is torch-free (H-5 audit)."""
    src = _read_npu_source()
    bad = [line for line in src.splitlines()
           if ("import torch" in line) or ("torch." in line and not line.lstrip().startswith("#"))]
    assert not bad, f"npu.py still has torch refs:\n{bad}"


def test_no_device_kwarg_in_npu_source():
    """npu.py source has zero `device=` kwarg refs (uses xp instead)."""
    src = _read_npu_source()
    bad = [line for line in src.splitlines()
           if "device=" in line and not line.lstrip().startswith("#")]
    assert not bad, f"npu.py still has device= refs:\n{bad}"


def test_no_DEVICE_symbol_in_npu_source():
    """npu.py does not reference DEVICE (deprecated alias, Wave 3 removal)."""
    src = _read_npu_source()
    # Allow DEVICE only inside comments/docstrings; flag code refs.
    bad = [line for line in src.splitlines()
           if "DEVICE" in line and not line.lstrip().startswith("#")]
    assert not bad, f"npu.py still references DEVICE symbol:\n{bad}"


def test_xp_import_present_in_npu_source():
    """npu.py imports xp / to_host from config_params."""
    src = _read_npu_source()
    assert "from .config_params import" in src and "xp" in src, (
        "npu.py must import `xp` from config_params"
    )
    assert "to_host" in src, "npu.py must import `to_host` (line 354 replacement)"


def test_xp_zeros_call_count_in_npu_source():
    """npu.py has ≥3 xp.zeros calls (_mxe_accum, _credit_ld, _credit_st)."""
    src = _read_npu_source()
    assert src.count("xp.zeros") >= 3, (
        f"expected ≥3 `xp.zeros` calls in npu.py; found {src.count('xp.zeros')}"
    )


def test_to_host_call_present_in_npu_source():
    """npu.py uses to_host(...) somewhere (line 354 area)."""
    src = _read_npu_source()
    assert "to_host(" in src, "npu.py must call to_host(...) at the L2 DDR-flush bridge"


def test_npu_construct_state_arrays_are_xp():
    """GtxNpu constructor produces xp.ndarray state arrays."""
    from riscv.gtx.npu import GtxNpu
    npu = GtxNpu()
    # State arrays should be xp.ndarray, not torch.Tensor.
    assert type(npu._mxe_accum).__module__.startswith(("numpy", "cupy")), (
        f"_mxe_accum is not xp: type={type(npu._mxe_accum)}"
    )
    assert type(npu._credit_ld).__module__.startswith(("numpy", "cupy"))
    assert type(npu._credit_st).__module__.startswith(("numpy", "cupy"))


def test_npu_construct_dtypes():
    """State-array dtypes preserved: float32 / int32 / int32."""
    from riscv.gtx.npu import GtxNpu
    npu = GtxNpu()
    assert npu._mxe_accum.dtype == xp.float32
    assert npu._credit_ld.dtype == xp.int32
    assert npu._credit_st.dtype == xp.int32


def test_npu_construct_shapes():
    """State-array shapes preserved: (NEST, SPU) = (4, 16)."""
    from riscv.gtx.npu import GtxNpu
    from riscv.gtx.config_params import GTX_NEST_NUM, GTX_SPU_NUM
    npu = GtxNpu()
    expected = (GTX_NEST_NUM, GTX_SPU_NUM)
    assert tuple(npu._mxe_accum.shape) == expected
    assert tuple(npu._credit_ld.shape) == expected
    assert tuple(npu._credit_st.shape) == expected


def test_npu_register_file_storage_is_xp():
    """GSPR/NSPR/LSPR RegisterFile storage is xp.ndarray (post-Task-1 + Task-2)."""
    from riscv.gtx.npu import GtxNpu
    npu = GtxNpu()
    assert type(npu.gspr.tensor).__module__.startswith(("numpy", "cupy"))
    assert type(npu.nspr.tensor).__module__.startswith(("numpy", "cupy"))
    assert type(npu.lspr.tensor).__module__.startswith(("numpy", "cupy"))
    assert npu.gspr.tensor.dtype == xp.int64
    assert npu.nspr.tensor.dtype == xp.int64
    assert npu.lspr.tensor.dtype == xp.int64


def test_npu_reset_zeros_state_arrays():
    """reset() must zero state arrays via xp-uniform in-place op (no .fill_())."""
    from riscv.gtx.npu import GtxNpu
    npu = GtxNpu()
    # Seed with non-zero values
    npu._mxe_accum[...] = 3.14
    npu._credit_ld[...] = 7
    npu._credit_st[...] = 11
    # Reset must zero them. Pass a dummy "proc" (None) — reset only touches
    # state arrays + RegisterFile defaults, not proc state directly here
    # (XPR write is wrapped in try/except, swallowed if proc is None).
    try:
        npu.reset(None)  # type: ignore[arg-type]
    except AttributeError:
        # If proc.state.XPR.write fails, the state-array reset still ran.
        pass
    assert bool(xp.all(npu._mxe_accum == 0))
    assert bool(xp.all(npu._credit_ld == 0))
    assert bool(xp.all(npu._credit_st == 0))
