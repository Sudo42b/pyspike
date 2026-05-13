import importlib
import sys
import types
import warnings

from ._utils import load_spike_library

try:
    from ._version import __version__
except ImportError:
    warnings.warn("Missing `riscv._version`, run `python -m setuptools_scm --force-write-version-files` to generate.")

__all__ = ["ENV_PYSPIKE_LIBS", "ENV_PYSPIKE_EXTS"]

ENV_PYSPIKE_LIBS = "PYSPIKE_LIBS"

ENV_PYSPIKE_EXTS = "PYSPIKE_EXTS"

# load spike runtime library (libriscv.so, libcustomext.so)
try:
    load_spike_library("riscv")
except RuntimeError:
    warnings.warn("Missing `libriscv.so`, run `python install -e '.[dev]' to build it.")
else:
    try:
        # import pyspike module (_riscv)
        _riscv: types.ModuleType = importlib.import_module('._riscv', __package__)
        self: types.ModuleType = sys.modules[__package__]
        # mount submodules _riscv.* to riscv.*
        for name in dir(_riscv):
            _attr = getattr(_riscv, name)
            if isinstance(_attr, types.ModuleType):
                sys.modules[f'{__package__}.{name}'] = _attr
                setattr(self, name, _attr)
        # bootstrap spike-in-python
        getattr(_riscv, "bootstrap")()
    except (ImportError, AttributeError) as exc:
        warnings.warn("Missing `riscv._riscv`, run `python setup.py build_ext --inplace` to build it.")
