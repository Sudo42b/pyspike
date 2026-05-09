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
"""Lazy numba shim. P7 D-02 / NJIT-01 single source-of-truth.

Allows P7 hot kernels to write `@njit(cache=True)` uniformly. When numba
is not installed (base `pip install spike`), `njit` becomes a no-op
decorator that returns the wrapped function unchanged -- kernels run as
pure NumPy (P4/P5 lineage).

Per CONTEXT D-02 / NJIT-01: `pip install spike` (base) -> NumPy-only;
`pip install spike[fast]` -> numba acceleration. Both paths must produce
bit-exact strict-mode regression PASS.
"""
from __future__ import annotations
from typing import Any, Callable, TypeVar

F = TypeVar('F', bound=Callable[..., Any])

try:
    from numba import njit as _real_njit  # type: ignore[import-not-found]
    HAS_NUMBA: bool = True

    def njit(*args: Any, **kwargs: Any) -> Any:
        """Real numba.njit re-export."""
        return _real_njit(*args, **kwargs)

except ImportError:  # pragma: no cover -- exercised when `spike[fast]` not installed
    HAS_NUMBA = False

    def njit(*args: Any, **kwargs: Any) -> Any:
        """No-op passthrough. Two call patterns:

        @njit                  -> args=(fn,), kwargs={}            -> return fn
        @njit(cache=True)      -> args=(), kwargs={'cache': True}  -> return decorator
        """
        if len(args) == 1 and callable(args[0]) and not kwargs:
            return args[0]

        def decorator(fn: F) -> F:
            return fn
        return decorator


__all__ = ["njit", "HAS_NUMBA"]
