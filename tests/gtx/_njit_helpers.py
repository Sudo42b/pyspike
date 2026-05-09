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
"""P7 NJIT-05 helper: registry of 28 stateless kernels for per-kernel ULP-0 parity test.

Plan 02/03/04 will GREEN-fill the `_impl` (FP32-only) and `_njit` (compiled)
aliases in gemm_core/vec_core/act_core. This helper exposes them via name-key
so test_njit_parity.py can iterate uniformly.

LAZY IMPORT discipline: This module does NOT import gemm_core/vec_core/act_core
at module-level -- those imports happen inside `get_public_fn(name)` and
`get_impl_fn(name)`. Reason: in Wave 0 the `_impl` aliases don't yet exist;
parametrize collection must succeed regardless.
"""
from __future__ import annotations
from typing import Any, Callable

import numpy as np

# 28-kernel registry. Tuple = (kernel_name, module_dotted_path, public_fn, impl_fn).
# `public_fn` = current public API (FP16 in/out) -- numpy oracle.
# `impl_fn`   = FP32-only `_impl` Plan 02/03/04 will create -- @njit-decorated.
ALL_NJIT_KERNELS: list[tuple[str, str, str, str]] = [
    # gemm_core (3) -- Plan 02
    ("gemm_core",          "riscv.gtx.gemm_core", "gemm_core",          "_gemm_core_impl"),
    ("gemm_reduce_sum_a",  "riscv.gtx.gemm_core", "gemm_reduce_sum_a",  "_gemm_reduce_sum_a_impl"),
    ("gemm_dot",           "riscv.gtx.gemm_core", "gemm_dot",           "_gemm_dot_impl"),
    # vec_core (7) -- Plan 03
    ("sasmd_kernel",       "riscv.gtx.vec_core",  "sasmd_kernel",       "_sasmd_impl"),
    ("dot_kernel",         "riscv.gtx.vec_core",  "dot_kernel",         "_dot_impl"),
    ("vsum_kernel",        "riscv.gtx.vec_core",  "vsum_kernel",        "_vsum_impl"),
    ("clamp_min_kernel",   "riscv.gtx.vec_core",  "clamp_min_kernel",   "_clamp_min_impl"),
    ("clamp_max_kernel",   "riscv.gtx.vec_core",  "clamp_max_kernel",   "_clamp_max_impl"),
    ("accum_kernel",       "riscv.gtx.vec_core",  "accum_kernel",       "_accum_impl"),
    ("arange_kernel",      "riscv.gtx.vec_core",  "arange_kernel",      "_arange_impl"),
    # act_core activations (7) -- Plan 04 (5 transcendental need objmode escape per D-09)
    ("relu",               "riscv.gtx.act_core",  "relu",               "_relu_impl"),
    ("prelu",              "riscv.gtx.act_core",  "prelu",              "_prelu_impl"),
    ("gelu",               "riscv.gtx.act_core",  "gelu",               "_gelu_impl"),         # objmode
    ("tanh_act",           "riscv.gtx.act_core",  "tanh_act",           "_tanh_act_impl"),     # objmode
    ("sigmoid",            "riscv.gtx.act_core",  "sigmoid",            "_sigmoid_impl"),      # objmode
    ("softmax",            "riscv.gtx.act_core",  "softmax",            "_softmax_impl"),      # objmode
    ("esum",               "riscv.gtx.act_core",  "esum",               "_esum_impl"),         # objmode
    # act_core pool (2) -- Plan 04
    ("pool_max",           "riscv.gtx.act_core",  "pool_max",           "_pool_max_impl"),
    ("pool_avg",           "riscv.gtx.act_core",  "pool_avg",           "_pool_avg_impl"),
    # act_core cvt (9) -- Plan 04
    ("cvt_qh",             "riscv.gtx.act_core",  "cvt_qh",             "_cvt_qh_impl"),
    ("cvt_hq",             "riscv.gtx.act_core",  "cvt_hq",             "_cvt_hq_impl"),
    ("cvt_ih",             "riscv.gtx.act_core",  "cvt_ih",             "_cvt_ih_impl"),
    ("cvt_hi",             "riscv.gtx.act_core",  "cvt_hi",             "_cvt_hi_impl"),
    ("cvt_hn",             "riscv.gtx.act_core",  "cvt_hn",             "_cvt_hn_impl"),
    ("cvt_sh",             "riscv.gtx.act_core",  "cvt_sh",             "_cvt_sh_impl"),
    ("cvt_hs",             "riscv.gtx.act_core",  "cvt_hs",             "_cvt_hs_impl"),
    ("cvt_dh",             "riscv.gtx.act_core",  "cvt_dh",             "_cvt_dh_impl"),
    ("cvt_hd",             "riscv.gtx.act_core",  "cvt_hd",             "_cvt_hd_impl"),
]

# Convenience constants for parametrize @ collection time
ALL_NJIT_KERNEL_NAMES: list[str] = [k[0] for k in ALL_NJIT_KERNELS]
TRANSCENDENTAL_KERNELS: set[str] = {"gelu", "tanh_act", "sigmoid", "softmax", "esum"}

assert len(ALL_NJIT_KERNELS) == 28, (
    f"P7 D-06: expected 28 kernels (gemm 3 + vec 7 + act 18), "
    f"got {len(ALL_NJIT_KERNELS)}"
)


def get_public_fn(kernel_name: str) -> Callable[..., Any]:
    """Lazy import: returns FP16 in/out public API (numpy oracle path)."""
    for name, mod_path, pub_fn, _impl_fn in ALL_NJIT_KERNELS:
        if name == kernel_name:
            import importlib
            mod = importlib.import_module(mod_path)
            return getattr(mod, pub_fn)
    raise KeyError(f"unknown kernel: {kernel_name}")


def get_impl_fn(kernel_name: str) -> Callable[..., Any]:
    """Lazy import: returns FP32-only `_impl` (njit-wrapped). Plan 02/03/04 GREEN-fills."""
    for name, mod_path, _pub_fn, impl_fn in ALL_NJIT_KERNELS:
        if name == kernel_name:
            import importlib
            mod = importlib.import_module(mod_path)
            return getattr(mod, impl_fn)
    raise KeyError(f"unknown kernel: {kernel_name}")


def generate_test_inputs(kernel_name: str, *, seed: int = 42) -> tuple:
    """Generate fixed-seed FP16 inputs matching kernel signature.

    Plan 02/03/04 may extend this for kernels with non-trivial signatures
    (e.g. `arange_kernel(n, start, step)` takes 3 scalars). Wave 0 returns
    a sentinel placeholder so test_njit_parity.py collection succeeds.
    """
    rng = np.random.default_rng((seed + hash(kernel_name)) & 0xFFFFFFFF)
    # Wave 0 placeholder: 1-D FP16 length-16 array. Plan 02/03/04 will
    # specialize per kernel signature.
    return (rng.random(16, dtype=np.float32).astype(np.float16),)
