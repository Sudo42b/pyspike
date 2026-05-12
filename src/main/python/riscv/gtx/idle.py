#
# Copyright 2026 WuXi EsionTech Co., Ltd.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#    http://www.apache.org/licenses/LICENSE-2.0
#
"""IDLE state — terminal / initial pipeline state.

The FSM driver (`GtxNpu._run_pipeline`) exits its `while state ≠ IDLE`
loop when the NPU returns to IDLE; therefore IDLE has no transition
function. This module exists only for symmetry with the other state
files and to give debuggers a stable import target.
"""
from __future__ import annotations


def state_idle(npu) -> None:
    """No-op. IDLE is the loop-exit sentinel — never invoked inside _step.

    Raises if called; reaching this means the FSM driver lost its guard.
    """
    raise RuntimeError(
        "GtxNpu FSM: IDLE has no transition function; "
        "the _run_pipeline loop should have exited"
    )
