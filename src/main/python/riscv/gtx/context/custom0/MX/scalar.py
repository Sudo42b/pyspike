"""Scalar (_VS/_IS) arithmetic — MERGED INTO vector.py.

Every funct7=0x10 (SASMD), 0x11 (FMADD_S), and 0x13 (MINMAX_S) handler that
previously lived here is now registered exactly once in
``custom0/MX/vector.py`` (the single comprehensive owner of all VEC/SCALAR
funct7 ops). The handler registry keys solely on ``(funct7, funct3)``, so
keeping a second module that registered the same keys would silently overwrite
the vector.py entries. This module is intentionally left as a stub — do NOT
import it for handler registration.
"""
from __future__ import annotations

import torch

from ...inst_handler import inst_register

from ....config_params import L0_SIZE_BYTES, NEST_NUM, SPU_NUM
from ....csr import GSPR, LSPR
from ... import _resolve_nest_spu

@inst_register.custom0(name='add.vs', funct7=0b0010000, funct3=0b000)
def add_vs(npu, proc, inst, cxt) -> int:
    return vec_op(npu, proc, inst)


@inst_register.custom0(name='sub.vs', funct7=0b0010000, funct3=0b001)
def sub_vs(npu, proc, inst, cxt) -> int:
    return vec_op(npu, proc, inst)


@inst_register.custom0(name='mul.vs', funct7=0b0010000, funct3=0b010)
def mul_vs(npu, proc, inst, cxt) -> int:
    return vec_op(npu, proc, inst)


@inst_register.custom0(name='div.vs', funct7=0b0010000, funct3=0b011)
def div_vs(npu, proc, inst, cxt) -> int:
    return vec_op(npu, proc, inst)


@inst_register.custom0(name='add.is', funct7=0b0010000, funct3=0b100)
def add_is(npu, proc, inst, cxt) -> int:
    return vec_op(npu, proc, inst)


@inst_register.custom0(name='sub.is', funct7=0b0010000, funct3=0b101)
def sub_is(npu, proc, inst, cxt) -> int:
    return vec_op(npu, proc, inst)


@inst_register.custom0(name='mul.is', funct7=0b0010000, funct3=0b110)
def mul_is(npu, proc, inst, cxt) -> int:
    return vec_op(npu, proc, inst)


@inst_register.custom0(name='div.is', funct7=0b0010000, funct3=0b111)
def div_is(npu, proc, inst, cxt) -> int:
    return vec_op(npu, proc, inst)


@inst_register.custom0(name='fmadd.iss', funct7=0b0010001, funct3=0b100)
def fmadd_iss(npu, proc, inst, cxt) -> int:
    return vec_op(npu, proc, inst)


@inst_register.custom0(name='max.is', funct7=0b0010011, funct3=0b100)
def max_is(npu, proc, inst, cxt) -> int:
    return vec_op(npu, proc, inst)


@inst_register.custom0(name='min.is', funct7=0b0010011, funct3=0b101)
def min_is(npu, proc, inst, cxt) -> int:
    return vec_op(npu, proc, inst)

