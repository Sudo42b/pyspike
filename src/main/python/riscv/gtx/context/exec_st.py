from enum import Enum, auto

class CXT(Enum):
    """GTX context state definitions & Valid Instructions."""
    C1 = auto()  # Plan outside (before START_P)
    C2 = auto()  # Shared inside (inside START_P + START_S)
    C3 = auto()  # Thread inside (inside START_P + START_T)
    C4 = auto()  # Plan inside only (inside START_P, outside S/T)

    @property
    def valid_instructions(self) -> set:
        """자신의 컨텍스트에서 실행 가능한 명령어 셋을 O(1)로 반환합니다."""
        return _VALID_INSTRUCTIONS_MAP[self]

"""GTX context state definitions.

NPU context FSM — C1/C2/C3/C4 contexts, transitions, validity.

NPU contexts (persistent across instructions):
  C1: plan outside     — before START_P (initial / after END_P)
  C2: shared inside    — inside START_P + START_S
  C3: thread inside    — inside START_P + START_T
  C4: plan inside only — inside START_P, outside S/T

States:
	C1 (PLAN_OUTSIDE):
		Plan outside state. This is the state before ``START_P`` and the state
		restored after ``END_P``.

	C4 (PLAN_INSIDE):
		Plan-inside state with both shared and thread scopes inactive. This is
		the state inside ``START_P`` and outside ``START_S`` and ``START_T``.

	C2 (SHARED):
		Plan-inside and shared-inside state. This is the state reached inside
		``START_P`` and ``START_S``.

	C3 (THREAD):
		Plan-inside and thread-inside state. This is the state reached inside
		``START_P`` and ``START_T``.

Public API:
  NpuContext                  — Enum (C1/C2/C3/C4)
  INITIAL_CONTEXT             — NpuContext.C1
  get_group(mnemonic)         — group name or None
  is_valid_in_context(mn, ctx)— bool
  is_warp_marker(mnemonic)    — bool
  apply_transition(ctx, mn)   — next context (or unchanged)
  is_legal_transition(ctx, mn)— strict check

"""
_VALID_INSTRUCTIONS_MAP = {    
    CXT.C1: {
        "mm.s", "mm.o", "mm.v", "mm", "mm.t",
        "mmc.s", "mmc.o", "mmc.v", "mmc", "mmc.t",
        "im2col.n", "im2col.d",
        "add.vs", "sub.vs", "mul.vs", "div.vs", "fmadd.vss",
        "max.vs", "min.vs",
        "add.is", "sub.is", "mul.is", "div.is", "fmadd.iss",
        "max.is", "min.is",
        "add.vv", "sub.vv", "mul.vv", "div.vv", "fmadd.vvv",
        "sqrt.v", "exp.v", "ln.v", "abs.v", "neg.v",
        "sign.v", "step.v", "ceil.v", "trunc.v", "floor.v", "rne.v",
        "clamp.min", "clamp.max",
        "accum", "arange",
        "add.ii", "sub.ii", "mul.ii", "div.ii", "fmadd.iii",
        "sqrt.i", "exp.i", "ln.i", "abs.i", "neg.i",
        "sign.i", "step.i", "ceil.i", "trunc.i", "floor.i", "rne.i",
        "and.ii", "or.ii", "not.i", "shift.i",
        "scvt.qh", "scvt.hq", "scvt.ih", "scvt.hi", "scvt.hn",
        "fcvt.sh", "fcvt.hs", "fcvt.dh", "fcvt.hd",
        "prelu", "gelu", "tanh", "sigm",
        "prelu.i", "gelu.i", "tanh.i", "sigm.i",
        "esum", "softmax", "esum.i", "softmax.i",
        "pool.m", "pool.a",
        "tpose", "fill",
        "load.svr", "store.svr",
        "mcast.s2l", "mcast.g2s", "mcast.s2s",
        "copy.mem",
        "rdspr", "wrspr",
        "opset", "cpsvr", "mvsvr",
        "credit.ld",
        "mexec", "mbar", "eom",
        "bar", "wait", "intr",
        "flush", "halt"
    },
    CXT.C2 : {
        "tpose", "fill", 
        "load", "store", "copy",
        "mcast.s2l",
        "opset", "credit.ld", "credit.st", "credit.chk",
        "bar", "wait"
    },

    CXT.C3 : {
        "mm.s", "mm.o", "mm.v", "mm", "mm.t",
        "mmc.s", "mmc.o", "mmc.v", "mmc", "mmc.t",
        "im2col.n", "im2col.d",
        "add.vs", "sub.vs", "mul.vs", "div.vs", "fmadd.vss",
        "max.vs", "min.vs",
        "add.is", "sub.is", "mul.is", "div.is", "fmadd.iss",
        "max.is", "min.is",
        "add.vv", "sub.vv", "mul.vv", "div.vv", "fmadd.vvv",
        "sqrt.v", "exp.v", "ln.v", "abs.v", "neg.v",
        "sign.v", "step.v", "ceil.v", "trunc.v", "floor.v", "rne.v",
        "clamp.min", "clamp.max",
        "accum", "arange",
        "add.ii", "sub.ii", "mul.ii", "div.ii", "fmadd.iii",
        "sqrt.i", "exp.i", "ln.i", "abs.i", "neg.i",
        "sign.i", "step.i", "ceil.i", "trunc.i", "floor.i", "rne.i",
        "and.ii", "or.ii", "not.i", "shift.i",
        "scvt.qh", "scvt.hq", "scvt.ih", "scvt.hi", "scvt.hn",
        "prelu", "gelu", "tanh", "sigm",
        "prelu.i", "gelu.i", "tanh.i", "sigm.i",
        "esum", "softmax", "esum.i", "softmax.i",
        "pool.m", "pool.a",
        "load", "store", "copy",
        "load.svr", "store.svr",
        "wrspr", "opset", "cpsvr", "mvsvr",
        "credit.ld", "credit.st", "credit.chk",
        "bar", "wait"
    },
    CXT.C4 : {
        "mm.s", "mm.o", "mm.v", "mm", "mm.t",
        "mmc.s", "mmc.o", "mmc.v", "mmc", "mmc.t",
        "im2col.n", "im2col.d",
        "add.vs", "sub.vs", "mul.vs", "div.vs", "fmadd.vss",
        "max.vs", "min.vs",
        "add.is", "sub.is", "mul.is", "div.is", "fmadd.iss",
        "max.is", "min.is",
        "add.vv", "sub.vv", "mul.vv", "div.vv", "fmadd.vvv",
        "sqrt.v", "exp.v", "ln.v", "abs.v", "neg.v",
        "sign.v", "step.v", "ceil.v", "trunc.v", "floor.v", "rne.v",
        "clamp.min", "clamp.max",
        "accum", "arange",
        "add.ii", "sub.ii", "mul.ii", "div.ii", "fmadd.iii",
        "sqrt.i", "exp.i", "ln.i", "abs.i", "neg.i",
        "sign.i", "step.i", "ceil.i", "trunc.i", "floor.i", "rne.i",
        "and.ii", "or.ii", "not.i", "shift.i",
        "scvt.qh", "scvt.hq", "scvt.ih", "scvt.hi", "scvt.hn",
        "prelu", "gelu", "tanh", "sigm",
        "prelu.i", "gelu.i", "tanh.i", "sigm.i",
        "esum", "softmax", "esum.i", "softmax.i",
        "pool.m", "pool.a",
        "tpose", "fill",
        "load.svr", "store.svr",
        "mcast.s2l", "wrspr", "opset", "cpsvr", "mvsvr",
        "bar", "wait"
    }
}
