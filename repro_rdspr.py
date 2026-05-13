import torch
from riscv.gtx.npu import GtxNpu
from riscv.gtx._registry import handler
from riscv.gtx.unit.ins.encoding import CUSTOM0, GTX_ISS_F7_OPSET
from riscv.gtx.unit.ins.disasm import Instruction

# =============================================================================
# 1. Dummy 명령어 등록 (Registration)
# =============================================================================

# funct7=0x66: rs1 + rs2 + 7 -> rd
@handler(kind='custom0', funct7=0x66, mnemonic='gtx.dummy')
def dummy_handler(npu, proc, insn, xs1, xs2):
    """Simple dummy instruction that adds registers."""
    val1 = proc.state.XPR[insn.rs1]
    val2 = proc.state.XPR[insn.rs2]
    res = val1 + val2 + 7
    print(f">>> [HANDLER] gtx.dummy: {val1} + {val2} + 7 = {res}")
    if insn.rd != 0:
        proc.state.XPR.write(insn.rd, res)
    return res

# funct7=0x67, funct3=3: Sub-command example
# To get funct3=3: xd=0 (bit 2), xs1=1 (bit 1), xs2=1 (bit 0)
@handler(kind='custom0', funct7=0x67, funct3=3, mnemonic='gtx.dummy.sub', mask_funct3=True)
def dummy_sub_handler(npu, proc, insn, xs1, xs2):
    """Dummy sub-command that stages a value to GSPR."""
    val = proc.state.XPR[insn.rs1]
    print(f">>> [HANDLER] gtx.dummy.sub: Staging {val:#x} to GSPR[0x003]")
    # RegisterFile broadcasting / direct access
    npu.gspr[0x003] = val
    return 0

# =============================================================================
# 2. Standalone Mock Infrastructure
# =============================================================================

class FakeXPR:
    def __init__(self):
        self.regs = [0] * 32
    def __getitem__(self, idx):
        return self.regs[idx]
    def __setitem__(self, idx, val):
        self.regs[idx] = val
    def write(self, idx, val):
        self.regs[idx] = val

class FakeState:
    def __init__(self):
        self.XPR = FakeXPR()

class FakeProc:
    def __init__(self):
        self.state = FakeState()

class SimInsn:
    """Spike-compatible instruction object for NPU simulation.
    Uses Instruction class from disasm.py to decode raw bits.
    """
    def __init__(self, raw_bits: int):
        # Note: disasm.Instruction has a minor bug in .fn7 property 
        # (precedence of & vs >>). We manually extract here for reliability.
        self.funct = (raw_bits & 0xFE000000) >> 25
        self.rs2 = (raw_bits & 0x1F00000) >> 20
        self.rs1 = (raw_bits & 0xF8000) >> 15
        self.rd = (raw_bits & 0xF80) >> 7
        self.opcode = raw_bits & 0x7F
        
        # RoCC flags encoded in funct3 (bits 14:12)
        f3 = (raw_bits >> 12) & 0x7
        self.xd = (f3 >> 2) & 1
        self.xs1 = (f3 >> 1) & 1
        self.xs2 = f3 & 1

# =============================================================================
# 3. Execution Demo
# =============================================================================

def run_demo():
    print("=== GtxNpu Instruction Demo ===")
    
    # NPU instance (collects registered handlers)
    npu = GtxNpu()
    proc = FakeProc()

    # -------------------------------------------------------------------------
    # Verification 1: Disassembly
    # -------------------------------------------------------------------------
    print("\n[1] Checking registered disassembly entries:")
    disasms = npu.get_disasms(proc)
    for d in disasms:
        if "gtx.dummy" in d.name:
            print(f"  Name: {d.name:<15} | Match: {d.match:#010x} | Mask: {d.mask:#010x}")

    # -------------------------------------------------------------------------
    # Verification 2: gtx.dummy (funct7=0x66) execution
    # -------------------------------------------------------------------------
    # Instruction: gtx.dummy rs1=10, rs2=11, rd=12
    # Bits: [f7=0x66 | rs2=11 | rs1=10 | f3=7 (xd/s1/s2) | rd=12 | op=0x0b]
    raw_dummy = (0x66 << 25) | (11 << 20) | (10 << 15) | (7 << 12) | (12 << 7) | 0x0b
    insn_dummy = SimInsn(raw_dummy)
    
    proc.state.XPR[10] = 100
    proc.state.XPR[11] = 200
    print(f"\n[2] Executing gtx.dummy: XPR[10]=100, XPR[11]=200")
    npu.custom0(proc, insn_dummy, 0, 0)
    
    res = proc.state.XPR[12]
    print(f"    Result: XPR[12] = {res} (Expected 100+200+7 = 307)")
    assert res == 307

    # -------------------------------------------------------------------------
    # Verification 3: gtx.dummy.sub (funct7=0x67, funct3=3) execution
    # -------------------------------------------------------------------------
    # Instruction: gtx.dummy.sub rs1=15
    # Bits: [f7=0x67 | rs2=0 | rs1=15 | f3=3 | rd=0 | op=0x0b]
    raw_sub = (0x67 << 25) | (0 << 20) | (15 << 15) | (3 << 12) | (0 << 7) | 0x0b
    insn_sub = SimInsn(raw_sub)
    
    proc.state.XPR[15] = 0xDEADBEEF
    print(f"\n[3] Executing gtx.dummy.sub: XPR[15]=0xDEADBEEF")
    npu.custom0(proc, insn_sub, 0, 0)
    
    gspr_val = int(npu.gspr[0x003])
    print(f"    Result: GSPR[0x003] = {gspr_val:#x} (Expected 0xDEADBEEF)")
    assert gspr_val == 0xDEADBEEF

if __name__ == "__main__":
    try:
        run_demo()
        print("\nDEMO COMPLETED SUCCESSFULLY")
    except Exception as e:
        print(f"\nDEMO FAILED: {e}")
        import traceback
        traceback.print_exc()
