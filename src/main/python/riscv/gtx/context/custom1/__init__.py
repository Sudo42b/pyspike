CUSTOM1 = 0x2b # custom-1 0b101011

# ============================================================================
# custom1() — Warp control dispatch (custom-1 opcode 0x2b)
#
# Encoding: funct3 (bits[14:12]) selects the warp control variant:
#   funct3=000: START_T   funct3=001: END_T
#   funct3=010: START_S   funct3=011: END_S
#   funct3=100: SPLIT     funct3=101: JOIN
#   funct3=110: START_P   funct3=111: END_P
#
# In RoCC, bits[14:12] are {xd,xs1,xs2} flags, not funct3.
# We reconstruct funct3 from these bits and read registers directly.
# ============================================================================
