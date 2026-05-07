# Build the nop_wjoin.elf test fixture (D-22).
#
# Requires:  /opt/riscv/bin/riscv64-unknown-elf-gcc (15.2.0+ confirmed in env)
# Usage:     make
#
# The .elf is committed to git (D-22) so unit tests do not require the
# cross-toolchain at run time. Rebuild only when nop_wjoin.S changes.

# Override Make's implicit CC=cc default. To use a different toolchain, run
#   make CC=/path/to/riscv64-unknown-elf-gcc
CC       = /opt/riscv/bin/riscv64-unknown-elf-gcc
# -Ttext alone places .text inside a LOAD segment that GCC 15.2 pads
# backward by 0x1000 to fit .riscv.attributes, ending up at 0x7ffff000
# which is below spike's DRAM (0x80000000). -Wl,-Ttext-segment forces
# the LOAD VirtAddr to start at 0x80000000.
LDFLAGS  = -nostdlib -nostartfiles -static -Wl,-Ttext-segment=0x80000000

nop_wjoin.elf: nop_wjoin.S
	$(CC) $(LDFLAGS) -o $@ $<

# P4 mm_basic firmware (D-09 fallback). Pre-built .elf committed (P2 D-22 pattern).
mm_basic.elf: mm_basic.S
	$(CC) $(LDFLAGS) -o $@ $<

# P5 activation_relu_gelu firmware (D-04). Pre-built .elf committed.
activation_relu_gelu.elf: activation_relu_gelu.S
	$(CC) $(LDFLAGS) -o $@ $<

.PHONY: clean
clean:
	rm -f nop_wjoin.elf mm_basic.elf activation_relu_gelu.elf

.PHONY: verify
verify: nop_wjoin.elf
	@/opt/riscv/bin/riscv64-unknown-elf-readelf -h $< | grep -E 'Entry point|Type'
	@echo OK

.PHONY: verify-mm
verify-mm: mm_basic.elf
	@/opt/riscv/bin/riscv64-unknown-elf-readelf -h $< | grep -E 'Entry point|Type'
	@echo OK

.PHONY: verify-act
verify-act: activation_relu_gelu.elf
	@/opt/riscv/bin/riscv64-unknown-elf-readelf -h $< | grep -E 'Entry point|Type'
	@echo OK
