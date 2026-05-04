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
LDFLAGS  = -nostdlib -nostartfiles -static -Ttext=0x80000000

nop_wjoin.elf: nop_wjoin.S
	$(CC) $(LDFLAGS) -o $@ $<

.PHONY: clean
clean:
	rm -f nop_wjoin.elf

.PHONY: verify
verify: nop_wjoin.elf
	@/opt/riscv/bin/riscv64-unknown-elf-readelf -h $< | grep -E 'Entry point|Type'
	@echo OK
