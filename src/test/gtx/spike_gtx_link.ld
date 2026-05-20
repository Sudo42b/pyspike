/*
 * spike_gtx_link.ld — Modified gtx_link.ld for Spike simulator
 * ORIGIN = 0x80000000 (Spike's default memory start)
 */

OUTPUT_ARCH( "riscv" )
ENTRY(_start)

MEMORY {
  RAM (rwx) : ORIGIN = 0x80000000, LENGTH = 0x40000000 /* 1GB RAM */
}

STACK_SIZE = 1024*64; /* 64KB Stack */

CL_SIZE = 32;

SECTIONS {

  /* code segment */
  .text.init ORIGIN(RAM) : {
    FILL(0);
    . = ORIGIN(RAM) + 0x100 - 12;
    SIM_EXIT = .;
    LONG(0x13);
    SIM_STOP = .;
    LONG(0x6F);
    LONG(-1);
    . = ORIGIN(RAM) + 0x100;
    PROVIDE(__TEXT_START__ = .);
    *(.text.init)
  } >RAM

  .text  : {
    *crt.o(.text .text.*)
    *(.text .text.*)
    *(sc_test_section)
    . = ALIGN(CL_SIZE);
     PROVIDE(__TEXT_END__ = .);
  } >RAM

  .cmd_line : ALIGN(CL_SIZE)
  {
    PROVIDE(_cmd_line = .);
    . += (1024);
    PROVIDE(_cmd_line_end = . );
  }>RAM
  .heap : ALIGN(CL_SIZE)
  {
    PROVIDE(_heap_start = .);
    PROVIDE(__malloc_start = .);
    *(.heap);
    . += (1024*64);
  } >RAM
  /* data segment */
  .data : {
    *(.data .data.*)
    . = ALIGN(CL_SIZE);
  } >RAM

  .sdata : {
    _gp = . + 0x800;
    *(.srodata.cst16) *(.srodata.cst8) *(.srodata.cst4) *(.srodata.cst2) *(.srodata*)
    *(.sdata .sdata.* .gnu.linkonce.s.*)
    . = ALIGN(CL_SIZE);
  } >RAM

  /* thread-local data segment */
  .tdata : {
    PROVIDE(_tls_data = .);
    PROVIDE(_tdata_begin = .);
    *(.tdata .tdata.*)
    PROVIDE(_tdata_end = .);
    . = ALIGN(CL_SIZE);
  } >RAM

  .tbss : {
    *(.tbss .tbss.*)
    . = ALIGN(CL_SIZE);
    PROVIDE(_tbss_end = .);
  } >RAM

  /* bss segment */
  .sbss : {
    PROVIDE(__BSS_START__ = .);
    *(.sbss .sbss.* .gnu.linkonce.sb.*)
    *(.scommon)
  } >RAM

  .bss : {
    *(.bss .bss.*)
    . = ALIGN(CL_SIZE);
    PROVIDE(__BSS_END__ = .);
  } >RAM

  _end = .;
  PROVIDE(__end = .);

  .stack : {
    FILL(0);
    PROVIDE(__STACK_START__ = .);
    . += STACK_SIZE;
    PROVIDE(__C_STACK_TOP__ = .);
    PROVIDE(__STACK_END__ = .);
  } >RAM

  /* HTIF symbols for Spike communication */
  .tohost ALIGN(0x1000) : {
    tohost = .;
    . += 8;
    fromhost = .;
    . += 8;
  } >RAM

  /DISCARD/ : {
    *(.eh_frame .eh_frame.*)
  }
}
