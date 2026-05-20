"""
bar	4'b1111	3'b000	rsvd	rsvd	3'b000	gpr	gtx op	yes	no	spu/nest/nsu	5	N/A	N/A	N/A	N/A	N/A	status[0]	N/A	wait unitl all SPU are idle, returns 1 if operation was successful
wait	4'b1111	3'b001	gpr	gpr	3'b000	gpr	gtx op	yes	no	spu/nest/nsu	5	N/A	wait_clk_count[31:0], byte[63:32]	start_address[36:0]	N/A	N/A	status[0]	N/A	wait until clock counter expired, returns 1 if operation was successful
intr	4'b1111	3'b011	rsvd	gpr	3'b000	rsvd	gtx op	yes	no	spu/nest/nsu	5	N/A	intr_src[63:0]	N/A	N/A	N/A	N/A	N/A	interrunpt
flush	4'b1111	3'b100	rsvd	rsvd	3'b000	rsvd	gtx op	yes	no	nsu	1	N/A	N/A	N/A	N/A	N/A	N/A	N/A	instruction cache flush
halt	4'b1111	3'b111	rsvd	rsvd	3'b000	rsvd	gtx op	yes	no	nsu	1	N/A	N/A	N/A	N/A	N/A	N/A	N/A	halt with no condition

"""

# 
F7_WAIT:int = 0b1111001        # Wait
F7_INTR:int = 0b1111011        # Interrupt
F7_FLUSH:int = 0b1111100       # Flush
F7_HALT:int = 0b1111111        # Halt
