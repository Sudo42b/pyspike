//==================================================================
// Copyright   : (C) Supergate - All Rights Reserved
// Project     : GSF / VTS
// Test        : set_rows_4d
// Description : GTX set_rows in nest kernel
// Author      : kyumlee ( NPU Div - NPU Core Design Team )    
// Last Update : 2025/10/15
//==================================================================

// TEST GUARD
#ifndef __SET_ROWS_4D_N1_KERNEL_C
#define __SET_ROWS_4D_N1_KERNEL_C

// Intrinsic header file
#include "intrin.h"
#include "gtx_csr.h"
#include "gtx/address.h"
//#include "gtx_ckernel_param.h"

#define NEST_TARGET_ID      0               // Target nest ID
#define SPU_NUM_PER_NEST    16              // SPU num per nest
#define DTYPE               2               // FP16
#define INT32_DTYPE         4
#define USE_BANK_SIZE       64 * 1024       // 64 KB

#define BASE_DDR_A          GTX_MAIN_ADDR(0x1000000)
#define BASE_DDR_B          GTX_MAIN_ADDR(0x2000000)
#define BASE_DDR_INDEX      GTX_MAIN_ADDR(0x3000000)
#define BASE_DDR_RESULT     GTX_MAIN_ADDR(0xf000000)

#define BASE_L2_A           0x200000
#define BASE_L2_B           0x400000 
#define BASE_L2_RESULT      0x600000

#define BASE_L1_BANK_A      0X00000
#define BASE_L1_BANK_B      0X20000
#define BASE_L1_BANK_C      0X30000
#define BASE_L1_BANK_R      0X50000

#define I_WIDTH_SIZE            1024       // width   = column
#define I_HEIGHT_SIZE           1024       // height  = row
#define I_CHANNEL_SIZE          1           // channel = depth
#define I_BATCH_SIZE            1          // batch

#define O_WIDTH_SIZE            1024       // width   = column
#define O_HEIGHT_SIZE           1024       // height  = row
#define O_CHANNEL_SIZE          1          // channel = depth
#define O_BATCH_SIZE            1          // batch

#define I2_WIDTH_SIZE           1          // width   = column
#define I2_HEIGHT_SIZE          4          // height  = row
#define I2_CHANNEL_SIZE         1          // channel = depth
#define I2_BATCH_SIZE           1          // batch

int main(void) {
    /*
    //validation
    if(I_WIDTH_SIZE != O_WIDTH_SIZE 
        || I_CHANNEL_SIZE != O_CHANNEL_SIZE
        || I_BATCH_SIZE != O_BATCH_SIZE){
        printf("\n[SET_ROWS] IN/OUT UNMATCH\n");
        return 0;
        }
    if(I_HEIGHT_SIZE != I2_HEIGHT_SIZE
        || I_CHANNEL_SIZE % I2_CHANNEL_SIZE != 0
        || I_BATCH_SIZE % I2_BATCH_SIZE != 0){
        printf("\n[SET_ROWS] BROADCAST UNMATCH\n");
        return 0;
        }
    */

    //Params
    uint8_t nest_id   = NEST_TARGET_ID;
    uint8_t thread_id = SPU_NUM_PER_NEST -1;

    uint16_t sel_batch, sel_channel, sel_height;
    uint32_t *sel_val;
    
    uint16_t wid_size = I_WIDTH_SIZE * DTYPE;

    __copy_mem(BASE_DDR_A,
               BASE_DDR_RESULT,
               wid_size,
               wid_size,
               I_HEIGHT_SIZE,
               wid_size,
               wid_size >> 16);

    for(int i3=0; i3<I_BATCH_SIZE; i3++){
        for(int i2=0; i2<I_CHANNEL_SIZE; i2++){
            for(int i=0; i<I2_HEIGHT_SIZE; i++){
                    sel_batch = i3 % I2_BATCH_SIZE;
                    sel_channel = i2 % I2_CHANNEL_SIZE;
                    sel_height = i;

                    sel_val = (uint32_t *) (BASE_DDR_INDEX 
                                            + ((sel_batch * I2_CHANNEL_SIZE + sel_channel) * I2_HEIGHT_SIZE + sel_height) * INT32_DTYPE);

                    __copy_mem(BASE_DDR_B + ((i3 * I_CHANNEL_SIZE + i2) * I2_HEIGHT_SIZE + i) * I_WIDTH_SIZE * DTYPE,
                            BASE_DDR_RESULT + ((i3 * O_CHANNEL_SIZE + i2) * O_HEIGHT_SIZE + *sel_val) * O_WIDTH_SIZE * DTYPE,
                            O_WIDTH_SIZE * DTYPE,
                            O_WIDTH_SIZE * DTYPE,
                            1,
                            O_WIDTH_SIZE * DTYPE,
                            (O_WIDTH_SIZE * DTYPE) >> 16);
                }
            }
        } 

    return 0;
}   

#endif
