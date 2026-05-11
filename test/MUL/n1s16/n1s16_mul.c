// =================================================================
// Copyright   : (C) Supergate - All Rights Reserved
// Project     : VTS
// Description : N1 add kernel test (vector + vector)
// Author      : mh.kim ( NPU Div - NPU Core Design Team )    
// Last Update : 2026/03/04
// =================================================================
#include "gtx/intrinsics/intrin.h"
#include "gtx/address.h"

// Hardware
#define TARGET_NEST_ID                  0                 
#define SPU_NUM_PER_NEST                16                
#define D_TYPE                          2                 
#define L1_BANK_SIZE                    (64 * 3 * 1024)         
#define NOT_USE                         0xBEEF

// Memory - DDR
#define BASE_DDR_A                      GTX_MAIN_ADDR(0x1000000)
#define BASE_DDR_B                      GTX_MAIN_ADDR(0x2000000)
#define BASE_DDR_RES                    GTX_MAIN_ADDR(0xf000000)

// Memory - L2SPM
#define BASE_L2_A                       0x000000    
#define BASE_L2_B                       0x050000    
#define BASE_L2_RES                     0x0a0000    
#define BASE_L2_BANK_OFFSET             0x100000  

// Memory - L1SPM
#define BASE_L1_A                       0x00000
#define BASE_L1_B                       0x30000
#define BASE_L1_C                       0x30000
#define BASE_L1_R                       0x00000

// Input size
#define I_VECTOR_SIZE                   (1024 * 1024)          

// Equation
#define CEIL_DIV(a, b)                  (((a) + (b) - 1) / (b))
#define MIN(a,b)                        ((a) < (b) ? (a) : (b))
#define MAX(a,b)                        ((a) > (b) ? (a) : (b))

// Tiling parameter
#define CAL_QUOT                        (I_VECTOR_SIZE / SPU_NUM_PER_NEST)
#define CAL_MOD                         (I_VECTOR_SIZE % SPU_NUM_PER_NEST)

// Tiling calculation
#define CAL_NUM_PER_THREAD(tid)         ((tid) < CAL_MOD ? (CAL_QUOT + 1) : CAL_QUOT)
#define OFFSET_THREAD(tid)              (((MIN((tid), CAL_MOD) * (CAL_QUOT + 1)) + (MAX(0, (int)(tid) - (int)CAL_MOD) * CAL_QUOT)) * D_TYPE)


int main(void) {

    //=============================================================================
    // Set variables
    //=============================================================================
    uint32_t bank_offset_shared = 0;
    uint32_t bank_offset_thread = 0;


    //=============================================================================
    // GTX run
    //=============================================================================
    __split();

        __start_plani(TARGET_NEST_ID);

            // Set L1SPM bank start address 
            __set_spm_addr(BASE_L1_R, BASE_L1_C, BASE_L1_B, BASE_L1_A);

            __start_shared();

                for (uint8_t thread_id = 0; thread_id < SPU_NUM_PER_NEST; thread_id++) {
                    
                    // Adjust status
                    uint32_t cal_num_per_thread = CAL_NUM_PER_THREAD(thread_id);
                    
                    // Load vector A tile per thread
                    __load(
                        BASE_DDR_A + OFFSET_THREAD(thread_id)               ,   
                        BASE_L2_A  + bank_offset_shared                     ,   
                        cal_num_per_thread                                  ,   
                        cal_num_per_thread                                  ,   
                        D_TYPE                                              ,   
                        cal_num_per_thread                                                                                     
                    );

                    // Load vector B tile per thread
                    __load(
                        BASE_DDR_B + OFFSET_THREAD(thread_id)               ,   
                        BASE_L2_B  + bank_offset_shared                     ,   
                        cal_num_per_thread                                  ,   
                        cal_num_per_thread                                  ,   
                        D_TYPE                                              ,   
                        cal_num_per_thread                                                                                     
                    );

                    // Load credit inc 
                    __credit_ld(0x1 << (thread_id), NOT_USE);

                    // Adjust offset
                    bank_offset_shared += BASE_L2_BANK_OFFSET;
                }

                // Adjust offset
                bank_offset_shared = 0;

                for (uint8_t thread_id = 0; thread_id < SPU_NUM_PER_NEST; thread_id++) {
                    
                    // Adjust status
                    uint32_t cal_num_per_thread = CAL_NUM_PER_THREAD(thread_id);
                    uint16_t target             = (0x1 << thread_id);

                    // Store credit inc check
                    __credit_chk(target);

                    // Store vector RES tile per thread
                    __store(
                        BASE_L2_RES  + bank_offset_shared                   ,   
                        BASE_DDR_RES + OFFSET_THREAD(thread_id)             ,   
                        cal_num_per_thread                                  ,   
                        cal_num_per_thread                                  ,   
                        D_TYPE                                              ,   
                        cal_num_per_thread                                  
                    );

                    // Store credit dec
                    __credit_st(target);   

                    // Adjust offset
                    bank_offset_shared += BASE_L2_BANK_OFFSET;
                }

            __end_shared();

             for (uint8_t thread_id = 0; thread_id < SPU_NUM_PER_NEST; thread_id++) {

                __start_thread(thread_id);

                    // Assign base address
                    uint32_t L2_A_thread   = BASE_L2_A   + bank_offset_thread;
                    uint32_t L2_B_thread   = BASE_L2_B   + bank_offset_thread;
                    uint32_t L2_RES_thread = BASE_L2_RES + bank_offset_thread;

                    // Adjust status
                    uint32_t cal_num_per_thread = CAL_NUM_PER_THREAD(thread_id);   

                    // Load credit inc check
                    __credit_chk(NOT_USE);
                    
                    // Load vector A last tile
                    __load(
                        L2_A_thread                                                     ,
                        BASE_L1_A                                                       ,
                        cal_num_per_thread                                              ,
                        cal_num_per_thread                                              ,
                        D_TYPE                                                          ,
                        NOT_USE
                    );

                    // Load vector B last tile
                    __load(
                        L2_B_thread                                                     ,
                        BASE_L1_B                                                       ,
                        cal_num_per_thread                                              ,
                        cal_num_per_thread                                              ,
                        D_TYPE                                                          ,
                        NOT_USE
                    );

                    // Load credit dec
                    __credit_ld(NOT_USE, NOT_USE);

                    // [add.vv]
                    __mul_vv(cal_num_per_thread);

                    // Store vector result last tile
                    __store(
                        BASE_L1_R                                                       ,
                        L2_RES_thread                                                   ,
                        cal_num_per_thread                                              ,
                        cal_num_per_thread                                              ,
                        D_TYPE                                                          ,
                        NOT_USE                                                          
                    );

                    // Store credit inc
                    __credit_st(NOT_USE);

                    // Adjust offset
                    bank_offset_thread += BASE_L2_BANK_OFFSET;
                    
                __end_thread(thread_id);
            }


        __end_plani(TARGET_NEST_ID);
        
    __join();


    return 0;
}