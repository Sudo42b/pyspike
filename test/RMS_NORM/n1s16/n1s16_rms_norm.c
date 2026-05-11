// =================================================================
// Copyright   : (C) Supergate - All Rights Reserved
// Project     : VTS
// Description : N1 rms norm 1d kernel test
// Author      : mh.kim ( NPU Div - NPU Core Design Team )    
// Last Update : 2026/03/30
// =================================================================
#include "intrin.h"
#include "gtx_csr.h"
#include "gtx/address.h"

//=========================================
// Hardware
//=========================================
#define NEST_ID                         0                 
#define SPU_NUM                         16                
#define D_BYTE                          2                 
#define L1_BANK_BYTE                    (64 * 3 * 1024)         

//=========================================
// Global
//=========================================
#define NOT_USE                         0xBEEF
#define DUMMY_ADDRESS                   0x0
#define EPSILON                         0x00A8

//=========================================
// Equation
//=========================================
#define BYTE(i)                         ((i) * D_BYTE)
#define CEIL_DIV(a, b)                  (((a) + (b) - 1) / (b))
#define MIN(a, b)                       ((a) < (b) ? (a) : (b))
#define MAX(a, b)                       ((a) > (b) ? (a) : (b))

//=========================================
// Input
//=========================================
#define I_WIDTH                         1024
#define I_HEIGHT                        1024
#define I_CHANNEL                       1                 
#define I_BATCH                         1   

//=========================================
// Memory
//=========================================
#define BASE_DDR_INPUT                  GTX_MAIN_ADDR(0x1000000)
#define BASE_DDR_RESULT                 GTX_MAIN_ADDR(0xf000000)

#define BASE_L2_INPUT                   0x000000    
#define BASE_L2_RESULT                  0x080000    

#define BASE_L1_A                       0x00000
#define BASE_L1_R                       0x30000

//=========================================
// Tiling
//=========================================
#define CAL_UNIT                        I_WIDTH 
#define CAL_UNIT_BYTE                   BYTE(CAL_UNIT)
#define CAL_NUM                         (I_HEIGHT * I_CHANNEL * I_BATCH)
#define CAL_MAX                         (L1_BANK_BYTE / CAL_UNIT_BYTE)
#define EFF_SPU_NUM                     ((CAL_NUM / SPU_NUM) ? SPU_NUM : CAL_NUM)
#define CAL_QUOT                        (CAL_NUM / EFF_SPU_NUM)
#define CAL_MOD                         (CAL_NUM % EFF_SPU_NUM)

#define CAL_NUM_PER_THREAD(tid)         ((tid) < CAL_MOD ? (CAL_QUOT + 1) : CAL_QUOT)
#define CAL_LAST(tid)                   ((CAL_NUM_PER_THREAD(tid) % CAL_MAX) ? (CAL_NUM_PER_THREAD(tid) % CAL_MAX) : CAL_MAX)
#define ITERATION                       (CAL_NUM_PER_THREAD(0) - 2)

//=========================================
// Offset
//=========================================
#define OFFSET_L2_BANK                  0x100000  
#define OFFSET_THREAD(tid)              (((MIN((tid), CAL_MOD) * (CAL_QUOT + 1)) + (MAX(0, (int)(tid) - (int)CAL_MOD) * CAL_QUOT)) * CAL_UNIT_BYTE)


int main(void) {

    //=============================================================================
    // Set variables
    //=============================================================================
    uint32_t offset_bank_shared = 0;
    uint32_t offset_bank_thread = 0;
    uint32_t offset_cal         = 0;
    uint16_t exe_cnt            = 0;


    //=============================================================================
    // GTX run - preload & initial operation
    //=============================================================================
    __split();

        __start_plani(NEST_ID);

            __start_shared();

                for (uint8_t thread_id = 0; thread_id < EFF_SPU_NUM; thread_id++) {
                                        
                    // Load tensor INPUT per thread
                    __load(
                        BASE_DDR_INPUT + OFFSET_THREAD(thread_id)           ,   
                        BASE_L2_INPUT  + offset_bank_shared                 ,   
                        CAL_UNIT_BYTE                                       ,   
                        CAL_UNIT_BYTE                                       ,   
                        CAL_NUM_PER_THREAD(thread_id)                       ,   
                        CAL_UNIT_BYTE                                                                          
                    );

                    // Inc load credit 
                    __credit_ld(0x1 << (thread_id), NOT_USE);

                    // Adjust offset
                    offset_bank_shared += OFFSET_L2_BANK;
                }

            __end_shared();

            for (uint8_t thread_id = 0; thread_id < EFF_SPU_NUM; thread_id++) {

                __start_thread(thread_id);
                    
                    // Adjust L1SPM bank start address 
                    __set_spm_addr_A(BASE_L1_A);
                    __set_spm_addr_R(BASE_L1_R);
                    
                    // Check load credit inc
                    __credit_chk(NOT_USE);
                    
                    // Load tensor INPUT tile
                    __load(
                        BASE_L2_INPUT + offset_bank_thread                          ,
                        BASE_L1_A                                                   ,
                        CAL_UNIT_BYTE                                               ,
                        CAL_UNIT_BYTE                                               ,
                        CAL_LAST(thread_id)                                         ,
                        NOT_USE
                    );

                    // Dec load credit
                    __credit_ld(NOT_USE, NOT_USE);

                    // [rmsnorm]
                    __rmsnorm(
                        CAL_UNIT        ,
                        BASE_L1_A       ,
                        DUMMY_ADDRESS   ,
                        BASE_L1_R       ,
                        EPSILON
                    );

                    // Adjust offset
                    offset_bank_thread += OFFSET_L2_BANK;

                __end_thread(thread_id);
            }

        __end_plani(NEST_ID);
        
    __join();

    // Update execute count
    exe_cnt++;

    // Adjust offset
    offset_bank_shared = 0;
    offset_bank_thread = 0;


    //=============================================================================
    // GTX run - operation
    //=============================================================================
    for (uint16_t iter = 0; iter < ITERATION; iter++) {

        // Adjust offset
        offset_cal += CAL_UNIT_BYTE;

        __split();

            __start_plani(NEST_ID);

                for (uint8_t thread_id = 0; thread_id < EFF_SPU_NUM; thread_id++) {
                    
                    __start_thread(thread_id);

                        // Adjust L1SPM bank start address 
                        uint32_t l1_A_addr = BASE_L1_A + offset_cal;
                        uint32_t l1_R_addr = BASE_L1_R + offset_cal;

                        __set_spm_addr_A(l1_A_addr);
                        __set_spm_addr_R(l1_R_addr);

                        // [rmsnorm]
                        __rmsnorm(
                            CAL_UNIT        ,
                            l1_A_addr       ,
                            DUMMY_ADDRESS   ,
                            l1_R_addr       ,
                            EPSILON
                        );

                    __end_thread(thread_id);
                }

            __end_plani(NEST_ID);
            
        __join();

        // Update execute count
        exe_cnt++;
    }
    

    //=============================================================================
    // GTX run - last operation & store to DDR
    //=============================================================================
    // Adjust offset
    offset_cal += CAL_UNIT_BYTE;

    __split();

        __start_plani(NEST_ID);

            __start_shared();

                for (uint8_t thread_id = 0; thread_id < EFF_SPU_NUM; thread_id++) {

                    // Adjust target
                    uint16_t target = (0x1 << thread_id);

                    // Check store credit inc
                    __credit_chk(target);

                    // Store tensor RESULT per thread
                    __store(
                        BASE_L2_RESULT  + offset_bank_shared                ,   
                        BASE_DDR_RESULT + OFFSET_THREAD(thread_id)          ,   
                        CAL_UNIT_BYTE                                       ,   
                        CAL_UNIT_BYTE                                       ,   
                        CAL_NUM_PER_THREAD(thread_id)                       ,   
                        CAL_UNIT_BYTE                                       
                    );

                    // Dec store credit
                    __credit_st(target); 

                    // Adjust offset
                    offset_bank_shared += OFFSET_L2_BANK;      
                }

            __end_shared();

            for (uint8_t thread_id = 0; thread_id < EFF_SPU_NUM; thread_id++) {

                __start_thread(thread_id);

                    if (exe_cnt < CAL_NUM_PER_THREAD(thread_id)) {

                        // Adjust L1SPM bank start address 
                        uint32_t l1_A_addr = BASE_L1_A + offset_cal;
                        uint32_t l1_R_addr = BASE_L1_R + offset_cal;

                        __set_spm_addr_A(l1_A_addr);
                        __set_spm_addr_R(l1_R_addr);
    
                        // [rmsnorm]
                        __rmsnorm(
                            CAL_UNIT        ,
                            l1_A_addr       ,
                            DUMMY_ADDRESS   ,
                            l1_R_addr       ,
                            EPSILON
                        );
                    }
                    
                    // Store tensor result tile
                    __store(
                        BASE_L1_R                                       ,
                        BASE_L2_RESULT + offset_bank_thread             ,
                        CAL_UNIT_BYTE                                   ,
                        CAL_UNIT_BYTE                                   ,
                        CAL_LAST(thread_id)                             ,
                        NOT_USE                                                          
                    );

                    // Inc store credit
                    __credit_st(NOT_USE);

                    // Adjust offset
                    offset_bank_thread += OFFSET_L2_BANK;

                __end_thread(thread_id);
            }

        __end_plani(NEST_ID);
        
    __join();


    return 0;
}
