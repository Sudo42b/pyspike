// =================================================================
// Copyright   : (C) Supergate - All Rights Reserved
// Project     : VTS
// Description : N4S16 - matmul ntn (R = A x R(transposed))
// Author      : mh.kim ( NPU Div - NPU Core Design Team )    
// Last Update : 2026/03/20
// =================================================================
#include "intrin.h"
#include "gtx_csr.h"
#include "gtx/address.h"

// Hardware
#define NEST_NUM                        1
#define SPU_NUM_PER_NEST                16                  
#define D_BYTE                          2                   
#define L1_BANK_SIZE                    (64 * 1024)      
#define NOT_USE                         0xBEEF

// Memory - DDR     
#define BASE_DDR_A                      GTX_MAIN_ADDR(0x1000000)
#define BASE_DDR_B                      GTX_MAIN_ADDR(0x2000000)
#define BASE_DDR_RESULT                 GTX_MAIN_ADDR(0xf000000)

// Input        
#define I_A_ROW_SIZE                    1024
#define I_H_COL_SIZE                    1024
#define I_B_ROW_SIZE                    512
#define I_A_ROW_BYTE                    (I_A_ROW_SIZE * D_BYTE)
#define I_H_COL_BYTE                    (I_H_COL_SIZE * D_BYTE)
#define I_B_ROW_BYTE                    (I_B_ROW_SIZE * D_BYTE)

// Memory - L2SPM       
#define BASE_L2_A                       0x000000
#define BASE_L2_B                       (BASE_L2_A + I_H_COL_BYTE * A_ROW_PER_NEST(0))
#define BASE_L2_RESULT                  (BASE_L2_B + I_H_COL_BYTE * I_B_ROW_SIZE)

// Memory - L1SPM       
#define BASE_L1_A                       0x00000
#define BASE_L1_B                       0x20000
#define BASE_L1_C                       0x30000
#define BASE_L1_R                       0x50000

// Stack
#define STACK_POINTER                   0x3FFF50  
#define STACK_SIZE                      0x34   
#define STACK_ADDR                      GTX_MAIN_ADDR(0xFE00000)
#define STACK_ENABLE                    0x1 

// Equation     
#define CEIL_DIV(a, b)                  (((a) + (b) - 1) / (b))
#define LAST_TILE(a, b)                 ((a) % (b) == 0 ? (b) : ((a) % (b)))
#define LAST_INDEX(n)                   ((n) - 1)

// Effective nest       
#define EFF_NEST_NUM                    ((CEIL_DIV(I_A_ROW_SIZE, NEST_NUM) < NEST_NUM) ? CEIL_DIV(I_A_ROW_SIZE, NEST_NUM) : NEST_NUM)
#define EFF_NEST_NUM_HEX                ((1ULL << EFF_NEST_NUM) - 1)
#define LAST_EFF_NEST                   LAST_INDEX(EFF_NEST_NUM)

// Nest tiling A row    
#define A_ROW_QUOT_NEST                 (I_A_ROW_SIZE / EFF_NEST_NUM)         
#define A_ROW_MOD_NEST                  (I_A_ROW_SIZE % EFF_NEST_NUM)   
#define A_ROW_PER_NEST(nid)             (A_ROW_QUOT_NEST + ((nid) < A_ROW_MOD_NEST ? 1 : 0))

// Effective spu
#define EFF_SPU_NUM_PER_NEST(nid)       ((CEIL_DIV(A_ROW_PER_NEST(nid), SPU_NUM_PER_NEST) < SPU_NUM_PER_NEST) ? CEIL_DIV(A_ROW_PER_NEST(nid), SPU_NUM_PER_NEST) : SPU_NUM_PER_NEST)
#define EFF_SPU_NUM_PER_NEST_HEX(nid)   ((1ULL << EFF_SPU_NUM_PER_NEST(nid)) - 1)
#define LAST_EFF_SPU(nid)               LAST_INDEX(EFF_SPU_NUM_PER_NEST(nid))

// SPU tiling A row
#define A_ROW_QUOT_SPU(nid)             (A_ROW_PER_NEST(nid) / EFF_SPU_NUM_PER_NEST(nid))
#define A_ROW_MOD_SPU(nid)              (A_ROW_PER_NEST(nid) % EFF_SPU_NUM_PER_NEST(nid))
#define A_ROW_PER_SPU(nid, tid)         (A_ROW_QUOT_SPU(nid) + ((tid) < A_ROW_MOD_SPU(nid) ? 1 : 0))

// Tile size
#define TILE_H_COL                      (((L1_BANK_SIZE / 2) / A_ROW_PER_SPU(0, 0)) < I_H_COL_SIZE ? ((L1_BANK_SIZE / 2) / A_ROW_PER_SPU(0, 0)) : I_H_COL_SIZE)              
#define TILE_B_ROW                      (((L1_BANK_SIZE / 2) / TILE_H_COL) < I_B_ROW_SIZE ? ((L1_BANK_SIZE / 2) / TILE_H_COL) : I_B_ROW_SIZE)
#define TILE_H_COL_BYTE                 (TILE_H_COL * D_BYTE)
#define TILE_B_ROW_BYTE                 (TILE_B_ROW * D_BYTE)

// Tile num
#define NUM_TILE_H_COL                  CEIL_DIV(I_H_COL_SIZE, TILE_H_COL)
#define NUM_TILE_B_ROW                  CEIL_DIV(I_B_ROW_SIZE, TILE_B_ROW)

// Last tile size
#define LAST_TILE_H_COL                 LAST_TILE(I_H_COL_SIZE, TILE_H_COL)
#define LAST_TILE_B_ROW                 LAST_TILE(I_B_ROW_SIZE, TILE_B_ROW)

// Last index
#define LAST_IDX_H_COL                  LAST_INDEX(NUM_TILE_H_COL)
#define LAST_IDX_B_ROW                  LAST_INDEX(NUM_TILE_B_ROW)    

// Offset
#define OFFSET_H_COL                    TILE_H_COL_BYTE
#define OFFSET_B_ROW                    (I_H_COL_BYTE * TILE_B_ROW)
#define OFFSET_RESULT_COL               TILE_B_ROW_BYTE

// Calculation type enable
#define MM_EN                           (NUM_TILE_H_COL == 1)


int main(void) {

    //=============================================================================
    // Set varibles
    //=============================================================================
    uint32_t offset_ddr_A_row = 0;


    //=============================================================================
    // GTX RUN - preload
    //=============================================================================
    // Load matrix B
    __mcast_g2s(
        BASE_DDR_B                      ,
        BASE_L2_B                       ,
        I_H_COL_BYTE                    ,
        I_H_COL_BYTE                    ,
        I_B_ROW_SIZE                    ,
        EFF_NEST_NUM_HEX                  
    );

    __split();

        for (uint8_t nest_id = 0; nest_id < EFF_NEST_NUM; nest_id++) {
            
            // Adjust status
            uint8_t  eff_spu    = EFF_SPU_NUM_PER_NEST(nest_id);
            uint16_t nest_A_row = A_ROW_PER_NEST(nest_id);
            
            // Adjust offset
            uint32_t offset_l2_A_row = 0;

            __start_plan(nest_id);

                __start_shared();

                    // Load matrix A
                    __load(
                        BASE_DDR_A + offset_ddr_A_row       ,
                        BASE_L2_A                           ,
                        I_H_COL_BYTE                        ,
                        I_H_COL_BYTE                        ,
                        nest_A_row                          ,
                        I_H_COL_BYTE     
                    );
                    
                    // Load credit inc
                    __credit_ld(EFF_SPU_NUM_PER_NEST_HEX(nest_id), NOT_USE);
                    
                    // Adjust offset
                    offset_ddr_A_row += I_H_COL_BYTE * nest_A_row;
                    
                __end_shared();

                for (uint8_t thread_id = 0; thread_id < eff_spu; thread_id++) {

                    __start_thread(thread_id);

                        // Adjust status
                        uint16_t eff_A_row = A_ROW_PER_SPU(nest_id, thread_id);

                        // Set L1SPM bank start address
                        __set_spm_addr(BASE_L1_R, BASE_L1_C, BASE_L1_B, BASE_L1_A);
                        
                        // Load credit inc check
                        __credit_chk(NOT_USE);

                        // Load matrix A tile
                        __load(
                            BASE_L2_A + offset_l2_A_row             ,
                            BASE_L1_A                               ,
                            I_H_COL_BYTE                            ,
                            TILE_H_COL_BYTE                         ,
                            eff_A_row                               ,
                            NOT_USE
                        );
                        
                        // Load credit dec
                        __credit_ld(NOT_USE, NOT_USE);

                        // Adjust offset
                        offset_l2_A_row += I_H_COL_BYTE * eff_A_row;

                    __end_thread(thread_id);
                }

            __end_plan(nest_id);
        }

    __join();


    //=============================================================================
    // GTX RUN - calculation
    //=============================================================================
    // Only 1 column tile
    if (MM_EN) {
        
        // All b row tile
        for (uint16_t idx_b_row = 0; idx_b_row < NUM_TILE_B_ROW; idx_b_row++) {
            
            // Adjust status
            uint8_t  last_b_row_en = (idx_b_row == LAST_IDX_B_ROW);
            uint16_t eff_B_row     = last_b_row_en ? LAST_TILE_B_ROW : TILE_B_ROW;

            // Adjust offset
            uint32_t offset_B_row          = OFFSET_B_ROW      * idx_b_row;
            uint32_t offset_RESULT_col     = OFFSET_RESULT_COL * idx_b_row;
            uint32_t offset_ddr_RESULT_row = 0;

            __split();
                
                for (uint8_t nest_id = 0; nest_id < EFF_NEST_NUM; nest_id++) {
                    
                    // Adjust status
                    uint8_t  eff_spu     = EFF_SPU_NUM_PER_NEST(nest_id);
                    uint16_t eff_spu_hex = EFF_SPU_NUM_PER_NEST_HEX(nest_id);

                    // Adjust offset
                    uint32_t offset_l2_RESULT_row = 0;
                    
                    __start_plan(nest_id);

                        // Mcast matrix B tile
                        __mcast_s2l(
                            BASE_L2_B + offset_B_row    , 
                            BASE_L1_B                   , 
                            I_H_COL_BYTE                , 
                            I_H_COL_BYTE                , 
                            TILE_B_ROW                  , 
                            eff_spu_hex
                        );

                        __start_shared();

                            // Last B row tile
                            if (last_b_row_en) {
                                
                                // Store result tile
                                for (uint8_t thread_id = 0; thread_id < eff_spu; thread_id++) {
                                    
                                    // Adjust status
                                    uint16_t target    = (0x1 << thread_id);
                                    uint16_t eff_A_row = A_ROW_PER_SPU(nest_id, thread_id);
                                    
                                    // Adjust offset
                                    uint32_t offset_A_row_unit = I_B_ROW_BYTE * eff_A_row;

                                    // Store credit inc check
                                    __credit_chk(target);

                                    // Store tensor result first-level tile per thread
                                    __store(
                                        BASE_L2_RESULT  + offset_l2_RESULT_row                      ,   
                                        BASE_DDR_RESULT + offset_ddr_RESULT_row                     ,   
                                        I_B_ROW_BYTE                                                ,   
                                        I_B_ROW_BYTE                                                ,   
                                        eff_A_row                                                   ,   
                                        I_B_ROW_BYTE                                       
                                    );

                                    // Store credit dec
                                    __credit_st(target);   

                                    // Adjust offset
                                    offset_l2_RESULT_row  += offset_A_row_unit;
                                    offset_ddr_RESULT_row += offset_A_row_unit;
                                }

                                // Adjust offset
                                offset_l2_RESULT_row = 0;
                            }
                            
                        __end_shared();

                        for (uint8_t thread_id = 0; thread_id < eff_spu; thread_id++) {

                            __start_thread(thread_id);
                                
                                // Adjust status
                                uint16_t eff_A_row = A_ROW_PER_SPU(nest_id, thread_id);
                                
                                // [mm]
                                __mm(
                                    eff_A_row       ,
                                    I_H_COL_SIZE    ,
                                    eff_B_row   
                                );
                                
                                // Store result tile 
                                __store(
                                    BASE_L1_R                                                   ,
                                    BASE_L2_RESULT + offset_l2_RESULT_row + offset_RESULT_col   ,
                                    I_B_ROW_BYTE                                                ,
                                    eff_B_row * D_BYTE                                          ,
                                    eff_A_row                                                   ,
                                    NOT_USE                                                                                              
                                );

                                // Last row tile -> store credit inc 
                                if (last_b_row_en){
                                    __credit_st(NOT_USE);
                                }

                                // Adjust offset
                                offset_l2_RESULT_row += I_B_ROW_BYTE * eff_A_row;

                            __end_thread(thread_id);
                        }

                    __end_plan(nest_id);
                }

            __join();
        }
    }

    // Over 2 column tile
    else {

        // All B row tile
        for (uint16_t idx_b_row = 0; idx_b_row < NUM_TILE_B_ROW; idx_b_row++) {
            
            // Adjust status
            uint8_t  last_b_row_en = (idx_b_row == LAST_IDX_B_ROW);
            uint16_t eff_B_row     = last_b_row_en ? LAST_TILE_B_ROW : TILE_B_ROW;

            // Adjust offset
            uint32_t offset_B_row          = OFFSET_B_ROW      * idx_b_row;
            uint32_t offset_RESULT_col     = OFFSET_RESULT_COL * idx_b_row;
            uint32_t offset_ddr_RESULT_row = 0;
            
            // All h col tile
            for (uint16_t idx_h_col = 0; idx_h_col < NUM_TILE_H_COL; idx_h_col++) {

                // Adjust status
                uint8_t  last_h_col_en = (idx_h_col == LAST_IDX_H_COL);
                uint16_t eff_H_col     = last_h_col_en ? LAST_TILE_H_COL : TILE_H_COL;
                uint16_t eff_H_col_nxt = (idx_h_col == (LAST_IDX_H_COL - 1)) ? LAST_TILE_H_COL : TILE_H_COL;

                // Adjust offset
                uint32_t offset_H_col     = OFFSET_H_COL * idx_h_col;
                uint32_t offset_H_col_nxt = last_h_col_en ? 0 : OFFSET_H_COL * (idx_h_col + 1);

                // Set GSPR (stack recovery)
                __wrspr(
                    STACK_INFO                                          ,
                    0                                                   ,
                    ((uint64_t) STACK_SIZE << 48) | (STACK_POINTER)     ,
                    NOT_USE                                               
                );
            
                // Set GSPR (stack save)
                __wrspr(
                    STACK_SAVE                                          ,
                    0                                                   ,
                    ((uint64_t) STACK_ENABLE << 48) | (STACK_ADDR)      ,
                    NOT_USE                                               
                );

                __split();
                    
                    for (uint8_t nest_id = 0; nest_id < EFF_NEST_NUM; nest_id++) {
                        
                        // Adjust status
                        uint8_t  eff_spu     = EFF_SPU_NUM_PER_NEST(nest_id);
                        uint16_t eff_spu_hex = EFF_SPU_NUM_PER_NEST_HEX(nest_id);

                        // Adjust offset
                        uint32_t offset_l2_A_row      = 0;
                        uint32_t offset_l2_RESULT_row = 0;

                        __start_plan(nest_id);

                            // Mcast matrix B tile
                            __mcast_s2l(
                                BASE_L2_B + offset_B_row + offset_H_col         , 
                                BASE_L1_B                                       , 
                                I_H_COL_BYTE                                    , 
                                eff_H_col * D_BYTE                              , 
                                eff_B_row                                       , 
                                eff_spu_hex
                            );

                            __start_shared();

                                // Last B row & H col tile
                                if (last_b_row_en && last_h_col_en) {
                                    
                                    // Store result tile
                                    for (uint8_t thread_id = 0; thread_id < eff_spu; thread_id++) {
                                        
                                        // Adjust status
                                        uint16_t target    = (0x1 << thread_id);
                                        uint16_t eff_A_row = A_ROW_PER_SPU(nest_id, thread_id);

                                        // Adjust offset
                                        uint32_t offset_A_row_unit = I_B_ROW_BYTE * eff_A_row;

                                        // Store credit inc check
                                        __credit_chk(target);

                                        // Store tensor result first-level tile per thread
                                        __store(
                                            BASE_L2_RESULT  + offset_l2_RESULT_row                  ,   
                                            BASE_DDR_RESULT + offset_ddr_RESULT_row                 ,   
                                            I_B_ROW_BYTE                                            ,   
                                            I_B_ROW_BYTE                                            ,   
                                            eff_A_row                                               ,   
                                            I_B_ROW_BYTE                                       
                                        );

                                        // Store credit dec
                                        __credit_st(target); 

                                        // Adjust offset
                                        offset_l2_RESULT_row  += offset_A_row_unit;
                                        offset_ddr_RESULT_row += offset_A_row_unit;
                                    }

                                    // Adjust offset
                                    offset_l2_RESULT_row = 0;
                                }

                            __end_shared();

                            for (uint8_t thread_id = 0; thread_id < eff_spu; thread_id++) {

                                __start_thread(thread_id);
                                    
                                    // Adjust status
                                    uint16_t eff_A_row = A_ROW_PER_SPU(nest_id, thread_id);

                                    // [mm.s] initial tile
                                    if (idx_h_col == 0) {
                                        __mm_s(
                                            eff_A_row   ,
                                            eff_H_col   ,
                                            eff_B_row   
                                        );
                                    }
                                    // [mmc] last tile
                                    else if (last_h_col_en) {
                                        __mmc(
                                            eff_A_row    ,
                                            eff_H_col    ,
                                            eff_B_row                 
                                        );  
                                    }
                                    // [mmc.s] middle tile
                                    else{
                                        __mmc_s(
                                            eff_A_row    ,
                                            eff_H_col    ,
                                            eff_B_row                 
                                        );   
                                    }
                                    
                                    // Last column tile -> store matrix RES tile
                                    if (last_h_col_en) {
                                        __store(
                                            BASE_L1_R                                                   ,
                                            BASE_L2_RESULT + offset_l2_RESULT_row + offset_RESULT_col   ,
                                            I_B_ROW_BYTE                                                ,
                                            eff_B_row * D_BYTE                                          ,
                                            eff_A_row                                                   ,
                                            NOT_USE                                                                                              
                                        );

                                        if (last_b_row_en){
                                            __credit_st(NOT_USE); 
                                        }

                                        // Adjust offset
                                        offset_l2_RESULT_row += I_B_ROW_BYTE * eff_A_row;
                                    }

                                    // Load next A tile 
                                    if (!(last_h_col_en && last_b_row_en)) {
                                        __load(
                                            BASE_L2_A + offset_l2_A_row + offset_H_col_nxt              ,
                                            BASE_L1_A                                                   ,
                                            I_H_COL_BYTE                                                ,
                                            eff_H_col_nxt * D_BYTE                                      ,
                                            eff_A_row                                                   ,
                                            NOT_USE
                                        );

                                        // Adjust offset
                                        offset_l2_A_row += I_H_COL_BYTE * eff_A_row;
                                    }
                                    
                                __end_thread(thread_id);
                            }

                        __end_plan(nest_id);
                    }

                __join();
            }
        }
    }
    

    return 0;
}
