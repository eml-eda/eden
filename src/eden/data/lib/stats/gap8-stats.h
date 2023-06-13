#ifndef __STATS_H__
#define __STATS_H__

#ifdef STATS

#define HOTTING 1 /* Necessary Iterations to avoid cache cold effects */
#define REPEAT  5 /* Averaging on 5 successive Iterations */ 
#define COUNTER 17


#define INIT_STATS()  \
    PI_CL_L1 unsigned long _k           = 0; \
    PI_CL_L1 unsigned long _cycles      = 0; \
    PI_CL_L1 unsigned long _instr       = 0; \
    PI_CL_L1 unsigned long _active      = 0; \
    PI_CL_L1 unsigned long _ldext       = 0; \
    PI_CL_L1 unsigned long _tcdmcont    = 0; \
    PI_CL_L1 unsigned long _ldstall     = 0; \
    PI_CL_L1 unsigned long _jrstall     = 0; \
    PI_CL_L1 unsigned long _jump        = 0; \
    PI_CL_L1 unsigned long _st          = 0; \
    PI_CL_L1 unsigned long _ld          = 0; \
    PI_CL_L1 unsigned long _st_ext      = 0; \
    PI_CL_L1 unsigned long _st_ext_cyc  = 0; \
    PI_CL_L1 unsigned long _ld_ext_cyc  = 0; \
    PI_CL_L1 unsigned long _btaken      = 0; \
    PI_CL_L1 unsigned long _branch      = 0; \
    PI_CL_L1 unsigned long _rvc         = 0; \
    PI_CL_L1 unsigned long _imiss       = 0;


#define EXPORT_STATS()  \
    extern unsigned long _k; \
    extern unsigned long _cycles; \
    extern unsigned long _instr; \
    extern unsigned long _active; \
    extern unsigned long _ldext; \
    extern unsigned long _tcdmcont; \
    extern unsigned long _ldstall; \
    extern unsigned long _jrstall; \
    extern unsigned long _jump; \
    extern unsigned long _st; \
    extern unsigned long _ld; \
    extern unsigned long _st_ext; \
    extern unsigned long _st_ext_cyc; \
    extern unsigned long _ld_ext_cyc; \
    extern unsigned long _btaken; \
    extern unsigned long _branch; \
    extern unsigned long _rvc; \
    extern unsigned long _imiss;


#define ENTER_LOOP_STATS()  \
  for(_k = 0; _k < HOTTING + COUNTER*REPEAT; _k++) \
  { \
    { \
      if((_k - HOTTING) % COUNTER == 0)  pi_perf_conf((1<<PI_PERF_CYCLES)); \
      if((_k - HOTTING) % COUNTER == 1)  pi_perf_conf((1<<PI_PERF_INSTR)); \
      if((_k - HOTTING) % COUNTER == 2)  pi_perf_conf((1<<PI_PERF_ACTIVE_CYCLES)); \
      if((_k - HOTTING) % COUNTER == 3)  pi_perf_conf((1<<PI_PERF_LD_EXT)); \
      if((_k - HOTTING) % COUNTER == 4)  pi_perf_conf((1<<PI_PERF_TCDM_CONT)); \
      if((_k - HOTTING) % COUNTER == 5)  pi_perf_conf((1<<PI_PERF_LD_STALL)); \
      if((_k - HOTTING) % COUNTER == 7)  pi_perf_conf((1<<PI_PERF_JR_STALL)); \
      if((_k - HOTTING) % COUNTER == 8)  pi_perf_conf((1<<PI_PERF_JUMP)); \
      if((_k - HOTTING) % COUNTER == 9)  pi_perf_conf((1<<PI_PERF_ST)); \
      if((_k - HOTTING) % COUNTER == 10) pi_perf_conf((1<<PI_PERF_LD)); \
      if((_k - HOTTING) % COUNTER == 11) pi_perf_conf((1<<PI_PERF_ST_EXT)); \
      if((_k - HOTTING) % COUNTER == 12) pi_perf_conf((1<<PI_PERF_ST_EXT_CYC)); \
      if((_k - HOTTING) % COUNTER == 13) pi_perf_conf((1<<PI_PERF_LD_EXT_CYC)); \
      if((_k - HOTTING) % COUNTER == 14) pi_perf_conf((1<<PI_PERF_BTAKEN)); \
      if((_k - HOTTING) % COUNTER == 15) pi_perf_conf((1<<PI_PERF_BRANCH)); \
      if((_k - HOTTING) % COUNTER == 16) pi_perf_conf((1<<PI_PERF_RVC)); \
      if((_k - HOTTING) % COUNTER == 17) pi_perf_conf((1<<PI_PERF_IMISS)); \
    }


#define START_STATS()  \
    if (_k >= HOTTING) \
    { \
      pi_perf_reset(); \
      pi_perf_start(); \
    }


#define STOP_STATS() \
    if (_k >= HOTTING) \
    { \
      pi_perf_stop(); \
      if((_k - HOTTING) % COUNTER == 0)  _cycles     += pi_perf_read(PI_PERF_CYCLES); \
      if((_k - HOTTING) % COUNTER == 1)  _instr      += pi_perf_read(PI_PERF_INSTR); \
      if((_k - HOTTING) % COUNTER == 2)  _active     += pi_perf_read(PI_PERF_ACTIVE_CYCLES); \
      if((_k - HOTTING) % COUNTER == 3)  _ldext      += pi_perf_read(PI_PERF_LD_EXT); \
      if((_k - HOTTING) % COUNTER == 4)  _tcdmcont   += pi_perf_read(PI_PERF_TCDM_CONT); \
      if((_k - HOTTING) % COUNTER == 5)  _ldstall    += pi_perf_read(PI_PERF_LD_STALL); \
      if((_k - HOTTING) % COUNTER == 7)  _jrstall    += pi_perf_read(PI_PERF_JR_STALL); \
      if((_k - HOTTING) % COUNTER == 8)  _jump       += pi_perf_read(PI_PERF_JUMP); \
      if((_k - HOTTING) % COUNTER == 9)  _st         += pi_perf_read(PI_PERF_ST); \
      if((_k - HOTTING) % COUNTER == 10) _ld         += pi_perf_read(PI_PERF_LD); \
      if((_k - HOTTING) % COUNTER == 11) _st_ext     += pi_perf_read(PI_PERF_ST_EXT); \
      if((_k - HOTTING) % COUNTER == 12) _st_ext_cyc += pi_perf_read(PI_PERF_ST_EXT_CYC); \
      if((_k - HOTTING) % COUNTER == 13) _ld_ext_cyc += pi_perf_read(PI_PERF_LD_EXT_CYC); \
      if((_k - HOTTING) % COUNTER == 14) _btaken     += pi_perf_read(PI_PERF_BTAKEN); \
      if((_k - HOTTING) % COUNTER == 15) _branch     += pi_perf_read(PI_PERF_BRANCH); \
      if((_k - HOTTING) % COUNTER == 16) _rvc        += pi_perf_read(PI_PERF_RVC); \
      if((_k - HOTTING) % COUNTER == 17) _imiss      += pi_perf_read(PI_PERF_IMISS); \
    }


#define EXIT_LOOP_STATS()  \
  } \
  printf("Stats inference - start\n"); \
  printf("Cycles = %lu\n", _cycles/(REPEAT)); \
  printf("Instr = %lu\n",  _instr/(REPEAT)); \
  printf("Active Cycles = %lu\n", _active/(REPEAT)); \
  printf("EXT LD = %lu\n",_ldext/(REPEAT)); \
  printf("TCDM Conts = %lu\n",_tcdmcont/(REPEAT)); \
  printf("LD Stall = %lu\n",_ldstall/(REPEAT)); \
  printf("JR Stall = %lu\n",_jrstall/(REPEAT)); \
  printf("JUMP = %lu\n",_jump/(REPEAT)); \
  printf("ST = %lu\n",_st/(REPEAT)); \
  printf("LD = %lu\n",_ld/(REPEAT)); \
  printf("ST EXT = %lu\n",_st_ext/(REPEAT)); \
  printf("ST EXT Cycles = %lu\n",_st_ext_cyc/(REPEAT)); \
  printf("LD EXT Cycles = %lu\n",_ld_ext_cyc/(REPEAT)); \
  printf("RVC = %lu\n",_rvc/(REPEAT)); \
  printf("BR-Taken = %lu\n",_btaken/(REPEAT)); \
  printf("BR-NTaken = %lu\n",_branch/(REPEAT)); \
  printf("I-Miss = %lu\n",_imiss/(REPEAT)); \
  printf("Stats inference - end\n"); 
#else  // ! STATS

#define INIT_STATS()
#define EXPORT_STATS()
#define ENTER_LOOP_STATS()
#define START_STATS()
#define STOP_STATS()
#define EXIT_LOOP_STATS()

#endif

#endif