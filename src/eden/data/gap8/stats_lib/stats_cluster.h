#ifndef __STATS_CLUSTER_H__
#define __STATS_CLUSTER_H__

#ifdef STATS_CLUSTER

#define HOTTING 1  /* Necessary Iterations to avoid cache cold effects */
#define REPEAT  5 /* Averaging on 5 successive Iterations */ 
#define COUNTER 17

#define INIT_STATS_CLUSTER()  \
    PI_CL_L1 unsigned long _cycles[N_CORES]      = {0}; \
    PI_CL_L1 unsigned long _instr[N_CORES]       = {0}; \
    PI_CL_L1 unsigned long _active[N_CORES]      = {0}; \
    PI_CL_L1 unsigned long _ldext[N_CORES]       = {0}; \
    PI_CL_L1 unsigned long _tcdmcont[N_CORES]    = {0}; \
    PI_CL_L1 unsigned long _ldstall[N_CORES]     = {0}; \
    PI_CL_L1 unsigned long _jrstall[N_CORES]     = {0}; \
    PI_CL_L1 unsigned long _jump[N_CORES]        = {0}; \
    PI_CL_L1 unsigned long _st[N_CORES]          = {0}; \
    PI_CL_L1 unsigned long _ld[N_CORES]          = {0}; \
    PI_CL_L1 unsigned long _st_ext[N_CORES]      = {0}; \
    PI_CL_L1 unsigned long _st_ext_cyc[N_CORES]  = {0}; \
    PI_CL_L1 unsigned long _ld_ext_cyc[N_CORES]  = {0}; \
    PI_CL_L1 unsigned long _btaken[N_CORES]      = {0}; \
    PI_CL_L1 unsigned long _branch[N_CORES]      = {0}; \
    PI_CL_L1 unsigned long _rvc[N_CORES]         = {0}; \
    PI_CL_L1 unsigned long _imiss[N_CORES]       = {0};


#define EXPORT_STATS_CLUSTER()  \
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


#define ENTER_LOOP_CLUSTER()  \
  for(int _k = 0; _k < HOTTING + COUNTER*REPEAT; _k++) \
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


#define START_STATS_CLUSTER()  \
    if (_k >= HOTTING) \
    { \
      pi_perf_reset(); \
      pi_perf_start(); \
    }


#define STOP_STATS_CLUSTER() \
    if (_k >= HOTTING) \
    { \
      pi_cl_team_barrier(); \
      pi_perf_stop(); \
      if((_k - HOTTING) % COUNTER == 0)  _cycles[core_id]     += pi_perf_read(PI_PERF_CYCLES); \
      if((_k - HOTTING) % COUNTER == 1)  _instr[core_id]      += pi_perf_read(PI_PERF_INSTR); \
      if((_k - HOTTING) % COUNTER == 2)  _active[core_id]     += pi_perf_read(PI_PERF_ACTIVE_CYCLES); \
      if((_k - HOTTING) % COUNTER == 3)  _ldext[core_id]      += pi_perf_read(PI_PERF_LD_EXT); \
      if((_k - HOTTING) % COUNTER == 4)  _tcdmcont[core_id]   += pi_perf_read(PI_PERF_TCDM_CONT); \
      if((_k - HOTTING) % COUNTER == 5)  _ldstall[core_id]    += pi_perf_read(PI_PERF_LD_STALL); \
      if((_k - HOTTING) % COUNTER == 7)  _jrstall[core_id]    += pi_perf_read(PI_PERF_JR_STALL); \
      if((_k - HOTTING) % COUNTER == 8)  _jump[core_id]       += pi_perf_read(PI_PERF_JUMP); \
      if((_k - HOTTING) % COUNTER == 9)  _st[core_id]         += pi_perf_read(PI_PERF_ST); \
      if((_k - HOTTING) % COUNTER == 10) _ld[core_id]         += pi_perf_read(PI_PERF_LD); \
      if((_k - HOTTING) % COUNTER == 11) _st_ext[core_id]     += pi_perf_read(PI_PERF_ST_EXT); \
      if((_k - HOTTING) % COUNTER == 12) _st_ext_cyc[core_id] += pi_perf_read(PI_PERF_ST_EXT_CYC); \
      if((_k - HOTTING) % COUNTER == 13) _ld_ext_cyc[core_id] += pi_perf_read(PI_PERF_LD_EXT_CYC); \
      if((_k - HOTTING) % COUNTER == 14) _btaken[core_id]     += pi_perf_read(PI_PERF_BTAKEN); \
      if((_k - HOTTING) % COUNTER == 15) _branch[core_id]     += pi_perf_read(PI_PERF_BRANCH); \
      if((_k - HOTTING) % COUNTER == 16) _rvc[core_id]        += pi_perf_read(PI_PERF_RVC); \
      if((_k - HOTTING) % COUNTER == 17) _imiss[core_id]      += pi_perf_read(PI_PERF_IMISS); \
    }


#define EXIT_LOOP_CLUSTER()  \
  } 


#define PRINT_STATS_CLUSTER()  \
  printf("Cluster Statistics\n"); \
  printf("\n"); \
  for (int i = 0; i < N_CORES; i++) printf("Core %d: Cycles = %lu\n",i,_cycles[i]/(REPEAT)); \
  for (int i = 1; i < N_CORES; i++) _cycles[0] += _cycles[i]; \
  printf("Total Cycles = %lu\n",_cycles[0]/(REPEAT)); \
  printf("Avg Cycles = %lu\n",_cycles[0]/(REPEAT*N_CORES)); \
  printf("\n"); \
  for (int i = 0; i < N_CORES; i++) printf("Core %d: Instr = %lu\n",i,_instr[i]/(REPEAT)); \
  for (int i = 1; i < N_CORES; i++) _instr[0] += _instr[i]; \
  printf("Total Instr = %lu\n",_instr[0]/(REPEAT)); \
  printf("Avg Instr = %lu\n",_instr[0]/(REPEAT*N_CORES)); \
  printf("\n"); \
  for (int i = 0; i < N_CORES; i++) printf("Core %d: Active Cycles = %lu\n",i,_active[i]/(REPEAT)); \
  for (int i = 1; i < N_CORES; i++) _active[0] += _active[i]; \
  printf("Total Active Cycles = %lu\n",_active[0]/(REPEAT)); \
  printf("Avg Active Cycles = %lu\n",_active[0]/(REPEAT*N_CORES)); \
  printf("\n"); \
  for (int i = 0; i < N_CORES; i++) printf("Core %d: EXT LD = %lu\n",i,_ldext[i]/(REPEAT)); \
  for (int i = 1; i < N_CORES; i++) _ldext[0] += _ldext[i]; \
  printf("Total EXT LD = %lu\n",_ldext[0]/(REPEAT)); \
  printf("Avg EXT LD = %lu\n",_ldext[0]/(REPEAT*N_CORES)); \
  printf("\n"); \
  for (int i = 0; i < N_CORES; i++) printf("Core %d: TCDM Conts = %lu\n",i,_tcdmcont[i]/(REPEAT)); \
  for (int i = 1; i < N_CORES; i++) _tcdmcont[0] += _tcdmcont[i]; \
  printf("Total TCDM Conts = %lu\n",_tcdmcont[0]/(REPEAT)); \
  printf("Avg TCDM Conts = %lu\n",_tcdmcont[0]/(REPEAT*N_CORES)); \
  printf("\n"); \
  for (int i = 0; i < N_CORES; i++) printf("Core %d: LD Stall = %lu\n",i,_ldstall[i]/(REPEAT)); \
  for (int i = 1; i < N_CORES; i++) _ldstall[0] += _ldstall[i]; \
  printf("Total LD Stall = %lu\n",_ldstall[0]/(REPEAT)); \
  printf("Avg LD Stall = %lu\n",_ldstall[0]/(REPEAT*N_CORES)); \
  printf("\n"); \
  for (int i = 0; i < N_CORES; i++) printf("Core %d: JR Stall = %lu\n",i,_jrstall[i]/(REPEAT)); \
  for (int i = 1; i < N_CORES; i++) _jrstall[0] += _jrstall[i]; \
  printf("Total JR Stall = %lu\n",_jrstall[0]/(REPEAT)); \
  printf("Avg JR Stall = %lu\n",_jrstall[0]/(REPEAT*N_CORES)); \
  printf("\n"); \
  for (int i = 0; i < N_CORES; i++) printf("Core %d: JUMP = %lu\n",i,_jump[i]/(REPEAT)); \
  for (int i = 1; i < N_CORES; i++) _jump[0] += _jump[i]; \
  printf("Total JUMP = %lu\n",_jump[0]/(REPEAT)); \
  printf("Avg JUMP = %lu\n",_jump[0]/(REPEAT*N_CORES)); \
  printf("\n"); \
  for (int i = 0; i < N_CORES; i++) printf("Core %d: ST = %lu\n",i,_st[i]/(REPEAT)); \
  for (int i = 1; i < N_CORES; i++) _st[0] += _st[i]; \
  printf("Total ST = %lu\n",_st[0]/(REPEAT)); \
  printf("Avg ST = %lu\n",_st[0]/(REPEAT*N_CORES)); \
  printf("\n"); \
  for (int i = 0; i < N_CORES; i++) printf("Core %d: LD = %lu\n",i,_ld[i]/(REPEAT)); \
  for (int i = 1; i < N_CORES; i++) _ld[0] += _ld[i]; \
  printf("Total LD = %lu\n",_ld[0]/(REPEAT)); \
  printf("Avg LD = %lu\n",_ld[0]/(REPEAT*N_CORES)); \
  printf("\n"); \
  for (int i = 0; i < N_CORES; i++) printf("Core %d: ST EXT = %lu\n",i,_st_ext[i]/(REPEAT)); \
  for (int i = 1; i < N_CORES; i++) _st_ext[0] += _st_ext[i]; \
  printf("Total ST EXT = %lu\n",_st_ext[0]/(REPEAT)); \
  printf("Avg ST EXT = %lu\n",_st_ext[0]/(REPEAT*N_CORES)); \
  printf("\n"); \
  for (int i = 0; i < N_CORES; i++) printf("Core %d: ST EXT Cycles = %lu\n",i,_st_ext_cyc[i]/(REPEAT)); \
  for (int i = 1; i < N_CORES; i++) _st_ext_cyc[0] += _st_ext_cyc[i]; \
  printf("Total ST EXT Cycles = %lu\n",_st_ext_cyc[0]/(REPEAT)); \
  printf("Avg ST EXT Cycles = %lu\n",_st_ext_cyc[0]/(REPEAT*N_CORES)); \
  printf("\n"); \
  for (int i = 0; i < N_CORES; i++) printf("Core %d: LD EXT Cycles = %lu\n",i,_ld_ext_cyc[i]/(REPEAT)); \
  for (int i = 1; i < N_CORES; i++) _ld_ext_cyc[0] += _ld_ext_cyc[i]; \
  printf("Total LD EXT Cycles = %lu\n",_ld_ext_cyc[0]/(REPEAT)); \
  printf("Avg LD EXT Cycles = %lu\n",_ld_ext_cyc[0]/(REPEAT*N_CORES)); \
  printf("\n"); \
  for (int i = 0; i < N_CORES; i++) printf("Core %d: BR-Taken = %lu\n",i,_btaken[i]/(REPEAT)); \
  for (int i = 1; i < N_CORES; i++) _btaken[0] += _btaken[i]; \
  printf("Total BR-Taken = %lu\n",_btaken[0]/(REPEAT)); \
  printf("Avg BR-Taken = %lu\n",_btaken[0]/(REPEAT*N_CORES)); \
  printf("\n"); \
  for (int i = 0; i < N_CORES; i++) printf("Core %d: Branch = %lu\n",i,_branch[i]/(REPEAT)); \
  for (int i = 1; i < N_CORES; i++) _branch[0] += _branch[i]; \
  printf("Total Branch = %lu\n",_branch[0]/(REPEAT)); \
  printf("Avg Branch = %lu\n",_branch[0]/(REPEAT*N_CORES)); \
  printf("\n"); \
  for (int i = 0; i < N_CORES; i++) printf("Core %d: RVC = %lu\n",i,_rvc[i]/(REPEAT)); \
  for (int i = 1; i < N_CORES; i++) _rvc[0] += _rvc[i]; \
  printf("Total RVC = %lu\n",_rvc[0]/(REPEAT)); \
  printf("Avg RVC = %lu\n",_rvc[0]/(REPEAT*N_CORES)); \
  printf("\n"); \
  for (int i = 0; i < N_CORES; i++) printf("Core %d: I-Miss = %lu\n",i,_imiss[i]/(REPEAT)); \
  for (int i = 1; i < N_CORES; i++) _imiss[0] += _imiss[i]; \
  printf("Total I-Miss = %lu\n",_imiss[0]/(REPEAT)); \
  printf("Avg I-Miss = %lu\n",_imiss[0]/(REPEAT*N_CORES)); 
  
#else  // ! STATS_CLUSTER

#define INIT_STATS_CLUSTER()
#define EXPORT_STATS_CLUSTER()
#define ENTER_LOOP_CLUSTER()
#define START_STATS_CLUSTER()
#define STOP_STATS_CLUSTER()
#define EXIT_LOOP_CLUSTER()
#define PRINT_STATS_CLUSTER()

#endif


#endif