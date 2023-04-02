#ifndef __STATS_FPU_H__
#define __STATS_FPU_H__

#ifdef STATS_FPU

#define FPU_TYPE_CONF (0x11)
#define FPU_CONT_CONF (0x12)
#define FPU_DEP_CONF  (0x13)
#define FPU_WRB_CONF  (0x14)

#define FPU_TYPE_READ (0x791)
#define FPU_CONT_READ (0x12)
#define FPU_DEP_READ  (0x13)
#define FPU_WRB_READ  (0x14)

#define HOTTING 1
#define REPEAT  2
#define COUNTER 4

#define INIT_STATS_FPU()  \
    PI_CL_L1 unsigned long _apu_type[N_CORES] = {0}; \
    PI_CL_L1 unsigned long _apu_cont[N_CORES]  = {0}; \
    PI_CL_L1 unsigned long _apu_dep[N_CORES]   = {0}; \
    PI_CL_L1 unsigned long _apu_wb[N_CORES]    = {0};     

#define ENTER_LOOP_FPU()  \
  for(int _k = 0; _k < HOTTING + COUNTER*REPEAT; _k++) \
  { \
    if (_k >= HOTTING) \
    { \
      if((_k - HOTTING) % COUNTER == 0) pi_perf_conf((1<<FPU_TYPE_CONF)); \
      if((_k - HOTTING) % COUNTER == 1) pi_perf_conf((1<<FPU_CONT_CONF)); \
      if((_k - HOTTING) % COUNTER == 2) pi_perf_conf((1<<FPU_DEP_CONF)); \
      if((_k - HOTTING) % COUNTER == 3) pi_perf_conf((1<<FPU_WRB_CONF)); \
    }

#define START_STATS_FPU()  \
    if (_k >= HOTTING) \
    { \
      pi_perf_reset(); \
      pi_perf_start(); \
    }

#define STOP_STATS_FPU() \
    pi_cl_team_barrier(); \
    if (_k >= HOTTING) \
    { \
      pi_perf_stop(); \
      if((_k - HOTTING) % COUNTER == 0) _apu_type[core_id] += pi_perf_read(FPU_TYPE_READ); \
      if((_k - HOTTING) % COUNTER == 1) _apu_cont[core_id] += pi_perf_read(FPU_CONT_READ); \
      if((_k - HOTTING) % COUNTER == 2) _apu_dep[core_id]  += pi_perf_read(FPU_DEP_READ); \
      if((_k - HOTTING) % COUNTER == 3) _apu_wb[core_id]   += pi_perf_read(FPU_WRB_READ); \
    }

#define EXIT_LOOP_FPU()\
  }

#define PRINT_STATS_FPU()  \
  printf("FPU Statistics\n"); \
  for (int i = 0; i < N_CORES; i++) printf("Core %d: APU Type = %lu\n",i,_apu_type[i]/(REPEAT*N_LOOP)); \
  for (int i = 1; i < N_CORES; i++) _apu_type[0] += _apu_type[i]; \
  printf("Total APU Type = %lu\n",_apu_type[0]/(REPEAT*N_LOOP)); \
  printf("Avg   APU Type = %lu\n",_apu_type[0]/(REPEAT*N_LOOP*N_CORES)); \
  printf("\n"); \
  for (int i = 0; i < N_CORES; i++) printf("Core %d: APU Conts = %lu\n",i,_apu_cont[i]/(REPEAT*N_LOOP)); \
  for (int i = 1; i < N_CORES; i++) _apu_cont[0] += _apu_cont[i]; \
  printf("Total APU Conts = %lu\n",_apu_cont[0]/(REPEAT*N_LOOP)); \
  printf("Avg   APU Conts = %lu\n",_apu_cont[0]/(REPEAT*N_LOOP*N_CORES)); \
  printf("\n"); \
  for (int i = 0; i < N_CORES; i++) printf("Core %d: APU Dep = %lu\n",i,_apu_dep[i]/(REPEAT*N_LOOP)); \
  for (int i = 1; i < N_CORES; i++) _apu_dep[0] += _apu_dep[i]; \
  printf("Total APU Dep = %lu\n",_apu_dep[0]/(REPEAT*N_LOOP)); \
  printf("Avg   APU Dep = %lu\n",_apu_dep[0]/(REPEAT*N_LOOP*N_CORES)); \
  printf("\n"); \
  for (int i = 0; i < N_CORES; i++) printf("Core %d: APU WB = %lu\n",i,_apu_wb[i]/(REPEAT*N_LOOP)); \
  for (int i = 1; i < N_CORES; i++) _apu_wb[0] += _apu_wb[i]; \
  printf("Total APU WB = %lu\n",_apu_wb[0]/(REPEAT*N_LOOP)); \
  printf("Avg   APU WB = %lu\n",_apu_wb[0]/(REPEAT*N_LOOP*N_CORES));

  
#else  // ! STATS


#define INIT_STATS_FPU()
#define ENTER_LOOP_FPU()
#define START_STATS_FPU()
#define STOP_STATS_FPU()
#define EXIT_LOOP_FPU()
#define PRINT_STATS_FPU()

#endif


#endif