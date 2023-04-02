#include <stdint.h>
#include <stdio.h>
/* PMSIS includes */
#include "pmsis.h"


/*Sets the data types*/
#include "ensemble_config.h"
/* GBT includes */
#include "ensemble.h"
/* Data include */
#include "ensemble_data.h"
#include "input.h"

#include "stats.h"
#include "stats_cluster.h"
#include "stats_fpu.h"

#define STACK_SIZE 512

INIT_STATS();
//INIT_STATS_FPU();
//INIT_STATS_CLUSTER();

/* Task executed by cluster cores. */
void cluster_inference(void *arg) {



  //ENTER_LOOP_CLUSTER();
  //START_STATS_CLUSTER();

  //ENTER_LOOP_FPU();
  //START_STATS_FPU();
#if (N_TREES==N_ESTIMATORS) && (N_CLASSES>1)
  ensemble_inference(OUTPUT, INPUT, NODES, ROOTS, LEAVES);
#else
  ensemble_inference(OUTPUT, INPUT, NODES, ROOTS);
#endif

  //STOP_STATS_FPU();
  //EXIT_LOOP_FPU();

  //STOP_STATS_CLUSTER();
  //EXIT_LOOP_CLUSTER();
}

/* Cluster main entry, executed by core 0. */
void cluster_delegate(void *arg) {
  printf("Cluster master core entry\n");

  ENTER_LOOP_STATS();
  #ifdef DYNAMIC
  early_stop = 0;
  #endif
  for (int i = 0; i < N_CLASSES; i++) {
    OUTPUT[i] = 0;
  }
  START_STATS();
  pi_cl_team_fork(N_CORES, cluster_inference, ((void *)0));
  STOP_STATS();
  EXIT_LOOP_STATS();

  //PRINT_STATS_FPU();
  //PRINT_STATS_CLUSTER();

  printf("Cluster master core exit\n");
#ifdef DEBUGGING
  int current_tree = 0;
  int error = 0;
  for (int i = 0; i < N_CLASSES; i++) {
    if (scores[i] != GOLDEN[i]) {
      printf("Score %d is different: Golden: %d Got: %d\n", i, GOLDEN[i],
             scores[i]);
      error = 1;
    }
  }
  if (error == 1)
    printf("Test failed\n");
  else
    printf("Test success\n");

#endif
}

void fabric_controller(void) {

  int errors = 0;
  int core_id = pi_core_id(), cluster_id = pi_cluster_id();
  printf("[%d %d] Starting RF2STRUCT PARALLEL!\n", cluster_id, core_id);

  struct pi_device cluster_dev = {0};
  struct pi_cluster_conf cl_conf = {0};

  /* Init cluster configuration structure. */
  pi_cluster_conf_init(&cl_conf);

  /* Set cluster ID. */
  cl_conf.id = 0;

  /* Configure & open cluster. */
  pi_open_from_conf(&cluster_dev, &cl_conf);

  if (pi_cluster_open(&cluster_dev)) {
    printf("Cluster open failed !\n");
    pmsis_exit(-1);
  }

  /* Allocating DMA Double-Buffering buffers in L1 */
  // dmaTestBuffer1 = (float *) pi_cl_l1_malloc(&cluster_dev, (uint32_t)
  // (4*DIM));

  // testBufferArr[0] = dmaTestBuffer1;

  // for (int i = 0; i < DIM; i++)
  //{
  //     dmaTestBuffer1[i] = 0;
  // }

  /* Prepare cluster task and send it to cluster. */
  struct pi_cluster_task cl_task = {0};
  pi_cluster_task(&cl_task, cluster_delegate, NULL);
  cl_task.stack_size = (uint32_t)STACK_SIZE;

  pi_cluster_send_task_to_cl(&cluster_dev, &cl_task);
  pi_cluster_close(&cluster_dev);

  // printf("Test success !\n");
  pmsis_exit(errors);
}

/* Program Entry. */
int main(void) {
  printf("\n\n\t *** PMSIS RF2STRUCT PARALLEL ***\n\n");
  return pmsis_kickoff((void *)fabric_controller);
}