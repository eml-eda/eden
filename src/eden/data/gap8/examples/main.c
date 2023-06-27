
/*
 * --------------------------------------------------------------------------
 *  Copyright (c) 2023 Politecnico di Torino, Italy
 *  SPDX-License-Identifier: Apache-2.0
 *
 *  Licensed under the Apache License, Version 2.0 (the "License");
 *  you may not use this file except in compliance with the License.
 *  You may obtain a copy of the License at
 *
 *  http://www.apache.org/licenses/LICENSE-2.0
 *
 *  Unless required by applicable law or agreed to in writing, software
 *  distributed under the License is distributed on an "AS IS" BASIS,
 *  WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 *  See the License for the specific language governing permissions and
 *  limitations under the License.
 *
 *  Author: Francesco Daghero francesco.daghero@polito.it
 * --------------------------------------------------------------------------
 */
#define STACK_SIZE 512

#include "eden.h"
#include "eden_cfg.h"
#include "eden_ensemble_data.h"
#include "eden_input.h"

INIT_STATS();

/* Task executed by cluster cores. */
void cluster_inference(void *arg) {
  ensemble_inference(
#if defined(EDEN_NODE_STRUCT)
      NODES,
#elif defined(EDEN_NODE_ARRAY)
      THRESHOLDS, CHILDREN_RIGHT, FEATURE_IDX,
#endif
#if defined(EDEN_LEAF_STORE_EXTERNAL)
      LEAVES,
#endif
      ROOTS, INPUT, OUTPUT);
}

void cluster_delegate(void *arg) {

  ENTER_LOOP_STATS();
  for (int k = 0; k < OUTPUT_LEN; k++) {
    OUTPUT[k] = 0;
  }
  START_STATS();
  pi_cl_team_fork(N_CORES, cluster_inference, ((void *)0));
  STOP_STATS();
  EXIT_LOOP_STATS();
#ifdef DEBUG
  for (int k = 0; k < OUTPUT_LEN; k++) {
    printf("Logit[%d]= %f\n", k, OUTPUT[k]/1.0);
  }
#endif
}

void fabric_controller(void) {

  int errors = 0;
  int core_id = pi_core_id(), cluster_id = pi_cluster_id();

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

  /* Prepare cluster task and send it to cluster. */
  struct pi_cluster_task cl_task = {0};
  pi_cluster_task(&cl_task, cluster_delegate, NULL);
  cl_task.stack_size = (uint32_t)STACK_SIZE;

  pi_cluster_send_task_to_cl(&cluster_dev, &cl_task);
  pi_cluster_close(&cluster_dev);

  pmsis_exit(errors);
}

/* Program Entry. */
int main(void) { return pmsis_kickoff((void *)fabric_controller); }