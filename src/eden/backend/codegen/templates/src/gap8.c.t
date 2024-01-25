<%def name="main(config)">
void cluster_delegate(void *arg) {

  ENTER_LOOP_STATS();
  for (int k = 0; k < OUTPUT_LEN; k++) {
    OUTPUT[k] = 0;
  }
  START_STATS();
  pi_cl_team_fork(N_CORES, inference, ((void *)0));
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
</%def>