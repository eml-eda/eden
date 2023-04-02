/*
Gradient Boosting Decision Tree Classifier
Versione Statica e dinamica senza buffer
*/

#include "ensemble.h"
#if defined(DYNAMIC)

void ensemble_inference(logit_dtype output[N_CLASSES],
                        feature_dtype input[N_FEATURES],
                        struct Node nodes[N_NODES],
                        node_idx_dtype roots[N_TREES]) {
  int core_id = pi_core_id();
  logit_dtype tree_prediction;
  int massimo, secondo_massimo;
  int current_tree = 0;
  int leftover = LEFT_TREES;

  // Iterazioni prima di fermarsi
  for (int i = 0; i<ITER; i++) {
    for (int t = 0; t < ESTIMATOR_BATCH; t++) {
      if (core_id == (current_tree % N_CORES)) {
        tree_prediction = tree_predict(input, nodes + roots[current_tree]);
        pi_cl_team_critical_enter();
        output[current_tree % N_CLASSES] += tree_prediction;
        pi_cl_team_critical_exit();
      }
      current_tree += 1;
    }
    // Tutti i cores hanno terminato le inferenze
    pi_cl_team_barrier();
    if (core_id == 0) {
// Early stop
//  Compute the score_margin
#if defined(SCORE_MARGIN)
      SM_SCORE()
      early_stop = (massimo - secondo_massimo) > THRESHOLD;
#elif defined(MAX_MARGIN)
      MAX_SCORE()
      early_stop = massimo > THRESHOLD;
#elif defined(QWYC)
      early_stop = (output[0]>THRESHOLD_HIGH) || (output[0]<THRESHOLD_LOW);
#else
#error No early stopping defined, options (SCORE_MARGIN, MAX_MARGIN)
#endif
    }

    pi_cl_team_barrier();
  }
  #if LEFT_TREES>0
  pi_cl_team_barrier();
  //current_tree = N_TREES - LEFT_TREES;
  if (early_stop == 0 || PROFILING>0) {
    for (int t = 0; t < LEFT_TREES; t++) {
      if (core_id == (current_tree % N_CORES)) {
        tree_prediction = tree_predict(input, nodes + roots[current_tree]);
        pi_cl_team_critical_enter();
        output[current_tree % N_CLASSES] += tree_prediction;
        pi_cl_team_critical_exit();
      }
      current_tree += 1;
    }
  }
  pi_cl_team_barrier();
  #endif
  ARGMAX()

}

#else //! DYNAMIC
void ensemble_inference(logit_dtype output[N_CLASSES],
                        feature_dtype input[N_FEATURES],
                        struct Node nodes[N_NODES],
                        node_idx_dtype roots[N_TREES]) {

  int core_id = pi_core_id();
  int leftover = N_TREES - N_TREES % N_CORES;
  logit_dtype tree_prediction;

#ifdef DEBUGGING
  int executions = 0;
#endif

  for (int i = 0; i < N_TREES; i++) {
    if (core_id == (i % N_CORES)) {
      tree_prediction = tree_predict(input, nodes + roots[i]);
      pi_cl_team_critical_enter();
      output[i % N_CLASSES] += tree_prediction;
      pi_cl_team_critical_exit();
// Wait for all cores to perform at least an inference
//  If it is core 0 we can now update the buffer, otherwise we wait for core 0
#ifdef DEBUGGING
      executions++;
#endif
    }
  }
#ifdef DEBUGGING
  printf("Core %d - Waiting - End %d\n", core_id, executions);
#endif
  pi_cl_team_barrier();
  ARGMAX()
}
#endif // DYNAMIC
