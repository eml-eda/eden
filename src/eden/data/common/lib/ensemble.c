/*
    Static inference function for any supported EDEN bit-width
*/



#include "eden.h"
#ifndef DYNAMIC_INFERENCE
void ensemble_inference(
    #if defined(EDEN_NODE_STRUCT)
    node_struct nodes[N_NODES],
    #elif defined(EDEN_NODE_ARRAY)
    CHILDREN_RIGHT_CTYPE children_right[N_NODES],
    THRESHOLD_CTYPE threshold[N_NODES],
    FEATURE_CTYPE feature[N_NODES],
    #endif // EDEN_NODE_STRUCT
    #if defined(EDEN_LEAF_STORE_EXTERNAL)
    OUTPUT_CTYPE leaves[N_LEAVES][LEAF_LEN],
    #endif // EDEN_LEAF_STORE_EXTERNAL
    ROOTS_CTYPE roots[N_TREES],
    INPUT_CTYPE input[INPUT_LEN],
    OUTPUT_CTYPE output[OUTPUT_LEN]
)
{
    #if defined(GAP8) && (N_CORES>1)
    int core_id = pi_core_id();
    #endif

    #if defined(EDEN_NODE_ARRAY)

    #elif defined(EDEN_NODE_STRUCT)
    node_struct *tree_prediction;
    #endif

    for (int t = 0; t < N_TREES; t ++) {
        #if defined(GAP8) && (N_CORES>1)
        if(core_id == (t%(N_CORES))) {
        #endif 
        // TREE PREDICTION
        #if defined(EDEN_NODE_ARRAY)
        #error NotImplemented

        #elif defined(EDEN_NODE_STRUCT)
        tree_prediction = tree_inference(input, nodes + roots[t]);
            // Extract the prediction
            #if defined(EDEN_LEAF_STORE_EXTERNAL)
            OUTPUT_CTYPE *leaf = leaves[tree_prediction->children_right];
            #elif defined(EDEN_LEAF_STORE_INTERNAL)
            OUTPUT_CTYPE leaf = tree_prediction->threshold;
            #endif // EDEN_LEAF_STORE_EXTERNAL
        #endif // EDEN_NODE_ARRAY/STRUCT
        accumulate(output, leaf);



        #if defined(GAP8) && (N_CORES>1) 
        } 
        #endif 
    }

    #if defined(GAP8) && (N_CORES>1)
    pi_cl_team_barrier(); // All cores terminate together
    #endif //CS_END 
}

#else
int GLOBAL_STOPPING_SCORE = 0;

void ensemble_inference(
    #if defined(EDEN_NODE_STRUCT)
    node_struct nodes[N_NODES],
    #elif defined(EDEN_NODE_ARRAY)
    CHILDREN_RIGHT_CTYPE children_right[N_NODES],
    THRESHOLD_CTYPE threshold[N_NODES],
    FEATURE_CTYPE feature[N_NODES],
    #endif // EDEN_NODE_STRUCT
    #if defined(EDEN_LEAF_STORE_EXTERNAL)
    OUTPUT_CTYPE leaves[N_LEAVES][LEAF_LEN],
    #endif // EDEN_LEAF_STORE_EXTERNAL
    ROOTS_CTYPE roots[N_TREES],
    INPUT_CTYPE input[INPUT_LEN],
    OUTPUT_CTYPE output[OUTPUT_LEN]
) {
  #if defined(GAP8) && (N_CORES>1)
  int core_id = pi_core_id();
  #endif
  #if defined(EDEN_NODE_ARRAY)

  #elif defined(EDEN_NODE_STRUCT)
  node_struct *tree_prediction;
  #endif

  int t = 0;
  for (int i = 0; i<N_ADAPTIVE_STEPS; i++) {
    for(int tb = 0; tb< N_TREES_BATCH; tb++ , t++) {
        #if defined(GAP8) && (N_CORES>1)
        if(core_id == (t%(N_CORES))) {
        #endif 
        // TREE PREDICTION
        #if defined(EDEN_NODE_ARRAY)
        #error NotImplemented

        #elif defined(EDEN_NODE_STRUCT)
        tree_prediction = tree_inference(input, nodes + roots[t]);
        // Extract the prediction
        #if defined(EDEN_LEAF_STORE_EXTERNAL)
        OUTPUT_CTYPE *leaf = leaves[tree_prediction->children_right];
        #elif defined(EDEN_LEAF_STORE_INTERNAL)
        OUTPUT_CTYPE leaf = tree_prediction->threshold;
        #endif // EDEN_LEAF_STORE_EXTERNAL
        #endif // EDEN_NODE_ARRAY/STRUCT
        accumulate(output,leaf);
        #if defined(GAP8) && (N_CORES>1)
        }
        #endif 
    } // TREE_BATCH END
    //SYNC
    #if defined(GAP8) && (N_CORES>1)
    pi_cl_team_barrier();
    #endif 
    // Policy computation - Only core 0
    #if defined(GAP8) && (N_CORES>1)
    if(core_id == 0) {
    #endif 
    GLOBAL_STOPPING_SCORE = compute_early_stopping_policy();
    #if defined(GAP8) && (N_CORES>1)
    }
    #endif 
    #if defined(GAP8) && (N_CORES>1)
    pi_cl_team_barrier();
    #endif // SYNC- WAIT FOR POLICY COMPUTATION
    if (GLOBAL_STOPPING_SCORE > ADAPTIVE_THRESHOLD) {
        break;
    }
  }
  // LEFTOVER - ADAPTIVE 
  #if (LEFTOVER_ADAPTIVE!=0)
  if(GLOBAL_STOPPING_SCORE <= ADAPTIVE_THRESHOLD) { // Compute trees only if adaptive was not triggered
    for (int t = t ; t<N_TREES; t++) {
        #if defined(GAP8) && (N_CORES>1)
        if(core_id == (t%(N_CORES))) {
        #endif 
        // TREE PREDICTION
        #if defined(EDEN_NODE_ARRAY)
        #error NotImplemented
        #elif defined(EDEN_NODE_STRUCT)
        tree_prediction = tree_inference(input, nodes + roots[t]);
            // Extract the prediction
            #if defined(EDEN_LEAF_STORE_EXTERNAL)
            OUTPUT_CTYPE *leaf = leaves[tree_prediction->children_right];
            #elif defined(EDEN_LEAF_STORE_INTERNAL)
            OUTPUT_CTYPE leaf = tree_prediction->threshold;
            #endif // EDEN_LEAF_STORE_EXTERNAL
        #endif // EDEN_NODE_ARRAY/STRUCT
        accumulate(output, leaf);
        #if defined(GAP8) && (N_CORES>1)
        }
        #endif 
    }
  }

  #endif //LEFTOVER_ADAPTIVE!=0
 #if defined(GAP8) && (N_CORES>1)
 pi_cl_team_barrier(); // All cores terminate together
 #endif //CS_END 
}

#endif