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


        #if defined(GAP8) && (N_CORES>1) //CS_START
        pi_cl_team_critical_enter();
        #endif 
        // Single element leaves - internal - (Regression, Multiclass - OVO)
        #if (LEAF_LEN == 1) && defined(EDEN_LEAF_STORE_EXTERNAL)
        output[t%OUTPUT_LEN] += leaf[0];
        // Single element leaves - internal - (Regression, Multiclass - OVO)
        #elif (LEAF_LEN == 1) && defined(EDEN_LEAF_STORE_INTERNAL)
        output[t%OUTPUT_LEN] += leaf;
        // Accumulation - Arrays
        // 8-bit vectorial
        #elif defined(SIMD)
        SIMD_CTYPE *output_vector = (SIMD_CTYPE*) output;
        SIMD_CTYPE *leaf_vector = (SIMD_CTYPE*) leaf;
        for(int p = 0; p<(OUTPUT_LEN>>SIMD_SHIFT) ; p++) {
           output_vector[p] = ADD(output_vector[p], leaf_vector[p]);
        }
        #if ((OUTPUT_LEN%SIMD_SHIFT)!=0)
        int leftover = (OUTPUT_LEN>>SIMD_SHIFT)>>SIMD_SHIFT;
        while (leftover<OUTPUT_LEN) {
            output[leftover] += leaf[leftover];
            leftover++;
        }
        #endif // OUTPUT_LEN
        #else //32-bit, no SIMD
        for(int p = 0; p<(OUTPUT_LEN); p++) {
            output[p] += leaf[p];
        }
        #endif  // Accumulation
        #if defined(GAP8) && (N_CORES>1)
        pi_cl_team_critical_exit();
        #endif //CS_END 
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
  int first_max, second_max;

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
        #if defined(GAP8) && (N_CORES>1) //CS_START
        pi_cl_team_critical_enter();
        #endif 
        // Single element leaves - internal - (Regression, Multiclass - OVO)
        #if (LEAF_LEN == 1) && defined(EDEN_LEAF_STORE_EXTERNAL)
        output[t%OUTPUT_LEN] += leaf[0];
        // Single element leaves - internal - (Regression, Multiclass - OVO)
        #elif (LEAF_LEN == 1) && defined(EDEN_LEAF_STORE_INTERNAL)
        output[t%OUTPUT_LEN] += leaf;
        // Accumulation - Arrays
        // 8-bit vectorial
        #elif defined(SIMD)
        SIMD_CTYPE *output_vector = (SIMD_CTYPE*) output;
        SIMD_CTYPE *leaf_vector = (SIMD_CTYPE*) leaf;
        for(int p = 0; p<(OUTPUT_LEN>>SIMD_SHIFT) ; p++) {
           output_vector[p] = ADD(output_vector[p], leaf_vector[p]);
        }
        #if ((OUTPUT_LEN%SIMD_SHIFT)!=0)
        int leftover = (OUTPUT_LEN>>SIMD_SHIFT)*SIMD_SHIFT;
        while (leftover<OUTPUT_LEN) {
            output[leftover] += leaf[leftover];
            leftover++;
        }
        #endif // OUTPUT_LEN
        #else //32-bit, no SIMD
        for(int p = 0; p<(OUTPUT_LEN); p++) {
            output[p] += leaf[p];
        }
        #endif  // Accumulation
        #if defined(GAP8) && (N_CORES>1)
        pi_cl_team_critical_exit();
        #endif //CS_END 
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
    #if defined(AGG_MAX_SCORE)
    second_max = 0;
    first_max = output[0];
    for(int pol =1; pol < OUTPUT_LEN; pol++) {
        if (first_max<output[pol]) first_max = output[pol];
    }
    #elif defined(AGG_SCORE_MARGIN)
    second_max = output[0] >= output[1] ? output[1] :output[0];
    first_max = output[0] >= output[1] ? output[0] : output[1];
    for(int pol =1; pol < OUTPUT_LEN; pol++) {
        if (first_max<output[pol]) {
            second_max = first_max;
            first_max = output[pol];
        }
        else if (second_max < output[pol]) {
            second_max = output[pol];
        }
    }
    #else 
    #error
    #endif
    GLOBAL_STOPPING_SCORE = first_max - second_max;
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

        #if defined(GAP8) && (N_CORES>1) //CS_START
        pi_cl_team_critical_enter();
        #endif 
        // Single element leaves - internal - (Regression, Multiclass - OVO)
        #if (LEAF_LEN == 1) && defined(EDEN_LEAF_STORE_EXTERNAL)
        output[t%OUTPUT_LEN] += leaf[0];
        // Single element leaves - internal - (Regression, Multiclass - OVO)
        #elif (LEAF_LEN == 1) && defined(EDEN_LEAF_STORE_INTERNAL)
        output[t%OUTPUT_LEN] += leaf;
        // Accumulation - Arrays
        // 8-bit vectorial
        #elif defined(SIMD)
        SIMD_CTYPE *output_vector = (SIMD_CTYPE*) output;
        SIMD_CTYPE *leaf_vector = (SIMD_CTYPE*) leaf;
        for(int p = 0; p<(OUTPUT_LEN>>SIMD_SHIFT) ; p++) {
           output_vector[p] = ADD(output_vector[p], leaf_vector[p]);
        }
        #if ((OUTPUT_LEN%SIMD_SHIFT)!=0)
        int leftover = (OUTPUT_LEN>>SIMD_SHIFT)>>SIMD_SHIFT;
        while (leftover<OUTPUT_LEN) {
            output[leftover] += leaf[leftover];
            leftover++;
        }
        #endif // OUTPUT_LEN
        #else //32-bit, no SIMD
        for(int p = 0; p<(OUTPUT_LEN); p++) {
            output[p] += leaf[p];
        }
        #endif  // Accumulation
        #if defined(GAP8) && (N_CORES>1)
        pi_cl_team_critical_exit();
        #endif //CS_END 

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