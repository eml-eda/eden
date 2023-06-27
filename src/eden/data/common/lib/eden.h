#ifndef __EDEN__H
#define __EDEN__H

#include "eden_cfg.h"

// Denotes how leaf nodes can be found
#define EDEN_LEAF_INDICATOR (INPUT_LEN + 1)
#if defined(EDEN_NODE_STRUCT)
typedef struct Node
{
    /* data */
    FEATURE_CTYPE feature;
    THRESHOLD_CTYPE threshold;
    CHILDREN_RIGHT_CTYPE children_right;
} node_struct;
#elif defined(EDEN_NODE_ARRAY)

#endif //defined(EDEN_NODE_STRUCT)

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
);

// Function body here to force inlining
inline node_struct* __attribute__((always_inline)) tree_inference(
    INPUT_CTYPE input[INPUT_LEN],
    node_struct *nodes
)
{
    node_struct *current_node = nodes;
    while (current_node->feature != EDEN_LEAF_INDICATOR) {
        if (input[current_node->feature] <=
            current_node->threshold) { // False(0) -> Right, True(1) -> Left
          current_node++;
        } else {
          current_node += current_node->children_right;
        }
      }
    return current_node;
}

inline void __attribute__((always_inline)) accumulate(
    OUTPUT_CTYPE output[OUTPUT_LEN],
#if defined(EDEN_LEAF_STORE_EXTERNAL)
    OUTPUT_CTYPE *leaf
#elif defined(EDEN_LEAF_STORE_INTERNAL)
    OUTPUT_CTYPE leaf
#endif // EDEN_LEAF_STORE_EXTERNAL
)
{
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
  }

#ifdef DYNAMIC_INFERENCE
inline OUTPUT_CTYPE __attribute__((always_inline)) compute_early_stopping_metric(
  OUTPUT_CTYPE output[OUTPUT_LEN]
)
{
    int first_max, second_max;
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
    return first_max - second_max;
}

#endif //DYNAMIC_INFERENCE

#endif // __EDEN__H