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

#endif // __EDEN__H