/*
Tree inference, fully parallelizable
*/
#include "eden.h"

/*
inline node_struct* __attribute__((always_inline)) tree_inference(
    ROOTS_CTYPE tree_root, // Root node
    INPUT_CTYPE input[INPUT_LEN],
    node_struct nodes[N_NODES]
)
{
    node_struct *current_node = nodes + tree_root;
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
*/
//#elif defined(EDEN_ARRAY_NODES)

//#endif //EDEN_STRUCT_NODES