#ifndef __EDEN_ENSEMBLE_DATA_H__
#define __EDEN_ENSEMBLE_DATA_H__
//OUTPUT
OUTPUT_LTYPE OUTPUT_CTYPE OUTPUT[OUTPUT_LEN] = {0};
//ROOTS
ROOTS_LTYPE ROOTS_CTYPE ROOTS[N_TREES] = {
    ${roots_string}
};

//NODES
#if defined(EDEN_NODE_STRUCT)
NODES_LTYPE node_struct NODES[N_NODES] = {
    ${node_string}
};
#elif defined(EDEN_NODE_ARRAY)
FEATURE_LTYPE FEATURE_CTYPE FEATURE[N_NODES] = {
    ${feature_string}
};
THRESHOLD_LTYPE THRESHOLD_CTYPE THRESHOLD[N_NODES] = {
    ${threshold_string}
};
CHILDREN_RIGHT_LTYPE CHILDREN_RIGHT_CTYPE CHILDREN_RIGHT[N_NODES] = {
    ${children_right_string}
};
#endif // defined(EDEN_NODE_STRUCT)

//LEAVES
#ifdef EDEN_LEAF_STORE_EXTERNAL
LEAF_LTYPE OUTPUT_CTYPE LEAVES[N_LEAVES][LEAF_LEN] = {
    ${leaf_string}
};
#endif
#endif //__EDEN_ENSEMBLE_DATA_H

