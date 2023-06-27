#ifndef __EDEN_ENSEMBLE_DATA_H__
#define __EDEN_ENSEMBLE_DATA_H__
//OUTPUT
OUTPUT_LTYPE OUTPUT_CTYPE OUTPUT[OUTPUT_LEN] = {0};
//ROOTS
ROOTS_LTYPE ROOTS_CTYPE ROOTS[N_TREES] = {
    ${formatter.to_c_array(data.root_)}
};

//NODES
#if defined(EDEN_NODE_STRUCT)
NODES_LTYPE node_struct NODES[N_NODES] = {
    ${formatter.to_node_struct(data.root_, data.feature_, data.threshold_, data.children_right_)}
};
#elif defined(EDEN_NODE_ARRAY)
FEATURE_LTYPE FEATURE_CTYPE FEATURE[N_NODES] = {
    ${formatter.to_c_array(data.feature_)}
};
THRESHOLD_LTYPE THRESHOLD_CTYPE THRESHOLD[N_NODES] = {
    ${formatter.to_c_array(data.threshold_)}
};
CHILDREN_RIGHT_LTYPE CHILDREN_RIGHT_CTYPE CHILDREN_RIGHT[N_NODES] = {
    ${formatter.to_c_array(data.children_right_)}
};
#endif // defined(EDEN_NODE_STRUCT)

//LEAVES
#ifdef EDEN_LEAF_STORE_EXTERNAL
LEAF_LTYPE OUTPUT_CTYPE LEAVES[N_LEAVES][LEAF_LEN] = {
    ${formatter.to_c_array2d(data.leaf_value_)}
};
#endif
#endif //__EDEN_ENSEMBLE_DATA_H

