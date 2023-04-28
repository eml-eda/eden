#ifndef __ENSEMBLE_DATA_H__
#define __ENSEMBLE_DATA_H__

${config.memory_map["ROOTS"]} ${config.root_ctype} ROOTS[N_TREES]={
    ${formatter.to_c_array(config.ROOTS)}
};

%if config.ensemble_structure_mode == "struct":
${config.memory_map["NODES"]} struct Node NODES[N_NODES]= {
    ${formatter.to_node_struct(config.FEATURE_IDX, config.THRESHOLDS, config.RIGHT_CHILDREN)}
};
%elif config.ensemble_structure_mode == "array":
${config.memory_map["FEATURE_IDX"]} ${config.feature_idx_ctype} FEATURE_IDX[N_NODES] = { 
    ${formatter.to_c_array(config.FEATURE_IDX)}
};
${config.memory_map["THRESHOLDS"]} ${config.threshold_ctype} THRESHOLDS[N_NODES] = { 
    ${formatter.to_c_array(config.THRESHOLDS)}
};
${config.memory_map["RIGHT_CHILDREN"]} ${config.right_child_ctype} RIGHT_CHILDREN[N_NODES] = { 
    ${formatter.to_c_array(config.RIGHT_CHILDREN)}
};
%endif

%if config.leaf_store_mode == "external":
${config.memory_map["LEAVES"]} ${config.leaf_ctype} LEAVES[N_LEAVES][LEAF_SHAPE] = {
    ${formatter.to_c_array2d(config.LEAVES)}
};
%endif

//${config.leaf_qtype}
${config.memory_map["OUTPUT"]} ${config.leaf_ctype} OUTPUT[OUTPUT_SHAPE] ={0};
#endif //__ENSEMBLE_DATA_H__