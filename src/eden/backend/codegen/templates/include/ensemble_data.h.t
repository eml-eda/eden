#ifndef __ENSEMBLE_DATA_H__
#define __ENSEMBLE_DATA_H__
${config.buffer_allocation["output"]} ${config.output_ctype} OUTPUT[OUTPUT_LENGTH] = {0};

${config.buffer_allocation["roots"]} ${config.root_ctype} ROOTS[N_TREES] = {
    ${config.roots_str}
};

%if config.data_structure == "struct":
${config.buffer_allocation["nodes"]} struct Node NODES[N_NODES] = {
    ${config.nodes_str}
};
%elif config.data_structure == "arrays":

${config.buffer_allocation["features"]} ${config.feature_ctype} FEATURES[N_NODES] = {
    ${config.features_str}
};
${config.buffer_allocation["alphas"]} ${config.alpha_ctype} ALPHAS[N_NODES] = {
    ${config.alphas_str}
};
${config.buffer_allocation["children_right"]} ${config.child_right_ctype} CHILDREN_RIGHT[N_NODES] = {
    ${config.children_right_str}
};

%endif

%if config.leaves is not None:
${config.buffer_allocation["leaves"]} ${config.output_ctype} LEAVES[N_LEAVES][LEAF_LENGTH] = {
    ${config.leaves_str}
};
%endif

#endif //__ENSEMBLE_DATA_H