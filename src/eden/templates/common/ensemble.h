#ifndef __ENSEMBLE_H__
#define __ENSEMBLE_H__

// STORE MODE ${config.ensemble_structure_mode}
%if  config.ensemble_structure_mode == "struct":
struct Node {
    ${config.feature_idx_ctype} feature_idx;
    // QTYPE : ${config.threshold_qtype}
    ${config.threshold_ctype} threshold;
    ${config.right_child_ctype} right_child;
};
%endif

// Constants
#define N_ESTIMATORS ${config.n_estimators}
#define N_TREES ${config.n_trees}
#define LEAF_SHAPE ${config.leaf_shape}
#define N_NODES ${config.n_nodes}
#define N_FEATURES ${config.n_features}
#define N_LEAVES ${config.n_leaves}
#define OUTPUT_SHAPE ${config.output_shape}

#endif //__ENSEMBLE_H__