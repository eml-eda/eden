<%namespace name="tree" file="tree.c.t"/>
<%namespace name="accumulate_struct" file="accumulate_struct.c.t"/>
<%namespace name="accumulate_arrays" file="accumulate_arrays.c.t"/>

<%def name="ensemble_arrays(config)">
void inference();

void ensemble_inference(
    ${config.child_right_ctype} children_right[N_NODES],
    ${config.alpha_ctype} alphas[N_NODES],
    ${config.feature_ctype} features[N_NODES],
    %if config.leaves is not None:
    ${config.output_ctype} leaves[N_LEAVES][LEAF_LENGTH],
    %endif 
    ${config.root_ctype} roots[N_TREES],
    ${config.input_ctype} input[INPUT_LENGTH],
    ${config.output_ctype} output[OUTPUT_LENGTH]
);


void ensemble_inference(
    ${config.child_right_ctype} children_right[N_NODES],
    ${config.alpha_ctype} alphas[N_NODES],
    ${config.feature_ctype} features[N_NODES],
    %if config.leaves is not None:
    ${config.output_ctype} leaves[N_LEAVES][LEAF_LENGTH],
    %endif 
    ${config.root_ctype} roots[N_TREES],
    ${config.input_ctype} input[INPUT_LENGTH],
    ${config.output_ctype} output[OUTPUT_LENGTH]
)
{
    %if config.target == "gap8":
    int core_id = = pi_core_id();
    %endif

    for (int t = 0; t < N_TREES; t ++) {
        %if config.target == "gap8":
        if(core_id == (t%(N_CORES))) {
        %endif

        ${tree.tree_arrays(config)}

        %if config.target == "gap8":
        pi_cl_team_critical_enter();
        %endif

        %if config.task == "classification_multiclass" and config.leaves is not None:
        ${accumulate_arrays.accumulate_classification_multiclass(config)}
        %elif config.task == "classification_multiclass_ovo":
        ${accumulate_arrays.accumulate_classification_ovo(config)}
        %elif config.task == "classification_label":
        ${accumulate_arrays.accumulate_classification_labels(config)}
        %elif config.task == "regression":
        ${accumulate_arrays.accumulate_regression(config)}
        %elif config.task == "classification_multiclass" and config.leaves is None:
        ${accumulate_arrays.accumulate_classification_binary(config)}
        %endif

        %if config.target == "gap8":
        pi_cl_team_critical_exit();
        %endif


        %if config.target == "gap8":
        } 
        %endif
    }

    %if config.target == "gap8":
    pi_cl_team_barrier(); // All cores terminate together
    %endif
}

// Inference function that calls ensemble_inference, depends on the parameters
void inference()
{
    ensemble_inference(
        CHILDREN_RIGHT,
        ALPHAS,
        FEATURES,
        %if config.leaves is not None:
        LEAVES,
        %endif 
        ROOTS,
        INPUT,
        OUTPUT
    );
}

</%def>

<%def name="ensemble_struct(config)">
void inference();

void ensemble_inference(
    struct Node nodes[N_NODES],
    %if config.leaves is not None:
    ${config.output_ctype} leaves[N_LEAVES][LEAF_LENGTH],
    %endif 
    ${config.root_ctype} roots[N_TREES],
    ${config.input_ctype} input[INPUT_LENGTH],
    ${config.output_ctype} output[OUTPUT_LENGTH]
);


void ensemble_inference(
    struct Node nodes[N_NODES],
    %if config.leaves is not None:
    ${config.output_ctype} leaves[N_LEAVES][LEAF_LENGTH],
    %endif 
    ${config.root_ctype} roots[N_TREES],
    ${config.input_ctype} input[INPUT_LENGTH],
    ${config.output_ctype} output[OUTPUT_LENGTH]
)
{
    %if config.target == "gap8":
    int core_id = = pi_core_id();
    %endif

    for (int t = 0; t < N_TREES; t ++) {
        %if config.target == "gap8":
        if(core_id == (t%(N_CORES))) {
        %endif

        ${tree.tree_struct(config)}

        %if config.target == "gap8":
        pi_cl_team_critical_enter();
        %endif

        %if config.task == "classification_multiclass" and config.leaves is not None:
        ${accumulate_struct.accumulate_classification_multiclass(config)}
        %elif config.task == "classification_multiclass_ovo":
        ${accumulate_struct.accumulate_classification_ovo(config)}
        %elif config.task == "classification_label":
        ${accumulate_struct.accumulate_classification_labels(config)}
        %elif config.task == "regression":
        ${accumulate_struct.accumulate_regression(config)}
        %elif config.task == "classification_multiclass" and config.leaves is None:
        ${accumulate_struct.accumulate_classification_binary(config)}
        %endif

        %if config.target == "gap8":
        pi_cl_team_critical_exit();
        %endif


        %if config.target == "gap8":
        } 
        %endif
    }

    %if config.target == "gap8":
    pi_cl_team_barrier(); // All cores terminate together
    %endif
}

// Inference function that calls ensemble_inference, depends on the parameters
void inference()
{
    ensemble_inference(
        NODES,
        %if config.leaves is not None:
        LEAVES,
        %endif 
        ROOTS,
        INPUT,
        OUTPUT
    );
}

</%def>