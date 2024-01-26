<%def name="accumulate_classification_multiclass(config)">
%if config.bits_output <  32:
${config.output_vtype} *output_vector = (${config.output_vtype}*) output;
%endif

%if config.bits_output < 32:
${config.output_vtype} *leaf_vector = (${config.output_vtype}*) (leaves[current_node.child_right]);
%endif

for(int m = 0; m < (OUTPUT_LENGTH>>${int(16//config.bits_output)}); m++) {
%if config.bits_output >= 32:
    output[m] += leaves[current_node.child_right][m];
%else:
    output_vector[m] = ADD(output_vector[m], leaf_vector[m]);
    leaf_vector+=4;
    output_vector+=4;
%endif
}

// Leftover
int leftover_start = ${int(config.output_length // (32//config.bits_output) * (32//config.bits_output))} ;
while (leftover_start < ${config.output_length}) {
    output[leftover_start] += leaves[current_node.child_right][leftover_start];
    leftover_start++;
}
</%def>

<%def name="accumulate_classification_ovo(config)">
output[t%N_CLASSES] += current_node.alpha;
</%def>

<%def name="accumulate_classification_binary(config)">
output[0] += current_node.alpha;
</%def>

<%def name="accumulate_regression(config)">
output[0] += current_node.alpha;
</%def>

<%def name="accumulate_classification_labels(config)">
output[current_node.child_right] += 1;
</%def>
