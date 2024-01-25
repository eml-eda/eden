

<%def name="tree(config)">
int current_idx = roots[t];
int current_feature = features[current_idx];
while (current_feature != LEAF_INDICATOR) {
        if (input[current_feature] <=
            alphas[current_idx]) { // False(0) -> Right, True(1) -> Left
          current_idx++;
        } else {
          current_idx += children_right[current_idx];
        }
        current_feature = features[current_idx];
}
</%def>