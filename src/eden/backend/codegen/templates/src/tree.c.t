<%def name="tree_arrays(config)">
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

<%def name="tree_struct(config)">
int current_idx = roots[t];
struct Node current_node = nodes[current_idx];
while (current_node.feature != LEAF_INDICATOR) {
        if (input[current_node.feature] <=
            current_node.alpha) { // False(0) -> Right, True(1) -> Left
          current_idx++;
        } else {
          current_idx += current_node.child_right;
        }
        current_node = nodes[current_idx];
}
</%def>