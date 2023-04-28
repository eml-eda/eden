<%doc>
/*
 * --------------------------------------------------------------------------
 *  Copyright (c) 2023 Politecnico di Torino, Italy
 *  SPDX-License-Identifier: Apache-2.0
 * 
 *  Licensed under the Apache License, Version 2.0 (the "License");
 *  you may not use this file except in compliance with the License.
 *  You may obtain a copy of the License at
 *  
 *  http://www.apache.org/licenses/LICENSE-2.0
 *  
 *  Unless required by applicable law or agreed to in writing, software
 *  distributed under the License is distributed on an "AS IS" BASIS,
 *  WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 *  See the License for the specific language governing permissions and
 *  limitations under the License.
 * 
 *  Author: Francesco Daghero francesco.daghero@polito.it
 * --------------------------------------------------------------------------
 */
</%doc>
<%page args="vectorial_type, vectorial_div"/>
// TREE_C_START
struct Node *current_node = nodes+roots[t];
while (current_node->feature_idx != -2) {
    if (input[current_node->feature_idx] <=
        current_node->threshold) { // False(0) -> Right, True(1) -> Left
      current_node++;
    } else {
      current_node += current_node->right_child;
    }
  }
% if config.leaf_store_mode == "external":
  leaf = LEAVES[current_node->right_child];
% elif config.leaf_store_mode == "internal": 
  leaf = current_node->threshold;
% endif
#if N_CORES>1
    pi_cl_team_critical_enter();
#endif
  // Accumulation, depends on the leaf data type
<%include file="accumulation.c"  args = "vectorial_type=vectorial_type, vectorial_div=vectorial_div"/>

#if N_CORES>1
    pi_cl_team_critical_exit();
#endif
// TREE_C_END