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
<% 
    def get_vectorial_type(ctype, simd):
        vectorial_type = None
        vectorial_div = 1
        if "int32" in ctype or not simd:
            return vectorial_type, vectorial_div
        elif "int16" in ctype:
            vectorial_type = "v2"
            vectorial_div = 2
        elif "int8" in ctype:
            vectorial_type = "v4"
            vectorial_div = 4
        if "u" in ctype:
            vectorial_type += "u"
        else:
            vectorial_type += "s"
        return vectorial_type, vectorial_div

    vectorial_type, vectorial_div = get_vectorial_type(config.leaf_ctype, config.use_simd)
%>

%if vectorial_type == "v4s":
#define ADD(x,y) ((v4s) __builtin_pulp_add4((x),(y)))
%elif vectorial_type == "v4u":
#define ADD(x,y) ((v4u) __builtin_pulp_add4(((v4s)x),((v4s)y)))
%elif vectorial_type == "v2s":
#define ADD(x,y) __builtin_pulp_add2(x,y)
%elif vectorial_type == "v2u":
#define ADD(x,y) ((v2u) __builtin_pulp_add2(((v2s)x),((v2s)y)))
%endif

void ensemble_inference(
    %if config.leaf_store_mode == "external":
    ${config.leaf_ctype} leaves[N_LEAVES][LEAF_SHAPE],
    %endif
    ${config.leaf_ctype} output[LEAF_SHAPE], 
    ${config.input_ctype} input[N_FEATURES],
    ${config.root_ctype} roots[N_NODES],
    %if config.ensemble_structure_mode == "struct":
    struct Node nodes[N_NODES]
    %elif config.ensemble_structure_mode == "array":
    ${config.threshold_ctype} thresholds[N_NODES],
    ${config.right_child_ctype} right_children[N_NODES],
    ${config.feature_idx_ctype} feature_idx[N_NODES]
    %endif

) {
    #if N_CORES>1
    int core_id = pi_core_id();
    #endif

    %if config.leaf_store_mode == "internal":
    ${config.leaf_ctype} leaf;
    %else:
    ${config.leaf_ctype} *leaf;
    %endif

    for (int t =0 ; t < N_TREES; t++) {
        // Check necessary only when cores > 1
        #if N_CORES>1
        if(core_id == (t%(N_CORES))) {
        #endif
        %if config.ensemble_structure_mode == "struct":
        <%include file="tree-struct.c"  args = "vectorial_type=vectorial_type, vectorial_div=vectorial_div"/>
        %elif config.ensemble_structure_mode == "array":
        <%include file="tree-array.c"  args = "vectorial_type=vectorial_type, vectorial_div=vectorial_div"/>
        %endif
        #if N_CORES>1
        }
        #endif
    }
}