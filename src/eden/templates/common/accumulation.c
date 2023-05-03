
<%page args="vectorial_type, vectorial_div"/>
//ACCUMULATION_C_START
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
%if config.output_shape == 1:
    %if config.leaf_store_mode == "external":
        output[0]+= *leaf;
    %elif config.leaf_store_mode == "internal":
        output[0]+= leaf;
    %endif
%elif config.output_shape != 1 and config.leaf_shape == 1:
        output[t%OUTPUT_SHAPE] += leaf;
%elif vectorial_type is None or (config.output_shape/vectorial_div) < 1:
    for(int l=0; l< OUTPUT_SHAPE; l++) {
        output[l] += leaf[l];
    } 
%else:
    ${vectorial_type} *output_vector =(${vectorial_type} *) output;
    ${vectorial_type} *leaf_vector = (${vectorial_type} *) leaf;
    for(int l=0; l< (OUTPUT_SHAPE>>${int(vectorial_div/2)}); l++) {
        output_vector[l] = ADD(output_vector[l], leaf_vector[l]);
    }
    %if ((config.output_shape%vectorial_div) != 0):
    for(int l = ${int(config.output_shape/vectorial_div)}; l<LEAF_SHAPE; l++) {
        output[l] += leaf[l];
    }
    %endif
%endif
// ACCUMULATION_C_END