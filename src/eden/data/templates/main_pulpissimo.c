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
<%
    def dequantize_str(var_name, qparams):
      if qparams is None:
        return var_name
      s, z = qparams["S"], qparams["Z"]
      dequant = f"(({var_name} - {z})*{s})"
      return dequant
%>
#include "eden_cfg.h"
#include "eden_ensemble_data.h"
#include "eden_input.h"
#include "eden.h"
int main(int argc, char const *argv[])
{
    
    INIT_STATS();
    BEGIN_STATS_LOOP();
    for(int k=0; k<OUTPUT_LEN; k++) {
        OUTPUT[k] = 0;
    }
    START_STATS();

  ensemble_inference(
#if defined(EDEN_NODE_STRUCT)
                    NODES,
#elif defined(EDEN_NODE_ARRAY)
                    THRESHOLDS,
                    CHILDREN_RIGHT,
                    FEATURE_IDX,
#endif
#if defined(EDEN_LEAF_STORE_EXTERNAL)
                     LEAVES,
#endif
                    ROOTS,
                    INPUT,
                    OUTPUT
    );
	STOP_STATS();
    END_STATS_LOOP();
  #ifdef DEBUG
    //${output_qparams["S"]} ${output_qparams["Z"]}
    for(int k=0; k<OUTPUT_LEN; k++) {
        printf("Logit[%d]= %f\n", k, ${dequantize_str("OUTPUT[k]", output_qparams)});

    }
  #endif

	return 0;
}