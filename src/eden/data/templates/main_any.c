<%
    def dequantize_str(var_name, qparams):
      if qparams is None:
        return var_name
      s, z = qparams["S"], qparams["Z"]
      dequant = f"(({var_name} - {z})*{s})"
      return dequant
%>
#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>

#include "eden_cfg.h"
#include "eden_ensemble_data.h"
#include "eden_input.h"
#include "eden.h"


int main() {

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

    #ifdef DEBUG
    //${output_qparams["S"]} ${output_qparams["Z"]}
    for(int k=0; k<OUTPUT_LEN; k++) {
        printf("Logit[%d]= %f\n", k, ${dequantize_str("OUTPUT[k]", output_qparams)});

    }
    #endif
    return 0;
}