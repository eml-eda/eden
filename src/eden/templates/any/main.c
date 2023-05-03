<%
    def dequantize_str(var_name, qparams):
      if qparams is None:
        return var_name
      s, z = qparams["s"], qparams["z"]
      dequant = f"(({var_name} - {z})*{s})"
      return dequant
%>
#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>

#include "ensemble.h"
#include "ensemble_data.h"
#include "input.h"

<%include file="ensemble.c"/>

int main() {

  %if "classification" in config.task:
    int pred;
  %else:
    ${config.leaf_ctype} pred;
  %endif
  ensemble_inference(
%if config.leaf_store_mode == "external":
                     LEAVES,
%endif
                    OUTPUT,
                    INPUT,
                    ROOTS,
%if config.ensemble_structure_mode == "struct":
                    NODES
%elif config.ensemble_structure_mode == "array":
                    THRESHOLDS,
                    CHILDREN_RIGHT,
                    FEATURE_IDX
%endif
                  );

    <%include file="argmax.c"/>
    #ifdef DEBUG
    %if config.task=="regression":
    printf("INFERENCE OUTPUT:%f\n", ${dequantize_str("pred", config.leaf_qparams)});
    %else:
    printf("INFERENCE OUTPUT:%d\n", pred);
    %endif
    #endif
    return 0;
}