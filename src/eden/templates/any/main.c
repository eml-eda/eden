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
    printf("Ensemble inference output: ${"%d" if config.leaf_ctype!="float" or "classification" in config.task else "%f"}\n", pred);
    #endif
    return 0;
}