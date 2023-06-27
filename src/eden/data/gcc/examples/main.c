#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>

#include "eden.h"
#include "eden_cfg.h"
#include "eden_ensemble_data.h"
#include "eden_input.h"

int main() {

  ensemble_inference(
#if defined(EDEN_NODE_STRUCT)
      NODES,
#elif defined(EDEN_NODE_ARRAY)
      THRESHOLDS, CHILDREN_RIGHT, FEATURE_IDX,
#endif
#if defined(EDEN_LEAF_STORE_EXTERNAL)
      LEAVES,
#endif
      ROOTS, INPUT, OUTPUT);

#ifdef DEBUG
  for (int k = 0; k < OUTPUT_LEN; k++) {
    printf("Logit[%d]= %f\n", k, OUTPUT[k]/1.0);
  }
#endif
  return 0;
}