#ifndef __ENSEMBLE_H__
#define __ENSEMBLE_H__
%if config.target == "pulpissimo":
#include "rt/rt_api.h"
#include "stats.h"
#define L1 
%elif config.target == "gap8":
#include "pmsis.h"
#include "stats.h"
#define L1 PI_CL_L1
%else:
#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#define L1 
%endif


// Constants definition
#define LEAF_INDICATOR ${config.input_length}
#define N_ESTIMATORS ${config.n_estimators}
#define N_TREES ${config.n_trees}
#define LEAF_LENGTH ${config.leaf_length}
#define N_NODES ${config.n_nodes}
#define INPUT_LENGTH ${config.input_length}
#define N_LEAVES ${0 if config.leaves is None else config.leaves.shape[0]}
#define OUTPUT_LENGTH ${config.output_length}

// SIMD
%if config.target in ["gap8", "pulpissimo"]:

%if config.output_ctype == "uint16_t":
#define ADD(x,y) ((v2u) __builtin_pulp_add2(((v2u)x),((v2u)y)))
%elif config.output_ctype == "uint8_t":
#define ADD(x,y) ((v4u) __builtin_pulp_add4(((v4u)x),((v4u)y)))
%endif

%else:
typedef uint8_t v4u __attribute__ ((vector_size (4)));
typedef uint16_t v2u __attribute__ ((vector_size (4)));
#define ADD(x,y) ((x)+(y))
%endif

%if config.data_structure == "struct":
// Data structures
typedef struct NodeStruct
{
    ${config.feature_ctype} feature;
    ${config.alpha_ctype} alpha;
    ${config.child_right_ctype} children_right;
} node;
%endif

#endif //__ENSEMBLE_H__