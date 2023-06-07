/*
Type declaration needed for running eden,
 only this file should be modified
*/
#ifndef __EDEN_SETUP__H
#define __EDEN_SETUP__H

#if defined(PULPISSIMO)
#include "rt/rt_api.h"
#include "stats/pulpissimo-stats.h"
#elif defined(GAP8)
#include "pmsis.h"
#include "stats/gap8-stats.h"
#else
#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#endif



// Ensemble types
#define ROOTS_CTYPE ${roots_ctype} // Bits to store the roots
#define INPUT_CTYPE ${input_ctype} // Bits to store the input
#define THRESHOLD_CTYPE ${threshold_ctype} // Bits to store the thresholds
#define OUTPUT_CTYPE ${output_ctype}// Bits to store the output
#define CHILDREN_RIGHT_CTYPE ${children_right_ctype}// Bits to store the shift
#define FEATURE_CTYPE ${feature_ctype}// Bits to store the feature idx
#define CHILDREN_LEFT_CTYPE ${}//Currently unused

// Constants definition
#define N_ESTIMATORS ${n_estimators}
#define N_TREES ${n_trees}
#define LEAF_LEN ${leaf_len}
#define N_NODES ${n_nodes}
#define INPUT_LEN ${input_len}
#define N_LEAVES ${n_leaves}
#define OUTPUT_LEN ${output_len}
#define OUTPUT_BITS ${output_bits}

// Data storage flags
//TODO: Find a better way to implement this
// WARNING : Do not change this manually
%if leaves_store_mode=="internal":
//#define EDEN_LEAF_STORE_EXTERNAL
#define EDEN_LEAF_STORE_INTERNAL
%else:
#define EDEN_LEAF_STORE_EXTERNAL
//#define EDEN_LEAF_STORE_INTERNAL
%endif

// Both can be swapped manually
//#define EDEN_NODE_ARRAY
#define EDEN_NODE_STRUCT

// Memory location (L1/L2) - Only for GAP8, leave empty otherwise
#if defined(GAP8)
#define L1 PI_CL_L1
#else
#define L1 
#endif

#if defined(EDEN_NODE_ARRAY)
#define INPUT_LTYPE ${cache["EDEN_NODE_ARRAY"]["input_ltype"]}
#define THRESHOLD_LTYPE ${cache["EDEN_NODE_ARRAY"]["threshold_ltype"]} 
#define OUTPUT_LTYPE ${cache["EDEN_NODE_ARRAY"]["output_ltype"]}
#define ROOTS_LTYPE ${cache["EDEN_NODE_ARRAY"]["roots_ltype"]}
#define LEAF_LTYPE ${cache["EDEN_NODE_ARRAY"]["leaves_ltype"]}
#define NODES_LTYPE ${}
#define FEATURE_LTYPE ${cache["EDEN_NODE_ARRAY"]["feature_ltype"]}
#define CHILDREN_RIGHT_LTYPE ${cache["EDEN_NODE_ARRAY"]["children_right_ltype"]}
#elif defined(EDEN_NODE_STRUCT)
#define INPUT_LTYPE ${cache["EDEN_NODE_STRUCT"]["input_ltype"]}
#define OUTPUT_LTYPE ${cache["EDEN_NODE_STRUCT"]["output_ltype"]}
#define ROOTS_LTYPE ${cache["EDEN_NODE_STRUCT"]["roots_ltype"]}
#define LEAF_LTYPE ${cache["EDEN_NODE_STRUCT"]["leaves_ltype"]}
#define NODES_LTYPE ${cache["EDEN_NODE_STRUCT"]["nodes_ltype"]}
#define THRESHOLD_LTYPE ${} 
#define FEATURE_LTYPE ${}
#define CHILDREN_RIGHT_LTYPE ${}
#endif // EDEN_NODE_ARRAY
// SIMD
#if ((OUTPUT_BITS==8) && (defined(PULPISSIMO) || defined(GAP8)))
#define SIMD
#define SIMD_SHIFT 2
#define SIMD_CTYPE v4u
#define SIMD_LEFTOVER ((OUTPUT_LEN/SIMD_SHIFT)*SIMD_SHIFT)
#define ADD(x,y) ((v4u) __builtin_pulp_add4(((v4s)x),((v4s)y)))
#elif (OUTPUT_BITS==16) && (defined(PULPISSIMO) || defined(GAP8))
#define SIMD
#define SIMD_SHIFT 1
#define SIMD_CTYPE v2u
#define SIMD_LEFTOVER ((OUTPUT_LEN/SIMD_SHIFT)*SIMD_SHIFT)
#define ADD(x,y) ((v2u) __builtin_pulp_add2(((v2s)x),((v2s)y)))
#endif 

// Dynamic inference 
#define DYNAMIC_INFERENCE
#define BATCH 1// 
#define AGG_SCORE_MARGIN 
#define ADAPTIVE_THRESHOLD 128  // Stopping threshold

#define N_ADAPTIVE_STEPS (((N_ESTIMATORS + BATCH -1)/BATCH)-1)//((N_ESTIMATORS/BATCH)-1 + (N_ESTIMATORS%BATCH))
#if N_ESTIMATORS==N_TREES
#define N_TREES_BATCH (BATCH)
#define LEFTOVER_ADAPTIVE  (N_TREES - (N_ADAPTIVE_STEPS*BATCH))
#else // GBT
#define N_TREES_BATCH (BATCH*N_CLASSES)
#define LEFTOVER_ADAPTIVE  (N_TREES - (N_ADAPTIVE_STEPS*ESTIMATOR_BATCH))
#endif
#endif // __EDEN_SETUP__H