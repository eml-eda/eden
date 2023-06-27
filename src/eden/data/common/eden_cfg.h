/*
Type declaration needed for running eden,
 only this file should be modified
*/
#ifndef __EDEN_SETUP__H
#define __EDEN_SETUP__H

#if defined(PULPISSIMO)
#include "rt/rt_api.h"
#include "stats.h"
#elif defined(GAP8)
#include "pmsis.h"
#include "stats.h"
#else
#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#endif



// Ensemble types
#define ROOTS_CTYPE ${data.root_ctype_} // Bits to store the roots
#define INPUT_CTYPE ${data.input_ctype_} // Bits to store the input
#define THRESHOLD_CTYPE ${data.threshold_ctype_} // Bits to store the thresholds
#define OUTPUT_CTYPE ${data.output_ctype_}// Bits to store the output
#define CHILDREN_RIGHT_CTYPE ${data.children_right_ctype_}// Bits to store the shift
#define FEATURE_CTYPE ${data.feature_ctype_}// Bits to store the feature idx
#define CHILDREN_LEFT_CTYPE ${}//Currently unused

// Constants definition
#define N_ESTIMATORS ${data.n_estimators_}
#define N_TREES ${data.n_trees_}
#define LEAF_LEN ${data.leaf_len_}
#define N_NODES ${data.n_nodes_}
#define INPUT_LEN ${data.input_len_}
#define N_LEAVES ${data.n_leaves_}
#define OUTPUT_LEN ${data.output_len_}
#define OUTPUT_BITS ${data.output_bits_}

// Data storage flags
//TODO: Find a better way to implement this
// WARNING : Do not change this manually
%if data.c_leaf_data_store_=="internal":
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
#define INPUT_LTYPE ${data.input_ltype_}
#define THRESHOLD_LTYPE ${data.threshold_ltype_} 
#define OUTPUT_LTYPE ${data.output_ltype_}
#define ROOTS_LTYPE ${data.root_ltype_}
#define LEAF_LTYPE ${data.leaf_ltype_}
#define NODES_LTYPE ${}
#define FEATURE_LTYPE ${data.feature_ltype_}
#define CHILDREN_RIGHT_LTYPE ${data.children_right_ltype_}
#elif defined(EDEN_NODE_STRUCT)
#define INPUT_LTYPE ${data.input_ltype_}
#define OUTPUT_LTYPE ${data.output_ltype_}
#define ROOTS_LTYPE ${data.root_ltype_}
#define LEAF_LTYPE ${data.leaf_ltype_}
#define NODES_LTYPE ${data.node_ltype_}
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

// Dynamic inference - ONLY for classification, do not enable for regression
//#define DYNAMIC_INFERENCE
#define BATCH 1// 
#define AGG_MAX_SCORE //AGG_SCORE_MARGIN  // AGG_SM only for >2 classes 
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