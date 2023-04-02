#ifndef __ENSEMBLE_H__
#define __ENSEMBLE_H__

#include "pmsis.h"
#include "ensemble_config.h"

#ifdef DYNAMIC
int early_stop;
int exit_tree;

#include "dynamic_scores.h"
#include "ensemble_dynamic_config.h"
#endif // DYNAMIC

int pred;
#if N_CLASSES == 1
#define ARGMAX() \
    pred=output[0]>=0;
#else
#define ARGMAX() \
    pred = 0; \
    logit_dtype pred_v = output[0]; \
    for(int lo=1; lo<N_CLASSES; lo++) { \
        if(pred_v<output[lo]) { \
            pred = lo; \
            pred_v = output[lo]; \
        } \
    } 
#endif

threshold_dtype tree_predict(feature_dtype *, struct Node *);
#if (N_TREES==N_ESTIMATORS) && (N_CLASSES>1)
#include "rf_accumulate.h"
void ensemble_inference(logit_dtype *, feature_dtype *, struct Node *,
                        node_idx_dtype *, leaves_dtype *);
#else
void ensemble_inference(logit_dtype *, feature_dtype *, struct Node *,
                        node_idx_dtype *);
#endif

#endif //__ENSEMBLE_H__