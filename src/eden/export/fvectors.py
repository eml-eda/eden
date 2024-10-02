from bigtree import BinaryNode, print_tree, preorder_iter
from typing import List
from eden.utils import _compute_nptype
import numpy as np
from eden.model.ensemble import Ensemble
from eden.model.node import Node
from bigtree import levelordergroup_iter, levelorder_iter, get_subtree


   
def extract_fvectors_ensemble(estimator, c):
    # The estimator should be perfect
    names, thresholds, indices, outputs = [], [], [], [] 
    d = estimator.max_depth 
    ds = int((d)/(c))
    ns = sum(2**(i*c) for i in range(ds))

    for idx,tree in enumerate(estimator.flat_trees):
        nam, th, ind, out = extract_fvectors_tree(tree, c, ds, ns)
        names.append(nam)
        thresholds.append(th)
        indices.append(ind)
        outputs.append(out)
    
    return names, thresholds, indices, outputs



def extract_fvectors_tree(tree, c : int, ds:int, ns : int):
    
    thresholds = np.ones((ns, 2**c-1)) 
    thresholds *= -100
    indices = np.zeros((ns, 2**c-1))
    outputs = np.zeros((ns * (2**(c)), tree.values.shape[-1]))
    names = np.zeros((ns, 2**c-1), dtype=np.uint32)

    stack = [(tree, 0)]
    subtree_idx = 0
    ncurrent = 0
    dscounter = 0
    ds_counters =np.cumsum(np.asarray([ 2**(i*c)  for i in range(ds)]))
    ds_counter = {k: ds_counters[k] for k in range(ds)}
    while stack:
        t, ds_lvl = stack.pop()

        starting_depth = t.depth
        end_depth = starting_depth + c
        for idx, node in enumerate(levelorder_iter(t, max_depth=end_depth)):
            if node.is_leaf:
                # This is the final level
                names[subtree_idx, idx] = int(node.name)
                thresholds[subtree_idx, idx] = node.alpha
                indices[subtree_idx, idx] = node.feature

                outputs[ncurrent] = node.values.reshape(-1)
                ncurrent+=1
            elif (node.depth == end_depth):
                outputs[ncurrent] = (ds_counter[ds_lvl]) 
                ds_counter[ds_lvl]+=1
                
                ncurrent+=1
                stack.insert(0, (node, ds_lvl+1))
            else:
                names[subtree_idx, idx] = int(node.name)
                thresholds[subtree_idx, idx] = node.alpha
                indices[subtree_idx, idx] = node.feature
        subtree_idx+=1
    for n, o in zip(names, outputs.reshape(-1, 4)):
        print(n, o.astype(np.int8))
    return names, thresholds, indices, outputs



