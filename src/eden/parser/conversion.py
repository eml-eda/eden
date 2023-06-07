"""
Collection of functions to convert the flat tree dictionary in a hierarchical one, where 
each node is a sub-dictionary.

Notes
--------
Currently not implemented, still not convinced it is actually useful.
"""
raise NotImplementedError("This package is still a WIP.")
from typing import Mapping, List
import json
from jsonschema import RefResolver, Draft7Validator
import pkgutil
from eden.parser import validate_json
from copy import deepcopy

def convert_flow_to_flat(dict_flow: Mapping):
    pass



def convert_flat_to_flow(dict_flat: Mapping):
    # First validate it
    validate_json(dict_flat)
    # Deep copy to avoid any unexpected changes
    dict_flat = deepcopy(dict_flat)
    for idx, tree in enumerate(dict_flat["trees"]):
        flat_trees_list[idx] =  _to_recursive(tree)


    