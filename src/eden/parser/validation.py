# *--------------------------------------------------------------------------*
# * Copyright (c) 2023 Politecnico di Torino, Italy                          *
# * SPDX-License-Identifier: Apache-2.0                                      *
# *                                                                          *
# * Licensed under the Apache License, Version 2.0 (the "License");          *
# * you may not use this file except in compliance with the License.         *
# * You may obtain a copy of the License at                                  *
# *                                                                          *
# * http://www.apache.org/licenses/LICENSE-2.0                               *
# *                                                                          *
# * Unless required by applicable law or agreed to in writing, software      *
# * distributed under the License is distributed on an "AS IS" BASIS,        *
# * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. *
# * See the License for the specific language governing permissions and      *
# * limitations under the License.                                           *
# *                                                                          *
# * Author: Francesco Daghero francesco.daghero@polito.it                    *
# *--------------------------------------------------------------------------*

from typing import Mapping
import json
from jsonschema import RefResolver, Draft7Validator
import pkgutil


def validate_json(dictionary: Mapping) -> None:
    """
    Validates the input dictionary against the JSON schema in data/schema.

    Parameters
    ----------
    dictionary : Mapping
        Dictionary of an ensemble or a single tree.

    Notes
    ---------
    Only the fields type and presence are currently checked.
    """
    TREE_SCHEMA = json.loads(pkgutil.get_data("eden", "data/schema/tree.json"))
    ENSEMBLE_SCHEMA = json.loads(pkgutil.get_data("eden", "data/schema/ensemble.json"))
    if "trees" not in dictionary:
        validator = Draft7Validator(TREE_SCHEMA)
    else:
        schema_store = {
            TREE_SCHEMA["$id"]: TREE_SCHEMA,
        }
        resolver = RefResolver.from_schema(ENSEMBLE_SCHEMA, store=schema_store)
        validator = Draft7Validator(ENSEMBLE_SCHEMA, resolver=resolver)
    validator.validate(dictionary)

