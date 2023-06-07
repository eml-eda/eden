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

