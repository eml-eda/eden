# **Eden**: **E**fficient **D**ecision tree **En**sembles
## Installation
Install this library as a package either by cloning locally the repository or directy from the github url.

## Running the code
The only step required is calling the eden.convert_to_eden() function, providing an already trained model from scikit-learn

```python3
from sklearn.ensemble import RandomForestClassifier
from sklearn.datasets import load_iris

X, y = load_iris(return_X_y=True)
model = RandomForestClassifier(max_depth=3, random_state=0, n_estimators=16)
model.fit(X, y)
convert_to_eden(
    estimator=model,
    quantization_aware_training=False,
    input_qbits=8,
    input_data_range=(X.min(), X.max()),
    output_qbits=8,
    leaves_store_mode="auto",
    ensemble_structure_mode="auto",
)


```
This code will generate a compilable folder containing the ensemble, together with a Makefile.
Refer to the docstring of eden.convert for further details on the parameters that can be selected.

## Supported ensembles
RandomForests, GradientBoosting and DecisionTrees have been tested and should be easily exportable, while ExtraTrees and other flavours of BoostingEnsembles will be supported in future (they may already work, however, they have not been tested).

## Supported architectures
For now, model can be deployed either on standard GCC (target_architecture="any") or on GAP8 (target_architecture="gap8").

## Paper version
The version of the library described in the paper "" can be found in the main branch

## Citing 
TBD