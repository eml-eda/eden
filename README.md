# **Eden**: **E**fficient **D**ecision tree **En**sembles
## Installation
Install this library as a package either by cloning locally the repository or directy from the github url.

## Running the code
The only step required is calling the eden.convert() function, providing an already trained model from scikit-learn

```python3
from sklearn.datasets import make_classification
from sklearn.ensemble import RandomForestClassifier

X, y = make_classification( random_state=0, n_classes=2, n_informative=10,
n_features=256)
clf = RandomForestClassifier(max_depth=3, random_state=0, n_estimators=2)
clf= clf.fit(X, y)
test_data = np.zeros(shape=(1, X.shape[1]))
ensemble_summary = convert(
    model=clf,
    test_data=[test_data],
    input_qbits=32,
    output_qbits=32,
    input_data_range=(np.min(X), np.max(X)),
    quantization_aware_training=False,
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