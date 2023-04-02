# **Eden**: **E**fficient **D**ecision tree **En**sembles
## Installation
To run the all experiments present in the library, install the packages listed in the requirements file  first. Then pip install the top level directory of this repository. This should enable to import the eden package.

## Running the code
Experiments are divided in different directories depending on the classifier used (Gradient Boosting or Random Forest) and on their purpose (grid search, inference profiling, adaptive benhmarks). Each dataset has its own specific python script.
An example:
```bash
python src/eden/gradient_boosting/grid_search/unimib.py
```
Runs the grid search for the UniMiB-SHAR dataset
The inference code can be generated automatically with the scripts under src/eden/{classifier}/inference/ with the pulpissimo toolchain sourced before run.