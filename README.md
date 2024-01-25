<div align="left">
<img src=".assets/logo.png" width="300"/>
</div>

# **Eden**: **E**fficient **D**ecision tree **En**sembles
## Installation
Install this library as a package either by cloning locally the repository or directy from the github url.

## Running the code
Examples on how to quantize, export and benchmark adaptive models can be found in the examples/ folder.

## Supported ensembles
RandomForests, GradientBoosting and DecisionTrees should be easily exportable in C, while ExtraTrees and other flavours of BoostingEnsembles will be supported in future.
Eden supports both classifiers with logits in the leaves (see the DecisionTreeClassifier implementation of Sklearn) and with only class labels stored (see the examples/ folder).

## Supported architectures
- GAP8
- Pulpissimo
- x86 (no custom instructions)

## Citing 
If you use our code, please cite our paper:
```
@article{daghero2023dynamic,
  title={Dynamic Decision Tree Ensembles for Energy-Efficient Inference on IoT Edge Nodes},
  author={Daghero, Francesco and Burrello, Alessio and Macii, Enrico and Montuschi, Paolo and Poncino, Massimo and Pagliari, Daniele Jahier},
  journal={IEEE Internet of Things Journal},
  year={2023},
  publisher={IEEE}
}
```
