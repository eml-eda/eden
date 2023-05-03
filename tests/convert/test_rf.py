import pytest
import os
import eden
from ..fixtures.data import *
from ..fixtures.models import *
import numpy as np

EPS = 0.1


def setup_regression(
    n_features, signed_input, input_qbits, output_qbits, quantization_aware_training
):
    X, y = make_regression(n_samples=100, n_features=n_features, random_state=0)
    # Suppose that the input given makes sense for the QBITS
    if input_qbits is not None:
        X = np.clip(X, -(2 ** (input_qbits - 4)), 2 ** (input_qbits - 1) - 4)
    if not signed_input:
        X = np.abs(X)
    input_data_range = (X.min(), X.max())
    if output_qbits is not None:
        y = np.clip(y, -(2 ** (output_qbits - 2)), 2 ** (output_qbits - 1) - 2)

    if input_qbits is not None:
        if quantization_aware_training:
            X = eden.quantize(
                data=X, qbits=input_qbits, data_range=input_data_range, signed=False
            )
    return X, y, input_data_range


@pytest.mark.parametrize("target_architecture", ["any"])
@pytest.mark.parametrize("n_features", [256])
@pytest.mark.parametrize("n_estimators", [2, 4])
@pytest.mark.parametrize("signed_input", [True], ids=["signed_input=True"])
@pytest.mark.parametrize("quantization_aware_training", [False], ids=["NoQAT"])
@pytest.mark.parametrize(
    "input_qbits, output_qbits", [[8, 8], [16, 16], [32, 32], [None, None]]
)
@pytest.mark.parametrize("leaf_store_mode", ["auto"])
@pytest.mark.parametrize("ensemble_structure_mode", ["struct"])
@pytest.mark.parametrize("use_simd", ["auto"])
def test_regression_rf(
    signed_input,
    n_estimators,
    n_features,
    input_qbits,
    output_qbits,
    target_architecture,
    leaf_store_mode,
    ensemble_structure_mode,
    quantization_aware_training,
    use_simd,
):
    X, y, input_data_range = setup_regression(
        n_features=n_features,
        signed_input=signed_input,
        input_qbits=input_qbits,
        output_qbits=output_qbits,
        quantization_aware_training=quantization_aware_training,
    )
    # Get the first sample for test
    test_data = np.copy(X[0, :])
    # test_data = np.zeros_like(X[0, :])

    # MODEL TRAINING
    clf = RandomForestRegressor(max_depth=3, random_state=0, n_estimators=n_estimators)
    clf.fit(X, y)
    # MODEL TRAINING END

    os.system("rm -rf eden-ensemble")
    ensemble = eden.convert(
        model=clf,
        test_data=[test_data],
        input_qbits=input_qbits,
        output_qbits=output_qbits,
        input_data_range=input_data_range,
        quantization_aware_training=quantization_aware_training,
        target_architecture=target_architecture,
        leaf_store_mode=leaf_store_mode,
        ensemble_structure_mode=ensemble_structure_mode,
        use_simd=use_simd,
    )
    output = eden.run(ensemble=ensemble)
    golden = [t.predict(test_data.reshape(1, -1)) for t in clf.estimators_]
    # De-Quant block
    if ensemble.leaf_ctype != "float":
        for t in range(len(golden)):
            golden[t] = eden.quantize(
                data=golden[t],
                qbits=output_qbits,
                data_range=ensemble.output_data_range,
                signed=ensemble.output_data_range[0] < 0,
            )
    golden = np.asarray(golden)
    golden = golden.sum()
    if ensemble.leaf_ctype != "float":
        golden = eden.dequantize(data=golden, qparams=ensemble.leaf_qparams)
    assert (golden.item() - output) < EPS


def setup_classification(
    n_features,
    n_classes,
    signed_input,
    input_qbits,
    output_qbits,
    quantization_aware_training,
):
    X, y = make_classification(
        n_samples=100,
        n_features=n_features,
        random_state=0,
        n_classes=n_classes,
        n_informative=9,
    )
    # Suppose that the input given makes sense for the QBITS
    if input_qbits is not None:
        X = np.clip(X, -(2 ** (input_qbits - 4)), 2 ** (input_qbits - 1) - 4)
    if not signed_input:
        X = np.abs(X)
    input_data_range = (X.min(), X.max())

    if input_qbits is not None:
        if quantization_aware_training:
            X = eden.quantize(
                data=X, qbits=input_qbits, data_range=input_data_range, signed=False
            )
    return X, y, input_data_range


@pytest.mark.parametrize("target_architecture", ["any"])
@pytest.mark.parametrize("n_features", [256])
@pytest.mark.parametrize("n_estimators", [2, 4])
@pytest.mark.parametrize("signed_input", [True], ids=["signed_input=True"])
@pytest.mark.parametrize("quantization_aware_training", [False], ids=["NoQAT"])
@pytest.mark.parametrize(
    "input_qbits, output_qbits", [[8, 8], [16, 16], [32, 32], [None, None]]
)
@pytest.mark.parametrize("leaf_store_mode", ["auto"])
@pytest.mark.parametrize("ensemble_structure_mode", ["struct"])
@pytest.mark.parametrize("use_simd", ["auto"])
@pytest.mark.parametrize("n_classes", [2, 3, 4])
def test_classification_rf(
    signed_input,
    n_features,
    n_estimators,
    n_classes,
    input_qbits,
    output_qbits,
    target_architecture,
    leaf_store_mode,
    ensemble_structure_mode,
    quantization_aware_training,
    use_simd,
):
    X, y, input_data_range = setup_classification(
        n_features=n_features,
        n_classes=n_classes,
        signed_input=signed_input,
        input_qbits=input_qbits,
        output_qbits=output_qbits,
        quantization_aware_training=quantization_aware_training,
    )
    # Get the first sample for test
    test_data = np.zeros_like(X[0, :])

    # MODEL TRAINING
    clf = RandomForestClassifier(max_depth=3, random_state=0, n_estimators=n_estimators)
    clf.fit(X, y)
    # MODEL TRAINING END

    os.system("rm -rf eden-ensemble")
    ensemble = eden.convert(
        model=clf,
        test_data=[test_data],
        input_qbits=input_qbits,
        output_qbits=output_qbits,
        input_data_range=input_data_range,
        quantization_aware_training=quantization_aware_training,
        target_architecture=target_architecture,
        leaf_store_mode=leaf_store_mode,
        ensemble_structure_mode=ensemble_structure_mode,
        use_simd=use_simd,
    )
    output = eden.run(ensemble=ensemble)
    golden = [t.predict_proba(test_data.reshape(1, -1)) for t in clf.estimators_]
    if ensemble.leaf_ctype != "float":
        for t in range(len(golden)):
            golden[t] = eden.quantize(
                data=golden[t],
                qbits=output_qbits,
                data_range=ensemble.output_data_range,
                signed=ensemble.output_data_range[0] < 0,
            )
    golden = np.asarray(golden).sum(axis=0)
    if ensemble.leaf_ctype != "float":
        golden = eden.dequantize(data=golden, qparams=ensemble.leaf_qparams)
    golden = np.argmax(golden)
    assert golden.item() == output
