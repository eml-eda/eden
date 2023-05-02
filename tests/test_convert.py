import pytest
import eden
from .fixtures.data import *
from .fixtures.models import *
import numpy as np

EPS = 0.1


def setup_regression(
    n_features, signed_input, input_qbits, output_qbits, quantization_aware_training
):
    X, y = make_regression(n_samples=100, n_features=n_features, random_state=0)
    # Suppose that the input given makes sense for the QBITS
    input_qtype = "Q32.0"
    if input_qbits is not None:
        X = np.clip(X, -(2 ** (input_qbits - 4)), 2 ** (input_qbits - 1) - 4)
    if not signed_input:
        X = np.abs(X)
    input_data_range = (X.min(), X.max())
    if output_qbits is not None:
        y = np.clip(y, -(2 ** (output_qbits - 2)), 2 ** (output_qbits - 1) - 2)

    if input_qbits is not None:
        input_qtype = eden.suggest_qtype(bits=input_qbits, data_range=input_data_range)
        if quantization_aware_training:
            X = eden.quantize(data=X, qtype=input_qtype)
    return X, y, input_data_range, input_qtype


@pytest.mark.parametrize("target_architecture", ["any"])
@pytest.mark.parametrize("n_features", [128, 256])
@pytest.mark.parametrize("signed_input", [True, False])
@pytest.mark.parametrize("quantization_aware_training", [True, False])
@pytest.mark.parametrize("input_qbits", [8, 16, 32, None])
@pytest.mark.parametrize("output_qbits", [8, 16, 32, None])
@pytest.mark.parametrize("leaf_store_mode", ["internal", "auto"])
@pytest.mark.parametrize("ensemble_structure_mode", ["struct"])
@pytest.mark.parametrize("use_simd", ["auto"])
def test_regression_tree(
    signed_input,
    n_features,
    input_qbits,
    output_qbits,
    target_architecture,
    leaf_store_mode,
    ensemble_structure_mode,
    quantization_aware_training,
    use_simd,
):
    X, y, input_data_range, input_qtype = setup_regression(
        n_features=n_features,
        signed_input=signed_input,
        input_qbits=input_qbits,
        output_qbits=output_qbits,
        quantization_aware_training=quantization_aware_training,
    )
    # Get the first sample for test
    # test_data = np.copy(X[0, :])
    test_data = np.zeros_like(X[0, :])

    # MODEL TRAINING
    clf = DecisionTreeRegressor(max_depth=3, random_state=0)
    clf.fit(X, y)
    # MODEL TRAINING END
    golden = clf.predict(test_data.reshape(1, -1))

    if input_qbits is not None:
        test_data = eden.quantize(data=test_data, qtype=input_qtype)

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
    if output_qbits is not None:
        golden = eden.quantize(data=golden, qtype=ensemble.leaf_qtype)
        golden = eden.dequantize(data=golden, qtype=ensemble.leaf_qtype)
    assert (golden.item() - output) < EPS
