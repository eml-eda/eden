import onnx
from onnx import helper
from onnx import AttributeProto, TensorProto, GraphProto

from sklearn.datasets import load_iris, fetch_california_housing
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from eden.frontend.sklearn import parse_random_forest
from eden.export import ensemble_to_onnx

import onnxruntime as rt


def export_ensemble_to_onnx():
    iris = load_iris()
    model = RandomForestClassifier(n_estimators=3, random_state=0, max_depth=4)
    model.fit(iris.data, iris.target)

    result = parse_random_forest(model=model)
    sess = rt.InferenceSession("m3.onnx", providers=["CPUExecutionProvider"])
    input_name = sess.get_inputs()[0].name
    label_name = sess.get_outputs()[0].name
    pred_onx = sess.run([label_name], {input_name: iris.data.astype(np.float32)})[0]
    print(pred_onx)
