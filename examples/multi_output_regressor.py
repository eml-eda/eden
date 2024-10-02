# *--------------------------------------------------------------------------*
# * Copyright (c) 2024 Politecnico di Torino, Italy                          *
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

from sklearn.multioutput import MultiOutputRegressor
from lightgbm import LGBMRegressor
from sklearn.datasets import make_regression
from eden.frontend.lightgbm import parse_boosting_trees
from eden.model import Ensemble
from eden.backend.deployment import deploy_model
from sklearn.metrics import mean_absolute_error


def generate_multioutput_regressor():
    """
    Generates a multi-output regressor for testing purposes and the data.
    """
    X, y = make_regression(n_samples=100, n_features=4, n_targets=2, random_state=0)
    fitted_mo = MultiOutputRegressor(
        LGBMRegressor(random_state=0, max_depth=2, n_estimators=2)
    ).fit(X, y)
    return X, y, fitted_mo


def main():
    X, y, fitted_mo = generate_multioutput_regressor()
    for idx, model in enumerate(fitted_mo.estimators_):
        golden = model.predict(X)
        emodel: Ensemble = parse_boosting_trees(model=model)
        eden_preds = emodel.predict(X)
        print("Sklearn-Eden prediction error", mean_absolute_error(golden, eden_preds))
        print("Sklearn MAE", mean_absolute_error(golden, y[:, idx]))
        print("Eden MAE", mean_absolute_error(eden_preds, y[:, idx]))

        # Deployment step
        subsampled_X = X[:10]
        deploy_model(
            ensemble=emodel,
            target="default",
            output_path="generated_model_{0}".format(idx),
            input_data=subsampled_X,
            data_structure="arrays",
        )
        print(
            "Expected outputs",
            {
                f"Golden-Sklearn-Eden{i}": (y[i, idx], golden[i], eden_preds[i])
                for i in range(subsampled_X.shape[0])
            },
        )


if __name__ == "__main__":
    main()
