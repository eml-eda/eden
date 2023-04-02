from sklearn.model_selection import train_test_split
import numpy as np


def deployment_subset(classifier, X_data, n_samples_out=500):
    n_samples = X_data.shape[0]
    X_data = np.copy(X_data)
    if n_samples > n_samples_out:
        complexities = (
            classifier.predict_complexity(X_data).reshape(-1, X_data.shape[0]).mean(0)
        )
        unique, counts = np.unique(complexities, return_counts=True)
        single_repeat = unique[counts == 1]
        if len(single_repeat) > 0:
            print(f"Excluding {len(single_repeat)} elements")
            # Exclude the unique elements
            multiple_repeat_idx = np.where(
                np.isin(complexities, single_repeat, invert=True)
            )[0]
            X_data = X_data[multiple_repeat_idx]
            complexities = complexities[multiple_repeat_idx]

        n_samples_wanted = min(n_samples_out, X_data.shape[0])

        # Numero minimo di samples
        if n_samples_out < len(np.unique(complexities)):
            complexities = np.digitize(
                complexities,
                bins=np.histogram_bin_edges(complexities, bins=n_samples_out),
            )
        n_samples_wanted = len(np.unique(complexities))
        X_sub, _, y_sub, _ = train_test_split(
            X_data,
            complexities,
            stratify=complexities,
            random_state=0,
            train_size=n_samples_wanted,
        )
        return X_sub
    else:
        return X_data
