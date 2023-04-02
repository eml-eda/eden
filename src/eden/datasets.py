import numpy as np
import scipy.io as sp
import os
import pickle
from sklearn.model_selection import train_test_split
from imblearn import over_sampling
import logging
import pandas as pd
from pickle import load, dump


from eden.datasets_blackblaze import (
    feature_selection,
    import_data,
    dataset_partitioning,
    Y_target,
    filter_HDs_out,
)


def prepare_hdd(
    path="data/HDD_dataset/",
    # New args
    val_train_perc=0.1,
    bits_input=None,
    # Old args
    windowing=1,
    days_considered_as_failure=7,
    min_days_HDD=115,
    test_train_perc=0.3,
    num_features=18,
    history_signal=90,
    oversample_undersample=1,
    balancing_normal_failed=20,
    ranking="Ok",
    years=["2014", "2015", "2016", "2017", "2018"],
):
    def quantize(X, bits):
        eps = 1e-7
        assert X.min() >= (0 - eps), X.min()
        assert X.max() <= (1 + eps), X.max()
        X = np.clip(np.round(X * ((2**bits) - 1)), 0, 2**bits - 1)
        X = X - (2 ** (bits - 1))
        return X

    model = "ST3000DM001"
    years_str = "_".join(years)
    loaded = False
    fname = os.path.join(
        path,
        f"{model}-w{windowing}-hs{history_signal}_os{oversample_undersample}_vs{val_train_perc}.pkl",
    )
    if os.path.exists(fname):
        Xtrain, Xval, Xtest, ytrain, yval, ytest = load(open(fname, "rb"))
        loaded = True

    if loaded == False:
        # Funzioni di Alessio
        if os.path.exists(os.path.join(path, f"{model}-{years_str}.pkl")):
            df = pd.read_pickle(os.path.join(path, f"{model}-{years_str}.pkl"))
        else:
            df = import_data(path=path, years=years, model=model, name="iSTEP")
            df.to_pickle(os.path.join(path, f"{model}-{years_str}.pkl"))
        # Saving an intermediate step
        bad_missing_hds, bad_power_hds, df = filter_HDs_out(
            df, min_days=min_days_HDD, time_window="30D", tolerance=30
        )
        df["y"], df["val"] = Y_target(
            df, days=days_considered_as_failure, window=history_signal
        )
        if ranking != "None":
            df = feature_selection(df, num_features)

        Xtrain, Xval, Xtest, ytrain, yval, ytest = dataset_partitioning(
            df,
            model,
            overlap=1,
            rank=ranking,
            num_features=num_features,
            technique="random",
            test_train_perc=test_train_perc,
            windowing=windowing,
            window_dim=history_signal,
            resampler_balancing=balancing_normal_failed,
            oversample_undersample=oversample_undersample,
            val_train_perc=val_train_perc,
        )
        dump((Xtrain, Xval, Xtest, ytrain, yval, ytest), file=open(fname, "wb"))

    # New part
    # Quantization
    if bits_input is not None:
        Xtrain = quantize(Xtrain, bits=bits_input)
        if Xval is not None:
            Xval = quantize(Xval, bits=bits_input)
        Xtest = quantize(Xtest, bits=bits_input)
    Xtrain = Xtrain.reshape(Xtrain.shape[0], -1)
    if Xval is not None:
        Xval = Xval.reshape(Xval.shape[0], -1)
    Xtest = Xtest.reshape(Xtest.shape[0], -1)
    return Xtrain, Xval, Xtest, ytrain, yval, ytest, 2


def prepare_tuar(
    path="data/TUH/data",
    binary=True,
    balanced_train=True,
    bits_input=True,
    temporal=True,
    raw=True,
):
    def quantize_input_features(X, bits, range):
        # quantization
        # Balance the range
        delta = range / 2**bits
        X_round = np.round(X / delta)
        X = np.clip(X_round, -(2 ** (bits - 1)), (2 ** (bits - 1)) - 1)
        return X

    def quantize_input_raw(X, bits):
        X = X.astype(int)
        bits_to_shift = max(17 - bits, 0)
        X = np.right_shift(X, bits_to_shift)
        return X

    if not binary or not temporal:
        raise NotImplementedError

    if raw:
        X = np.load(os.path.join(path, "x_train_250_temporal_raw.npy"))
    else:
        X = np.load(os.path.join(path, "x_train_250_temporal.npy"))
    y = np.load(os.path.join(path, "y_train_binary_250_temporal.npy"))

    X, X_test, y, y_test = train_test_split(X, y, test_size=0.15, random_state=42)
    X_train, X_val, y_train, y_val = train_test_split(
        X, y, test_size=0.176, random_state=42
    )
    range = np.abs(X_train).max() * 2
    if bits_input is not None and not raw:
        X_train = quantize_input_features(X_train, bits=bits_input, range=range)
        X_val = quantize_input_features(X_val, bits=bits_input, range=range)
        X_test = quantize_input_features(X_test, bits=bits_input, range=range)
    elif bits_input is not None and raw:
        X_train = quantize_input_raw(X_train, bits=bits_input)
        X_val = quantize_input_raw(X_val, bits=bits_input)
        X_test = quantize_input_raw(X_test, bits=bits_input)

    # Augment
    if balanced_train:
        oversampler = over_sampling.RandomOverSampler(random_state=0)
        X_train, y_train = oversampler.fit_resample(X_train, y_train)

    num_classes = 2
    return X_train, X_val, X_test, y_train, y_val, y_test, num_classes


def check_if_ready(path, quantize_input, window_size, window_stride):
    data = None
    fname = f'{path}{"q_inp" if quantize_input else ""}'
    fname += "_wsize{window_size}_wstride{window_stride}.pkl"
    if os.path.isfile(fname):
        with open(fname, "rb") as f:
            data = pickle.load(f)
    return data


"""
ECG
"""
ECG_TEST_FILE = "ECG5000_TEST.txt"
ECG_TRAIN_FILE = "ECG5000_TRAIN.txt"


def prepare_ecg5000_dataset(
    bits_input,
    path="data/ecg5000/ECG5000",
    reversed_sets=False,
    balanced_train=True,
    binary=True,
):
    def quantize_input(X, bits):
        range = 8 - (-8)
        # quantization


# Infrared
thermo_train = [
    "006__11_44_59",
    "007__11_48_59",
    "008__11_52_59",
    "009__11_57_00",
    "000__14_15_19",
    "001__14_19_19",
    "002__14_23_19",
    "003__14_27_20",
    "004__14_31_20",
    "012__15_03_21",
    "013__15_07_21",
    "014__15_11_21",
    "015__15_15_21",
    "016__15_19_21",
    "011__13_38_20",
    "012__13_42_20",
    "013__13_46_21",
    "007__13_22_20",
]

thermo_val = [
    "004__13_10_20",
    "014__13_50_21",
    "005__14_35_20",
    "006__14_39_20",
    "007__14_43_20",
    "008__14_47_20",
]

thermo_test = [
    "008__13_26_20",
    "009__14_51_20",
    "010__14_55_20",
    "011__14_59_20",
    "015__13_54_21",
]
THERMO_PRESENCE_REPO = "https://github.com/PUTvision/thermo-presence.git"


def prepare_ir_dataset(
    path="data/thermo-presence", bits_inputs=None, balanced_train=True, binary=True
):
    def quantize_input(X, bits):
        # quantization
        # Measurement range [-40, 300], step 1 C
        # Balance the range
        delta = (300 * 2) / 2**bits
        X_round = np.round(X / delta)
        X = np.clip(X_round, -(2 ** (bits - 1)), (2 ** (bits - 1)) - 1)
        return X

    def process_thermo_set(fnames):
        X = np.zeros((0, 24, 32))
        y = list()
        for fname in fnames:
            fname = fname + ".h5"
            df = pd.read_hdf(os.path.join("data/thermo-presence/dataset/hdfs", fname))
            X_df = np.asarray(df["data"].to_list())  # [Bs, 23, 32]
            y_df = [len(t) for t in df["points"].to_list()]
            X = np.concatenate([X, X_df])
            y += y_df
        y = np.asarray(y)
        X = X.reshape(X.shape[0], -1)
        assert X.shape[0] == y.shape[0]
        return X, y

    if not os.path.exists(path):
        os.system(f"cd data && git clone -q {THERMO_PRESENCE_REPO}")
        # Train set

    X_train, y_train = process_thermo_set(thermo_train)
    # Val set
    X_val, y_val = process_thermo_set(thermo_val)
    # Test set
    X_test, y_test = process_thermo_set(thermo_test)
    if binary:
        y_train[y_train >= 1] = 1
        y_val[y_val >= 1] = 1
        y_test[y_test >= 1] = 1
    if balanced_train:
        logging.info(f"Dataset rebalancing: original-  {np.bincount(y_train)}")
        oversampler = over_sampling.RandomOverSampler(random_state=0)
        X_train, y_train = oversampler.fit_resample(X_train, y_train)
        logging.info(f"Dataset rebalancing: oversampled - {np.bincount(y_train)}")
    if bits_inputs is not None:
        X_train = quantize_input(X_train, bits=bits_inputs)
        X_val = quantize_input(X_val, bits=bits_inputs)
        X_test = quantize_input(X_test, bits=bits_inputs)
    n_classes = 2 if binary else 6
    return X_train, X_val, X_test, y_train, y_val, y_test, n_classes


if __name__ == "__main__":
    prepare_ir_dataset()

# UniMiB


UNIMIB_URL = "https://www.dropbox.com/s/x2fpfqj0bpf8ep6/UniMiB-SHAR.zip?dl=0"


def download_unimib_dataset(path, rank=0):
    if rank != 0:
        if path is None:
            path = "data/UniMiB-SHAR"
        return path
    if os.path.isdir("data/UniMiB-SHAR"):
        path = "data/UniMiB-SHAR"
    elif path is None or not os.path.isdir(path):
        print("Downloading  the dataset.")
        os.system(f"wget -q -O ./dataset.zip {UNIMIB_URL}")
        os.system("unzip -q dataset.zip")
        os.makedirs(path, exist_ok=True)
        os.system("mv UniMiB-SHAR/ data/")
        os.system("rm dataset.zip")
        path = "data/UniMiB-SHAR"
    return path


# Dataset preparation
def _prepare_unimib_dataset(path):
    file_folder = os.path.join(path, "data")
    data = sp.loadmat(
        os.path.join(file_folder, "split", "split151", "acc_splitted_data5_folds.mat"),
        squeeze_me=True,
    )["splitted_data"]
    idx_act_test = sp.loadmat(
        os.path.join(file_folder, "split", "split151", "acc_test_idxs5_folds.mat"),
        squeeze_me=True,
    )["test_idxs"]
    train_set = []
    test_set = []
    val_set = []
    y_train, y_test, y_val = [], [], []

    for i in range(data.shape[0]):
        test_idx = np.zeros(data[i].shape[0], dtype=bool)
        test_idx[idx_act_test[i][0] - 1] = 1
        val_idx = np.zeros(data[i].shape[0], dtype=bool)
        val_idx[idx_act_test[i][1] - 1] = 1
        assert np.sum(val_idx & test_idx) == 0, "Test and val set are overlapping"
        X_train_act = data[i][~(val_idx | test_idx)]
        X_val_act = data[i][val_idx]
        X_test_act = data[i][test_idx]
        train_set.append(X_train_act)
        val_set.append(X_val_act)
        test_set.append(X_test_act)
        y_train.append([i for _ in range(X_train_act.shape[0])])
        y_val.append([i for _ in range(X_val_act.shape[0])])
        y_test.append([i for _ in range(X_test_act.shape[0])])

    X_train = np.concatenate(train_set).reshape(-1, 3, 151).reshape(-1, 453)
    y_train = np.concatenate(y_train).reshape(-1)
    del train_set
    X_val = np.concatenate(val_set).reshape(-1, 3, 151).reshape(-1, 453)
    y_val = np.concatenate(y_val).reshape(-1)
    del val_set
    X_test = np.concatenate(test_set).reshape(-1, 3, 151).reshape(-1, 453)
    y_test = np.concatenate(y_test).reshape(-1)
    del test_set
    num_classes = np.unique(y_train).size
    return X_train, X_val, X_test, y_train, y_val, y_test, num_classes
    # return X_train,  X_test, y_train, y_test, num_classes


def prepare_unimib_dataset(path, bits_input, balanced_train=False):
    """
    17 classes unimib task
    """

    def quantize_input(X, bits):
        X = X / 9.81
        # return X
        # convert to [g]
        # quantization
        delta = (2 - (-2)) / 2**bits
        X_round = np.round(X / delta)
        X = np.clip(X_round, -(2 ** (bits - 1)), (2 ** (bits - 1)) - 1)
        return X

    # Additional channel when binarizing
    (
        X_train,
        X_val,
        X_test,
        y_train,
        y_val,
        y_test,
        num_classes,
    ) = _prepare_unimib_dataset(path)
    if balanced_train:
        logging.info(f"Dataset rebalancing: original-  {np.bincount(y_train)}")
        oversampler = over_sampling.RandomOverSampler(random_state=0)
        X_train, y_train = oversampler.fit_resample(X_train, y_train)
        logging.info(f"Dataset rebalancing: oversampled - {np.bincount(y_train)}")
    if bits_input is not None:
        X_train = quantize_input(X_train, bits=bits_input)
        X_val = quantize_input(X_val, bits=bits_input)
        X_test = quantize_input(X_test, bits=bits_input)
    return X_train, X_val, X_test, y_train, y_val, y_test, num_classes


def check_if_ready(path, quantize_input, window_size, window_stride):
    data = None
    fname = f'{path}{"q_inp" if quantize_input else ""}'
    fname += "_wsize{window_size}_wstride{window_stride}.pkl"
    if os.path.isfile(fname):
        with open(fname, "rb") as f:
            data = pickle.load(f)
    return data


"""
ECG
"""
ECG_TEST_FILE = "ECG5000_TEST.txt"
ECG_TRAIN_FILE = "ECG5000_TRAIN.txt"


def prepare_ecg5000_dataset(
    bits_input,
    path="data/ecg5000/ECG5000",
    reversed_sets=False,
    balanced_train=True,
    binary=True,
):
    def quantize_input(X, bits):
        range = 8 - (-8)
        # quantization
        delta = range / 2**bits
        X_round = np.round(X / delta).astype(np.int32).astype(np.float64)
        X = np.clip(X_round, -(2 ** (bits - 1)), (2 ** (bits - 1)) - 1)
        return X

    dataset_train = np.loadtxt(os.path.join(path, ECG_TRAIN_FILE))
    y_train = dataset_train[:, 0].astype(np.int) - 1
    if binary:
        y_train[y_train > 0] = 1  # Anomalie di qualsiasi tipo
    X_train = dataset_train[:, 1:]
    dataset_test = np.loadtxt(os.path.join(path, ECG_TEST_FILE))
    X_test = dataset_test[:, 1:]
    y_test = dataset_test[:, 0].astype(np.int) - 1
    if binary:
        y_test[y_test > 0] = 1
    if reversed_sets:
        X_train, X_test = X_test, X_train
        y_train, y_test = y_test, y_train

    X_train, X_val, y_train, y_val = train_test_split(
        X_train, y_train, stratify=y_train, random_state=0, test_size=0.2
    )
    # Oversampling
    if balanced_train:
        oversampler = over_sampling.RandomOverSampler(random_state=0)
        X_train, y_train = oversampler.fit_resample(X_train, y_train)

    if bits_input is not None:
        X_train = quantize_input(X_train, bits=bits_input)
        X_test = quantize_input(X_test, bits=bits_input)
        X_val = quantize_input(X_val, bits=bits_input)

    n_classes = 2 if binary else len(np.unique(y_train))
    del dataset_test
    del dataset_train
    return X_train, X_val, X_test, y_train, y_val, y_test, n_classes


"""
Ninapro
"""


def windowing(Y_instants, R_instants, X_instants):
    N = int(100 * 0.150)  # 150ms
    r = N // 2
    slide = int(100 * 0.04)  # 4ms, overlap ~75%
    M_instants, C = X_instants.shape
    M = (M_instants - N) // slide + 1
    Y_windows = Y_instants[-1 + N : M_instants : slide]
    del Y_instants
    R_windows = R_instants[r : M_instants - r : slide]
    del R_instants
    X_windows = np.zeros((M, N, C))
    for m in range(M):
        X_windows[m, :, :] = X_instants[m * slide : m * slide + N, :]
    return Y_windows, R_windows, X_windows


def test_and_train_and_val(Y, R, X):
    R[R == 0] = 1  # rerepetition=0
    is_window_train = (R != 2) & (R != 5) & (R != 7)  # 2 e 5 e 7 nel test set
    is_window_val = R == 4  # 1 validation set
    is_window_train &= ~is_window_val  # Rimuovo il val set dal training
    is_window_test = ~(is_window_train | is_window_val)
    XVal = X[is_window_val]
    YVal = Y[is_window_val]
    YTrain = Y[is_window_train]
    XTrain = X[is_window_train]
    YTest = Y[is_window_test]
    XTest = X[is_window_test]
    assert (
        is_window_train.sum() + is_window_val.sum() + is_window_test.sum()
    ) == X.shape[0]
    return YTrain, XTrain, YTest, XTest, YVal, XVal


def prepare_ninapro_dataset(
    bits_input, patient, path="data/emg_db1/extracted/", balanced_train=False
):
    def quantize_input(X, bits):
        # return X, None
        range = 10
        # quantization
        delta = range / 2**bits
        X = np.round(X / delta)
        X = np.clip(X, 0, (2**bits) - 1)
        X -= 2 ** (bits - 1)
        return X, range

    YTrain, XTrain, YTest, XTest, YVal, XVal = load_and_window_preprocess_and_split(
        path=path, s=patient
    )

    # 14 classes
    train_mask = np.where(np.logical_and(YTrain > 0, YTrain <= 14))
    test_mask = np.where(np.logical_and(YTest > 0, YTest <= 14))
    val_mask = np.where(np.logical_and(YVal > 0, YVal <= 14))
    XTrain, YTrain = XTrain[train_mask], YTrain[train_mask]
    XVal, YVal = XVal[val_mask], YVal[val_mask]
    XTest, YTest = XTest[test_mask], YTest[test_mask]

    XTrain = XTrain.transpose(0, 2, 1).reshape(-1, 150)
    XVal = XVal.transpose(0, 2, 1).reshape(-1, 150)
    XTest = XTest.transpose(0, 2, 1).reshape(-1, 150)

    if balanced_train:
        oversampler = over_sampling.RandomOverSampler(random_state=0)
        XTrain, YTrain = oversampler.fit_resample(XTrain, YTrain)

    # Quantize
    if bits_input is not None:
        XTrain, _ = quantize_input(XTrain, bits=bits_input)
        XTest, _ = quantize_input(XTest, bits=bits_input)
        XVal, _ = quantize_input(XVal, bits=bits_input)

    return (
        XTrain,
        XVal,
        XTest,
        YTrain - 1,
        YVal - 1,
        YTest - 1,
        14,
    )


def load_and_window_preprocess_and_split(path, s):
    def subsampling(Ytrain, Xtrain):
        Xtrain = Xtrain[::3, :, :]
        Ytrain = Ytrain[::3]
        return Ytrain, Xtrain

    data = sp.loadmat(path + f"s{s}/S{s}_A1_E1.mat")
    X, Y, R = data["emg"], data["restimulus"].squeeze(), data["rerepetition"].squeeze()
    data = sp.loadmat(path + f"s{s}/S{s}_A1_E2.mat")
    X2, Y2, R2 = (
        data["emg"],
        data["restimulus"].squeeze(),
        data["rerepetition"].squeeze(),
    )
    Y2[Y2 > 0] += 12
    data = sp.loadmat(path + f"s{s}/S{s}_A1_E3.mat")
    X3, Y3, R3 = (
        data["emg"],
        data["restimulus"].squeeze(),
        data["rerepetition"].squeeze(),
    )
    Y3[Y3 > 0] += 29
    X, Y, R = (
        np.concatenate((X, X2, X3), axis=0),
        np.concatenate((Y, Y2, Y3), axis=0),
        np.concatenate((R, R2, R3), axis=0),
    )
    del data, X2, Y2, R2
    # X = mean0std1(X, R)  # mean=0 std=1
    Y, R, X = windowing(Y, R, X)
    YTrain, XTrain, YTest, XTest, YVal, XVal = test_and_train_and_val(
        Y, R, X
    )  # test rep=2,5,7
    # print(XTrain.shape)
    del Y, R, X
    YTrain, XTrain = subsampling(
        YTrain, XTrain
    )  # subsample training set every 10 windows
    # print(XTrain.shape)
    return YTrain, XTrain, YTest, XTest, YVal, XVal


if __name__ == "__main__":
    prepare_ninapro_dataset(bits_input=None, patient=1)
