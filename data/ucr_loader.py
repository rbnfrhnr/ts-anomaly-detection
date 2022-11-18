import pathlib
from glob import glob

import math
import numpy as np
import pandas as pd
import torch
from sklearn.preprocessing import MinMaxScaler
from torch.utils.data import Dataset


def rel_normalize(data):
    diff = np.concatenate([[0], np.diff(data)])
    rel = diff / data
    return rel


class UCRDataset(Dataset):

    def __init__(self, transform=None, location=None, cache_location=None,
                 cache_name="ucr-dataset", compression="zip", set_number=138, t_steps=90, batch_transform=None, **cfg):
        super(UCRDataset, self).__init__()
        self.cfg = cfg
        self.location = location
        self.cache_location = cache_location
        self.cache_name = cache_name
        self.transform = transform
        self.set_number = set_number
        self.t_steps = t_steps
        self.train_data = None
        self.test_data = None
        self.use_cache = False
        self.batch_transform = batch_transform
        self.differentiate = cfg["differentiate"] if "differentiate" in cfg else False

        self.cache_name = self.cache_name if compression is None else self.cache_name + "." + compression
        cache_path = pathlib.Path(self.cache_location).joinpath(
            self.cache_name + ".csv") if self.cache_location is not None else pathlib.Path("foo")
        if cache_path.exists() and self.use_cache:
            self.train_data, self.test_data = self.load_from_cache()
        else:
            train_x, train_y, test_x, test_y = self.load_data()

            scaler = MinMaxScaler()
            scaler.fit(train_x.reshape(-1, train_x.shape[-1]))
            train_x = scaler.transform(train_x.reshape(-1, train_x.shape[-1])).reshape(train_x.shape)
            test_x = scaler.transform(test_x.reshape(-1, test_x.shape[-1])).reshape(test_x.shape)
            if self.batch_transform:
                n = train_x.shape[0]
                train_x_transformed = self.batch_transform(train_x)
                train_x = np.concatenate([train_x, train_x_transformed])
                train_y = np.repeat(train_y, train_x.shape[0] // n).reshape(-1, 1)

            self.train_data = (train_x, train_y)
            self.test_data = (test_x, test_y)

            # is_train = np.concatenate([np.ones((train_x.shape[0], 1)), np.zeros((test_x.shape[0], 1))],
            #                           axis=0)
            # data = np.concatenate([self.train_data, self.test_data], axis=0)
            # frame = pd.DataFrame(columns=["x" + str(i) for i in range(t_steps)] + ["y", "is_train"],
            #                      data=np.concatenate([data[:, :-1].reshape(-1, t_steps), data[:, -1].reshape(-1, 1),
            #                                           is_train.reshape(-1, 1)], axis=1))
            # cache_path.parent.absolute().mkdir(parents=True, exist_ok=True)
            # frame.to_csv(cache_path, compression=compression)

    def load_data(self):
        file = glob(self.location + '/' + str(self.set_number) + '*')[0]
        file_name = file.split('/')[-1]
        data = np.fromfile(file, sep='\n').reshape(-1, 1)
        dataset_name = file_name.split('.')[0]
        dataset_info = dataset_name.split('_')
        train_from = 0
        train_to = int(dataset_info[4])
        test_from = train_to
        anomaly_from = int(dataset_info[5])
        anomaly_to = int(dataset_info[6])
        anomaly_range = np.arange(anomaly_from, anomaly_to)

        window_size = self.t_steps

        train_data = data[:train_to]
        num_features = 1
        if self.differentiate:
            diff = np.concatenate([np.array([0]).reshape(-1, 1), np.diff(train_data, axis=0) / train_data[1:]])
            train_data = np.concatenate([train_data, diff], axis=1)
            num_features = 2

        train_data = train_data[:math.floor(train_data.shape[0] / window_size) * window_size]
        indexer = np.arange(num_features * window_size)[None, :] + num_features * np.arange(
            train_data.shape[0] - (window_size - 1))[:, None]
        train_data = train_data.reshape(-1)[indexer].reshape(-1, window_size, num_features)

        test_data = data[train_to:]
        if self.differentiate:
            diff = np.concatenate([np.array([0]).reshape(-1, 1), np.diff(test_data, axis=0) / test_data[1:]])
            test_data = np.concatenate([test_data, diff], axis=1)
        test_data = test_data[:math.floor(test_data.shape[0] / window_size) * window_size]

        indexer = np.arange(num_features * window_size)[None, :] + num_features * np.arange(
            test_data.shape[0] - (window_size - 1))[:, None]
        indexer_y = np.arange(window_size)[None, :] + np.arange(test_data.shape[0] - (window_size - 1))[:, None]

        test_data_idx = np.arange(test_data.shape[0])
        new_anomaly_from = anomaly_from - train_to
        new_anomaly_to = anomaly_to - train_to
        new_anomaly_range = np.arange(new_anomaly_from, new_anomaly_to)
        test_y = np.isin(test_data_idx, new_anomaly_range).astype(int)
        test_y = test_y.reshape(-1)[indexer_y].reshape(-1, window_size, 1)
        test_y = (test_y.sum(axis=1) > 0).astype(int)
        test_data = test_data.reshape(-1)[indexer].reshape(-1, window_size, num_features)

        return train_data, np.zeros((train_data.shape[0], 1)), test_data, test_y

    def load_from_cache(self):
        cache_path = pathlib.Path(self.cache_location).joinpath(self.cache_name + ".csv")
        data = pd.read_csv(cache_path)
        train_data = data[data["is_train"] == 1]
        test_data = data[data["is_train"] == 0]
        train_cols = ["x" + str(i) for i in range(data.columns.shape[0] - 2)]
        relevant_cols = train_cols + ["y"]
        return train_data[relevant_cols].values, test_data[relevant_cols].values

    def __len__(self):
        return len(self.train_data[0])

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()
        x, y = self.train_data[0][idx], self.train_data[1][idx]
        # x, y = self.train_data[idx][:-1].reshape(-1, 1), self.train_data[idx][-1].reshape(-1, 1)
        return x, y
        # return self.train_data[idx]
