from datetime import datetime
from itertools import product

import math
import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from torch.utils.data import Dataset

from utils.common import ctu_files


class CTUDataset(Dataset):

    def __init__(self, transform=None, location=None, cache_location=None,
                 cache_name="ctu-", compression="zip", set_number=6, period_len=60, t_steps=90, batch_transform=None,
                 **cfg):
        super(CTUDataset, self).__init__()
        self.location = location
        self.transform = transform
        self.cache_loaction = cache_location
        self.cache_name = cache_name
        self.compression = compression
        self.set_number = set_number
        self.period_len = period_len
        self.t_steps = t_steps
        self.batch_transform = batch_transform
        self.cfg = cfg
        self.norm, self.mal = self.load()

    def __len__(self):
        return len(self.norm)

    def __getitem__(self, index):
        return self.norm[index], 0

    def load(self):
        train_cache_name = 'train-' + self.cache_name + "-" + str(self.set_number)
        train_cache_file = self.cache_loaction + "/" + train_cache_name + '.csv'
        data = pd.read_csv(self.location + str(self.set_number) + '/' + ctu_files[self.set_number - 1])

        time_format = '%Y/%m/%d %H:%M:%S.%f'
        numerical_cols = pd.Series(['SrcBytes', 'TotBytes', 'Dur', 'TotPkts', 'sTos', 'dTos'])
        service_lst = ['dhcp', 'dns', 'http', 'ntp', 'smtp', 'ssh', 'ssl']
        # data = data[~data['Label'].str.contains('Background')]
        data['ts'] = data['StartTime'].apply(lambda time_string: datetime.strptime(time_string, time_format))
        data['rel_start'] = data['ts'] - data['ts'].min()
        data['rel_start_in_s'] = data['rel_start'].apply(lambda s: int(s.total_seconds()))
        data['time_bin'] = data['rel_start_in_s'].apply(lambda s: math.floor(s / self.period_len))
        data['malicious'] = data['Label'].apply(lambda s: int('bot' in s.lower()))

        grouped = data.sort_values(['SrcAddr', 'time_bin']).groupby(by=['SrcAddr', 'time_bin'])

        rename_mean = dict(zip(numerical_cols, numerical_cols + '_mean'))
        rename_std = dict(zip(numerical_cols, numerical_cols + '_std'))
        processed = grouped[numerical_cols].mean()
        processed = processed.rename(columns=rename_mean)
        processed = processed.join(grouped[numerical_cols].std())
        processed = processed.rename(columns=rename_std)
        processed = processed.fillna(0)

        protos_group = data.sort_values(['SrcAddr', 'time_bin']).groupby(
            by=['SrcAddr', 'time_bin', 'Proto']).size().unstack(fill_value=0)
        processed = processed.join(protos_group)
        processed = processed.join(grouped['DstAddr'].nunique())
        normalized = MinMaxScaler().fit_transform(processed)
        processed = pd.DataFrame(data=normalized, columns=processed.columns, index=processed.index)
        processed = processed.join((grouped['malicious'].sum() > 0).astype(int))

        idx = list(product(data['SrcAddr'].unique(), range(data['time_bin'].min(), data['time_bin'].max())))
        ts = processed.reset_index().set_index(['SrcAddr', 'time_bin']).reindex(idx)
        ts = ts.fillna(0)

        q = ts.values.reshape(data['SrcAddr'].nunique(), -1, ts.shape[-1])
        b = q[:, 0:self.t_steps * math.floor(q.shape[1] / self.t_steps):, ]
        c = b.reshape(-1, b.shape[-1])
        c = c.reshape(-1, self.t_steps, c.shape[-1])
        z = np.unique((c[:, :, c.shape[-1] - 1] == 1).nonzero()[0])
        p = (~np.isin(np.arange(0, c.shape[0]), z)).nonzero()
        mal = c[z]
        norm = c[p]
        norm = norm[(np.sum(np.sum(norm, axis=2), axis=1) > 0).nonzero()[0]]
        mal = mal[(np.sum(np.sum(mal, axis=2), axis=1) > 0).nonzero()[0]]
        print('norm shape', norm.shape, 'mal shape', mal.shape)
        return norm[:, :, :norm.shape[-1]], mal[:, :, :mal.shape[-1]]
