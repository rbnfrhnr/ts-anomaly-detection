import numpy as np

from downstream.downstream_base import DownstreamBase
from utils.common import batch


class SingularRollingThreshold(DownstreamBase):

    def __init__(self, model=None, batch_size=None, **cfg):
        super(SingularRollingThreshold, self).__init__()
        self.loader = None
        self.model = model
        self.pdf = None
        self.cfg = cfg
        self.batch_size = batch_size

    def fit(self, loader):
        self.loader = loader
        return self

    def predict(self, X):
        window_size = X.shape[1]
        batch_size = self.batch_size if self.batch_size is not None else X.shape[0]
        errs = []
        for b in batch(X, batch_size):
            recon = self.model(b)
            err = ((b - recon) ** 2).cpu().detach().numpy().mean(axis=1)
            errs.append(err)
        errs = np.concatenate(errs)

        m = np.zeros(errs.shape[0] + window_size)
        for idx, rec in enumerate(errs):
            m[idx:idx + window_size] += rec.repeat(window_size)

        weights = np.minimum(np.arange(1, m.shape[0] + 1), np.repeat(window_size, m.shape[0]))
        weights[-window_size + 1:] = np.arange(window_size - 1, 0, -1)
        means = m / weights
        means = means[:-window_size - 1]
        return means
