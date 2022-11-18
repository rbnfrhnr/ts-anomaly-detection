from abc import ABC, abstractmethod

import numpy as np


class BaseAugmentation(ABC, object):

    def __init__(self, augmentation={}, **cfg):
        self.augment_cfg = augmentation["config"]
        self.replication_factor = self.replication_factor = self.augment_cfg[
            "replication_factor"] if "replication_factor" in self.augment_cfg else 1
        super(BaseAugmentation, self).__init__()

    @abstractmethod
    def augment(self, sample, *args, **kwargs):
        raise NotImplementedError()

    def __call__(self, sample, *args, **kwargs):
        replications = []
        for i in range(self.replication_factor):
            replications.append(self.augment(sample, *args, **kwargs))
        return np.concatenate(replications)
