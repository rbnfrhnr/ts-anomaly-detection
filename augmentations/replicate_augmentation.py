import numpy as np

from augmentations.augmentation_base import BaseAugmentation


class ReplicateAugemnation(BaseAugmentation):

    def __init__(self, **cfg):
        super(ReplicateAugemnation, self).__init__(**cfg)
        self.replication_factor = self.augment_cfg[
            "replication_factor"] if "replication_factor" in self.augment_cfg else 1

    def augment(self, sample, *args, **kwargs):
        replications = []
        for i in range(self.replication_factor):
            replications.append(sample)
        return np.concatenate(replications)
