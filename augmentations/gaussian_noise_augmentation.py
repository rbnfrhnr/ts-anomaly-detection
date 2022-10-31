from augmentations.augmentation_base import BaseAugmentation
import numpy as np


class GaussianNoiseAugmentation(BaseAugmentation):

    def __init__(self, **cfg):
        super(GaussianNoiseAugmentation, self).__init__(**cfg)
        self.mean = self.augment_cfg["mean"]
        self.std = self.augment_cfg["std"]

    def augment(self, sample, *args, **kwargs):
        noise = np.random.normal(self.mean, self.std, sample.shape)
        return sample + noise
