import numpy as np

from augmentations.augmentation_base import BaseAugmentation


class ReverseAugemnation(BaseAugmentation):

    def __init__(self, **cfg):
        super(ReverseAugemnation, self).__init__(**cfg)

    def augment(self, sample, *args, **kwargs):
        flipped = np.flip(sample, 1)
        return flipped
