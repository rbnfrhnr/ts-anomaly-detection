from augmentations.augmentation_base import BaseAugmentation


class ReplicateAugemnation(BaseAugmentation):

    def __init__(self, **cfg):
        super(ReplicateAugemnation, self).__init__(**cfg)

    def augment(self, sample, *args, **kwargs):
        return sample
