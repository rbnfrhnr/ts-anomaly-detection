from abc import ABC, abstractmethod


class BaseAugmentation(ABC, object):

    def __init__(self, augmentation={}, **cfg):
        self.augment_cfg = augmentation["config"]
        super(BaseAugmentation, self).__init__()

    @abstractmethod
    def augment(self, sample, *args, **kwargs):
        raise NotImplementedError()

    def __call__(self, sample, *args, **kwargs):
        return self.augment(sample, *args, **kwargs)
