from abc import ABC, abstractmethod


class BaseAugmentation(ABC):

    def __init__(self, dataset=None, **cfg):
        super(BaseAugmentation, self).__init__()

    @abstractmethod
    def augment(self):
        raise NotImplementedError()
