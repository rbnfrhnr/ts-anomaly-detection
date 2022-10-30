from abc import ABC, abstractmethod


class DownstreamBase(ABC, object):

    def __init__(self):
        super(DownstreamBase, self).__init__()

    @abstractmethod
    def fit(self, loader):
        raise NotImplementedError()

    @abstractmethod
    def predict(self, X):
        raise NotImplementedError()
