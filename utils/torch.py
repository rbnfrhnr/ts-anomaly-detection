import torch.nn as nn


class RNNExtract(nn.Module):

    def forward(self, X):
        x, _ = X
        return x


class Reshape(nn.Module):

    def __init__(self, out_shape=None):
        super(Reshape, self).__init__()
        self.out_shape = out_shape

    def forward(self, X):
        return X.reshape(self.out_shape)
