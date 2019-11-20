from torch import nn
from abc import ABC, abstractmethod


class UncertaintyModel(nn.Module, ABC):

    @abstractmethod
    def loss(self, *args):
        pass

    @abstractmethod
    def predict(self, *args):
        pass

    @abstractmethod
    def predict_with_uncertainty(self, *args):
        pass