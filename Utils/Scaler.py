import torch as th
from sklearn.preprocessing import PowerTransformer
from sklearn.preprocessing import QuantileTransformer
import numpy as np


class Scaler(object):
    def __init__(
            self,
            method
    ):
        if method == 'box-cox':
            self.transformer = PowerTransformer(method='box-cox')
        else:
            exit(f'Scaler method {method} is not implemented')

    def fit(self, sample):
        self.transformer.fit(np.array(sample).reshape(-1, 1))
        return self

    def __call__(self, sample):
        return self.transform(sample)

    def transform(self, sample):
        return self.transformer.transform(sample)

    def inverse_transform(self, sample):
        return self.transformer.inverse_transform(sample)
