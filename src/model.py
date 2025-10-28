import numpy as np


# inspired by https://towardsdatascience.com/the-math-behind-lstm-9069b835289d/
class WeightInitializer:
    def __init__(self, method='random'):
        self.method = method

    def initialize(self, shape):
        if self.method == 'random':
            return np.random.randn(*shape)
        elif self.method == 'xavier':
            return np.random.randn(*shape) / np.sqrt(shape[0])
        elif self.method == 'kaiming':
            return np.random.randn(*shape) * np.sqrt(2 / shape[0])