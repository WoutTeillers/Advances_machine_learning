import numpy as np

class WeightInitializer:
    def __init__(self, method='random'):
        self.method = method

    def initialize(self, shape):
        if self.method == 'random':
            return np.random.randn(*shape)
        elif self.method == 'xavier':
            return np.random.randn(*shape) / np.sqrt(shape[0])