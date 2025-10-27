import torch.nn as nn
import torch


class MLP(nn.Module):
    def __init__(self, input_size, layers, output_size, initializer_method=None, activation=nn.ReLU):
        super(MLP, self).__init__()
        modules = []
        prev = input_size
        for layer_size in layers:
            modules.append(nn.Linear(prev, layer_size))
            modules.append(activation())
            prev = layer_size
        modules.append(nn.Linear(prev, output_size))
        # register modules properly
        self.seq = nn.Sequential(*modules)

        if initializer_method:
            self._init_weights(initializer_method)

    def _init_weights(self, method):
        for m in self.seq:
            if isinstance(m, nn.Linear):
                if method == "xavier":
                    nn.init.xavier_uniform_(m.weight)
                elif method == "kaiming":
                    nn.init.kaiming_uniform_(m.weight, nonlinearity="relu")
                else:
                    nn.init.normal_(m.weight, 0.0, 0.02)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward(self, x):
        return self.seq(x)