import torch.nn as nn
import torch


class MLP(nn.Module):
    def __init__(self, input_size, layers, output_size, initializer_method=None, activation=nn.ReLU):
        super(MLP, self).__init__()
        modules = []
        self.sigmoid = nn.Sigmoid()
        prev = input_size
        for layer_size in layers:
            modules.append(nn.Linear(prev, layer_size))
            modules.append(activation())
            prev = layer_size
        modules.append(nn.Linear(prev, output_size))
        # register modules properly
        self.seq = nn.Sequential(*modules)

        if initializer_method:

            for layer in self.seq:
                if initializer_method == "xavier" and isinstance(layer, nn.Linear):
                    nn.init.xavier_uniform_(layer.weight)


    def forward(self, x):
        inp = self.sigmoid(x)
        return self.seq(inp)
        