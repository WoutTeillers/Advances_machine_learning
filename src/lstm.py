import numpy as np
import pandas as pd
import torch
import torch.nn as nn

from src.model import WeightInitializer
from src.trainer import EarlyStopping

# might need to change weight initialization for matrices as it could lead to exploding gradients

# inspired by https://towardsdatascience.com/the-math-behind-lstm-9069b835289d/
class LSTM(nn.Module):
    def __init__(self, input_size, hidden_size, output_size, initializer_method):
        super(LSTM, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.initializer_method = WeightInitializer(method=initializer_method)

        # Forget gate weights and biases
        self.wf = nn.Parameter(torch.tensor(
            self.initializer_method.initialize((hidden_size, hidden_size + input_size))
            ))
        self.bf = nn.Parameter(torch.zeros((hidden_size, 1)))

        # Input gate weights and biases
        self.wi = nn.Parameter(torch.tensor(
            self.initializer_method.initialize((hidden_size, hidden_size + input_size))
            ))
        self.bi = nn.Parameter(torch.zeros((hidden_size, 1)))

        # Cell gate weights and biases
        self.wc = nn.Parameter(torch.tensor(
            self.initializer_method.initialize((hidden_size, hidden_size + input_size))
            ))
        self.bc = nn.Parameter(torch.zeros((hidden_size, 1)))

        # Output gate weights and biases
        self.wo = nn.Parameter(torch.tensor(
            self.initializer_method.initialize((hidden_size, hidden_size + input_size))
            ))
        self.bo = nn.Parameter(torch.zeros((hidden_size, 1)))

        self.why = nn.Parameter(torch.tensor(
            self.initializer_method.initialize((output_size, hidden_size))
            ))
        self.by = nn.Parameter(torch.zeros((output_size, 1)))


        '''
        self.wf = self.weight_initializer.initialize((hidden_size, hidden_size + input_size))
        self.wi = self.weight_initializer.initialize((hidden_size, hidden_size + input_size))
        self.wo = self.weight_initializer.initialize((hidden_size, hidden_size + input_size))
        self.wc = self.weight_initializer.initialize((hidden_size, hidden_size + input_size))

        self.bf = np.zeros((hidden_size, 1))
        self.bi = np.zeros((hidden_size, 1))
        self.bo = np.zeros((hidden_size, 1))
        self.bc = np.zeros((hidden_size, 1))'''


    def forward(self, x, h_prev=None, c_prev=None, steps=None):
        outputs = []
        if h_prev is None:
            h_prev = torch.zeros((self.hidden_size, 1), dtype=x.dtype, device=x.device)
        if c_prev is None:
            c_prev = torch.zeros((self.hidden_size, 1), dtype=x.dtype, device=x.device)
        h = h_prev
        c = c_prev

        if steps is None:
            steps = x.shape[0]
        for t in range(steps):
            if t >= x.shape[0]:
                x_t = outputs[-50].unsqueeze(1)
            else:
                x_t = x[t].unsqueeze(1)
            #print(x_t.shape)
            #print(h_prev.shape)
            concat = torch.cat((h_prev, x_t), dim=0)
            #print(concat.dtype)

            #print("Weights shapes:")
            #print(self.wf.shape)
            #print(self.bf.shape)

            f = torch.sigmoid(self.wf @ concat + self.bf)
            i = torch.sigmoid(self.wi @ concat + self.bi)
            o = torch.sigmoid(self.wo @ concat + self.bo)
            c_ = torch.tanh(self.wc @ concat + self.bc)

            c = f * c_prev + i * c_
            h = o * torch.tanh(c)

            h_prev, c_prev = h, c

            y_t = self.why @ h + self.by
            outputs.append(y_t.squeeze(1))

        return torch.stack(outputs, dim=0), (h, c)

        '''
        y = self.why @ h + self.by
        return y
        '''


    def generate_timeseries(self, input_seq, steps):
        self.eval()
        h, c = None, None
        output, (h, c) = self.forward(input_seq, h, c, steps)
        output = output[input_seq.shape[0]:]
        return output


if __name__ == "__main__":
    # Example usage
    input_window_size = 12
    hidden_size = 50
    output_layers = 12

    model = LSTM(input_window_size, hidden_size, output_layers, 'random')
    #print(model)
    sample_input = torch.randn(5, input_window_size, dtype=torch.double)
    #print(sample_input.shape)
    sample_input = sample_input.to("cuda" if torch.cuda.is_available() else "cpu")
    output = model(sample_input)
    #print(output.shape)
