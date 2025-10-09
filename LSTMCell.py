import torch
import torch.nn as nn
from data_creation import get_trajectories
import numpy as np
# code taken from: https://medium.com/@wangdk93/lstm-from-scratch-c8b4baf06a8b

class LSTMCell(nn.Module):
    def __init__(self, input_size, hidden_size):
        super(LSTMCell, self).__init__()
        self.hidden_size = hidden_size
        self.input_size = input_size
        
        # Input gate components
        self.W_ii = nn.Parameter(torch.Tensor(hidden_size, input_size))
        self.W_hi = nn.Parameter(torch.Tensor(hidden_size, hidden_size))
        self.b_i = nn.Parameter(torch.Tensor(hidden_size))
        
        # Forget gate components
        self.W_if = nn.Parameter(torch.Tensor(hidden_size, input_size))
        self.W_hf = nn.Parameter(torch.Tensor(hidden_size, hidden_size))
        self.b_f = nn.Parameter(torch.Tensor(hidden_size))
        
        # Cell gate components
        self.W_ig = nn.Parameter(torch.Tensor(hidden_size, input_size))
        self.W_hg = nn.Parameter(torch.Tensor(hidden_size, hidden_size))
        self.b_g = nn.Parameter(torch.Tensor(hidden_size))
        
        # Output gate components
        self.W_io = nn.Parameter(torch.Tensor(hidden_size, input_size))
        self.W_ho = nn.Parameter(torch.Tensor(hidden_size, hidden_size))
        self.b_o = nn.Parameter(torch.Tensor(hidden_size))
        
        self.init_weights()

    def init_weights(self):
        for param in self.parameters():
            nn.init.uniform_(param, -0.1, 0.1)
        
    def forward(self, x, hidden):
        h_prev, c_prev = hidden

        i_t = torch.sigmoid(x @ self.W_ii.T + h_prev @ self.W_hi.T + self.b_i)
        f_t = torch.sigmoid(x @ self.W_if.T + h_prev @ self.W_hf.T + self.b_f)
        g_t = torch.tanh(x @ self.W_ig.T + h_prev @ self.W_hg.T + self.b_g)
        o_t = torch.sigmoid(x @ self.W_io.T + h_prev @ self.W_ho.T + self.b_o)
        
        c_t = f_t * c_prev + i_t * g_t
        h_t = o_t * torch.tanh(c_t)
        
        return h_t, (h_t, c_t)


class LSTM(nn.Module):
    def __init__(self, input_window_size, hidden_size, num_layers):
        super(LSTM, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.cells = nn.ModuleList([LSTMCell(input_window_size, hidden_size) if i == 0 
                                    else LSTMCell(hidden_size, hidden_size) 
                                    for i in range(num_layers)])
        self.fc = nn.Linear(hidden_size, 12)

    def forward(self, x):

        batch_size, seq_len, _ = x.size()
        h = [torch.zeros(batch_size, self.hidden_size).to(x.device) for _ in range(self.num_layers)]
        c = [torch.zeros(batch_size, self.hidden_size).to(x.device) for _ in range(self.num_layers)]
        
        for t in range(seq_len):
            x_t = x[:, t, :]
            for i, cell in enumerate(self.cells):
                h[i], (h[i], c[i]) = cell(x_t, (h[i], c[i]))
                x_t = h[i]
        
        out = self.fc(h[-1])
        return out


    def training_loop(self, num_epochs = 100, optimizer = None, criterion = None, trainX = None, trainY = None):

        for epoch in range(num_epochs):
            self.train()
            optimizer.zero_grad()

            for sample, y_sample in zip(trainX, trainY):

                outputs = self.forward(torch.from_numpy(sample).float().unsqueeze(0))

                loss = criterion(outputs, torch.from_numpy(y_sample).float().unsqueeze(0))

                loss.backward()
                optimizer.step()

            print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {loss.item():.4f}')

if __name__ == "__main__":
    # Example usage
    input_window_size = 12
    hidden_size = 50
    num_layers = 2

    model = LSTM(input_window_size, hidden_size, num_layers)
    print(model)
    sample_input = torch.randn(1, 5, input_window_size)
    sample_input = sample_input.to("cuda" if torch.cuda.is_available() else "cpu")
    output = model(sample_input) 
    print(output)  # Should output torch.Size([1, 12])



    
