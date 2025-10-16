import torch
import torch.nn as nn
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


    def training_loop_cv(self, num_epochs=100, optimizer=None, criterion=None, trainX=None, trainY=None, batch_size=128, n_splits=5):
        from sklearn.model_selection import TimeSeriesSplit
        import torch

        tscv = TimeSeriesSplit(n_splits=n_splits)
        fold_train_losses = []
        fold_val_losses = []

        for fold, (train_index, test_index) in enumerate(tscv.split(trainX)):
            print(f"Fold {fold + 1}")
            X_train, X_test = trainX[train_index], trainX[test_index]
            y_train, y_test = trainY[train_index], trainY[test_index]

            X_train = torch.tensor(X_train, dtype=torch.float32)
            y_train = torch.tensor(y_train, dtype=torch.float32)

            fold_epoch_train = []
            fold_epoch_val = []

            for epoch in range(num_epochs):
                self.train()
                epoch_loss = 0.0

                # Mini-batches
                for i in range(0, len(X_train), batch_size):
                    batch_X = X_train[i:i+batch_size]
                    batch_Y = y_train[i:i+batch_size]

                    optimizer.zero_grad()
                    outputs = self.forward(batch_X)
                    loss = criterion(outputs, batch_Y)
                    loss.backward()
                    optimizer.step()

                    epoch_loss += loss.item() * len(batch_X)

                epoch_loss /= len(X_train)
                fold_epoch_train.append(epoch_loss)

                # Validation loss
                with torch.no_grad():
                    X_val = torch.tensor(X_test, dtype=torch.float32)
                    y_val = torch.tensor(y_test, dtype=torch.float32)
                    val_outputs = self.forward(X_val)
                    val_loss = criterion(val_outputs, y_val).item()
                     # Compute validation MSE
                print('Epoch [{}/{}], Train Loss: {:.4f}, Val Loss: {:.4f}'.format(epoch+1, num_epochs, epoch_loss, val_loss))
                fold_epoch_val.append(val_loss)

            fold_train_losses.append(fold_epoch_train)
            fold_val_losses.append(fold_epoch_val)

        return fold_train_losses, fold_val_losses

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

    
    def generate_timeseries(model, start_sequence, steps, generated, device='cpu'):
        """
        Efficiently generate a time series using a single-step model.
        
        Args:
            model: PyTorch model with a sliding window input
            start_sequence: np.ndarray or torch.Tensor of shape (window_size, input_size)
            steps: int, number of future steps to generate
            device: str, 'cpu' or 'cuda'

        Returns:
            np.ndarray of shape (steps, input_size) containing generated values
        """
        model.eval()
        sliding_window_size = 10
        # Convert to tensor if needed and add batch dimension
        if isinstance(start_sequence, np.ndarray):
            input_t = torch.tensor(start_sequence, dtype=torch.float32).to(device)
        else:
            input_t = start_sequence.to(device).float()

        if isinstance(generated, np.ndarray):
            generated = torch.tensor(generated, dtype=torch.float32).to(device)
        else:
            generated = generated.to(device).float()
        
        print(generated.shape, type(generated))
        input_t = generated[:sliding_window_size]  # use the first 'window_size' elements as the initial input
        print(input_t.shape, type(input_t))
        # Pre-allocate array for generated sequence
        

        with torch.no_grad():
            for i in range(1,steps+1):
                input_t = input_t.unsqueeze(0)  # add batch dimension

                output = model(input_t)
                generated = torch.cat((generated, output), dim=0)
                
                input_t = generated[i:i+sliding_window_size]


        return np.array(generated)

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



    
