import numpy as np
import torch.optim as optim
import torch
import matplotlib.pyplot as plt
from sklearn.metrics import r2_score
from sklearn.model_selection import TimeSeriesSplit
from src.timeseriesdataloader import TimeSeriesDataset
from torch.utils.data import DataLoader


class EarlyStopping:
    """
    Early stopping to stop the training when the loss does not improve after

    Args:
    -----
        patience (int): Number of epochs to wait before stopping the training.
        verbose (bool): If True, prints a message for each epoch where the loss
                        does not improve.
        delta (float): Minimum change in the monitored quantity to qualify as an improvement.
    """
    def __init__(self, patience=7, verbose=False, delta=0):
        self.patience = patience
        self.verbose = verbose
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.delta = delta

    def __call__(self, val_loss):
        """
        Determines if the model should stop training.

        Args:
            val_loss (float): The loss of the model on the validation set.
        """
        score = -val_loss

        if self.best_score is None:
            self.best_score = score

        elif score < self.best_score + self.delta:
            self.counter += 1

            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_score = score
            self.counter = 0

    def reset(self):
        """Resets the early stopping state for a new training session or fold."""
        self.counter = 0
        self.best_score = None
        self.early_stop = False


class Trainer:
    def __init__(self, model, learning_rate=0.001, early_stopping=None, optimizer=None, criterion=torch.nn.MSELoss(), epochs=100):
        self.model = model
        self.learning_rate = learning_rate
        self.early_stopping = early_stopping
        self.train_losses = []
        self.val_losses = []
        self.optimizer = optimizer if optimizer else optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=1e-6)
        self.criterion = criterion
        self.epochs = epochs
        
    
    def train(self, X_train, y_train, n_splits=5):
        device = next(self.model.parameters()).device
        dtype = next(self.model.parameters()).dtype
        trainX = X_train.to(device).to(dtype)
        trainY = y_train.to(device).to(dtype)

        n_samples = len(X_train)   # CPU length
        tscv = TimeSeriesSplit(n_splits=n_splits)

        for fold, (train_idx, test_idx) in enumerate(tscv.split(np.arange(n_samples))):
            # print(f"fold {fold + 1}")
            self.early_stopping.reset()


            train_idx_t = torch.tensor(train_idx, dtype=torch.long, device=device)
            test_idx_t  = torch.tensor(test_idx,  dtype=torch.long, device=device)

            train_dataset = TimeSeriesDataset(trainX[train_idx_t], trainY[train_idx_t])
            X_val, y_val = trainX[test_idx_t], trainY[test_idx_t]

            train_dataloader = DataLoader(train_dataset, batch_size=500, shuffle=True)

            fold_train_losses = []
            fold_val_losses = []

            for epoch in range(self.epochs):
                self.model.train()
                running_loss = 0.0
                for idx, (X_batch, y_batch) in enumerate(train_dataloader):
                    #print(f'batch: {idx+1}/{len(train_dataloader)}')
                    self.optimizer.zero_grad()
                    output = self.model(X_batch)
                    loss = self.criterion(output, y_batch)
                    loss.backward()
                    self.optimizer.step()
                    running_loss += loss.item()
                fold_train_losses.append(loss.item())

                self.model.eval()
                with torch.no_grad():
                    val_output = self.model(X_val)
                    val_loss = self.criterion(val_output, y_val)
                    fold_val_losses.append(val_loss.item())
                    r2 = r2_score(y_val.cpu().numpy(), val_output.cpu().numpy())
                    # print(f"Epoch {epoch+1}/{self.epochs}, Train Loss: {loss.item():.6f}, Val Loss: {val_loss.item():.6f}, R^2: {r2:.2f}")

                    self.early_stopping(val_loss)
                    if self.early_stopping.early_stop:
                        # print("Early stopping")
                        break

            self.train_losses.extend(fold_train_losses)
            self.val_losses.extend(fold_val_losses)
                
            
    def plot_losses(self):
        plt.plot(self.train_losses, label='Training Loss')
        plt.plot(self.val_losses, label='Validation Loss')
        plt.xlabel('Epochs')
        plt.ylabel('Loss')
        plt.legend()
        plt.title('Training and Validation Loss over Epochs')
        plt.show()
