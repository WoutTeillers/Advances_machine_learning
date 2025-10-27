import numpy as np
import torch.optim as optim
import torch
import matplotlib.pyplot as plt

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


class Trainer:
    def __init__(self, model, learning_rate=0.001, early_stopping=None, optimizer=None, criterion=torch.nn.MSELoss()):
        self.model = model
        self.learning_rate = learning_rate
        self.early_stopping = early_stopping
        self.train_losses = []
        self.val_losses = []
        self.optimizer = optimizer if optimizer else optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=1e-6)
        self.criterion = criterion
        
    
    def train(self, X_train, y_train, X_val, y_val, epochs=100):
        device = next(self.model.parameters()).device
        dtype = next(self.model.parameters()).dtype
        X_train = X_train.to(device).to(dtype)
        y_train = y_train.to(device).to(dtype)
        X_val = X_val.to(device).to(dtype)
        y_val = y_val.to(device).to(dtype)

        for epoch in range(epochs):
            self.model.train()
            self.optimizer.zero_grad()
            output = self.model.forward(X_train)
            if output.shape != y_train.shape:
                raise RuntimeError(f"Shape mismatch: output {output.shape} vs target {y_train.shape}")
            loss = self.criterion(output, y_train)
            if not torch.isfinite(loss):
                raise RuntimeError(f"Non-finite training loss at epoch {epoch+1}")

            loss.backward()
            self.optimizer.step()
            self.train_losses.append(loss.item())
            print(f"Epoch {epoch+1}/{epochs}, Training Loss: {self.train_losses[-1]:.6f}")
            self.model.eval()
            with torch.no_grad():
                val_output= self.model.forward(X_val)
                val_loss = self.criterion(val_output, y_val)
                self.val_losses.append(val_loss.item())
                print(f"Epoch {epoch+1}/{epochs}, Validation Loss: {self.val_losses[-1]:.6f}")
                
                self.early_stopping(val_loss)
                if self.early_stopping.early_stop:
                    print("Early stopping")
                    break
            
            
    def plot_losses(self):
        plt.plot(self.train_losses, label='Training Loss')
        plt.plot(self.val_losses, label='Validation Loss')
        plt.xlabel('Epochs')
        plt.ylabel('Loss')
        plt.legend()
        plt.title('Training and Validation Loss over Epochs')
        plt.show()
