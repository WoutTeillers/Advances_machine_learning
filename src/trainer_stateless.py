import numpy as np
import torch.optim as optim
import torch
import matplotlib.pyplot as plt
from sklearn.model_selection import TimeSeriesSplit


class Trainer:
    def __init__(self, model, learning_rate=0.001, optimizer=None, criterion=torch.nn.MSELoss()):
        self.model = model
        self.learning_rate = learning_rate
        self.train_losses = []
        self.val_losses = []
        self.optimizer = optimizer
        self.criterion = criterion


    def train(self, epochs=100, trainX=None, trainY=None, batch_size=128, n_splits=5):
            
            tscv = TimeSeriesSplit(n_splits=n_splits)
            fold_train_losses = []
            fold_val_losses = []

            for fold, (train_index, test_index) in enumerate(tscv.split(trainX)):
                early_stopper = EarlyStopping(patience=3, min_delta=0.0001)

                print(f"Fold {fold + 1}")
                X_train, X_test = trainX[train_index], trainX[test_index]
                y_train, y_test = trainY[train_index], trainY[test_index]

                X_train = torch.tensor(X_train, dtype=torch.float32)
                y_train = torch.tensor(y_train, dtype=torch.float32)

                fold_epoch_train = []
                fold_epoch_val = []

                for epoch in range(epochs):
                    self.model.train()
                    epoch_loss = 0.0

                    # Mini-batches
                    for i in range(0, len(X_train), batch_size):
                        batch_X = X_train[i:i+batch_size]
                        batch_Y = y_train[i:i+batch_size]

                        self.optimizer.zero_grad()
                        outputs = self.model.forward(batch_X)
                        loss = self.criterion(outputs, batch_Y)
                        loss.backward()
                        self.optimizer.step()

                        epoch_loss += loss.item() * len(batch_X)

                    epoch_loss /= len(X_train)
                    fold_epoch_train.append(epoch_loss)

                    # Validation loss
                    with torch.no_grad():
                        X_val = torch.tensor(X_test, dtype=torch.float32)
                        y_val = torch.tensor(y_test, dtype=torch.float32)
                        val_outputs = self.model.forward(X_val)
                        val_loss = self.criterion(val_outputs, y_val).item()
                        # Compute validation MSE
                    print('Epoch [{}/{}], Train Loss: {:.4f}, Val Loss: {:.4f}'.format(epoch+1, epochs, epoch_loss, val_loss))
                    fold_epoch_val.append(val_loss)

                    if early_stopper.early_stop(val_loss):    
                        print("Early stopping triggered")         
                        break
                self.train_losses.append(fold_epoch_train)
                self.val_losses.append(fold_epoch_val)
                fold_train_losses.append(fold_epoch_train)
                fold_val_losses.append(fold_epoch_val)

            return fold_train_losses, fold_val_losses
    
    def plot_losses(self):
        losses = self.train_losses
        val_loss = self.val_losses
        max_len = 20
                # ensure sub arrays are of the same length by padding with -1
        losses_array = np.array([
            lst + [-1] * (max_len - len(lst)) if len(lst) < max_len else lst[:max_len]
            for lst in losses
        ])
        # compute average ignoring -1 values
        avg_loss = np.mean(losses_array, axis=0, where=(losses_array != -1))

        # ensure sub arrays are of the same length by padding with -1
        losses_array_val = np.array([
            lst + [-1] * (max_len - len(lst)) if len(lst) < max_len else lst[:max_len]
            for lst in val_loss
        ])
        # compute average ignoring -1 values
        avg_val_loss = np.mean(losses_array_val, axis=0, where=(losses_array_val != -1))
        plt.plot(avg_loss, label='Training Loss')
        plt.plot(avg_val_loss, label='Validation Loss')
        plt.xlabel('Epochs')
        plt.ylabel('Loss')
        plt.legend()
        plt.title('Training and Validation Loss over Epochs')
        plt.show()



# https://stackoverflow.com/questions/71998978/early-stopping-in-pytorch
class EarlyStopping:
    def __init__(self, patience=1, min_delta=0):
        self.patience = patience
        self.min_delta = min_delta
        self.counter = 0
        self.min_validation_loss = float('inf')

    def early_stop(self, validation_loss):
        if validation_loss < self.min_validation_loss:
            self.min_validation_loss = validation_loss
            self.counter = 0
        elif validation_loss > (self.min_validation_loss + self.min_delta):
            self.counter += 1
            if self.counter >= self.patience:
                return True
        return False