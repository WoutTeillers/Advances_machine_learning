import numpy as np
import pandas as pd
from data.data_creation import get_trajectories, plot_trajectories
from src.LSTMCell import LSTM
from src.trainer_stateless import Trainer
import os
import pickle
import torch
from sklearn.preprocessing import MinMaxScaler, RobustScaler
from sklearn.model_selection import KFold
import time
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import r2_score


def load_data():
    if not os.path.exists('data/trajectories.npy'):
        sol = get_trajectories()  # x,y,vx,vy,t
        pickle.dump(sol, open('data/trajectories.npy', 'wb'))
    else:
        sol = np.load('data/trajectories.npy', allow_pickle=True)
    return sol
    
def data_preperation(sol, train_test_split=0.85): 
    x,y,vx,vy,t = sol

    data = np.vstack((x, y, vx, vy)).T
    
    print(data.dtype)
    print(f"Data shape: {data.shape}")
    
    train_end = int(train_test_split * len(data))
    
    train_data = data[:train_end]
    test_data = data[train_end:]
    print(f"Train data shape: {train_data.shape}\nTest data shape: {test_data.shape}")

    return train_data, test_data


def generate_xy(data, lag=10):
    X = np.array([data[i:i+10] for i in range(len(data) - lag - 10)])
    y = np.array(data[lag:])
    return X, y


def normalize_data(train_data, test_data, scaler=None):
    if scaler is None:
        scaler = RobustScaler()
        scaler.fit(train_data)
    normalized_train_data = scaler.transform(train_data)
    normalized_test_data = scaler.transform(test_data)
    return normalized_train_data, normalized_test_data, scaler


def cross_validation_split(X, y, n_splits=5):
    # Note; split could harm the timeseries
    kf = KFold(n_splits=n_splits, shuffle=False)
    splits = []
    for train_index, val_index in kf.split(X):
        X_train, X_val = torch.tensor(X[train_index]), torch.tensor(X[val_index])
        y_train, y_val = torch.tensor(y[train_index]), torch.tensor(y[val_index])
        splits.append((X_train, y_train, X_val, y_val))
    return splits


def save_model(model):
    timestamp = time.strftime("%Y%m%d-%H%M%S")
    filename = f"models/model_{timestamp}.pt"

    torch.save(model, filename)

def plot_density(output):
    bodies = ["Body 1", "Body 2", "Body 3"]
    components = ["x", "y", "vx", "vy"]
    feature_names = [f"{body} {comp}" for comp in components for body in bodies]

    num_features = output.shape[1]
    fig, axes = plt.subplots(4, 3, figsize=(16, 10))  # 3 rows × 4 cols for 12 features
    axes = axes.flatten()

    for i in range(num_features):
        sns.kdeplot(output[:, i], ax=axes[i], fill=True, bw_adjust=0.5)
        axes[i].set_title(feature_names[i])
        axes[i].set_xlabel("Value")
        axes[i].set_ylabel("Density")

    plt.tight_layout()
    plt.show()


def validate_test_data(model, X_test, y_test, criterion):
    losses = []
    r2_scores = []
    test_preds = []
    for i, x in enumerate(X_test):
        with torch.no_grad():
            
            y_pred = model.forward(torch.from_numpy(x).float().unsqueeze(0))
            test_preds.append(y_pred.numpy().flatten())
            # Compute MSE
            y_val = torch.tensor(y_test[i], dtype=torch.float32)
            mse_loss = criterion(y_pred, y_val.unsqueeze(0)).item()
            losses.append(mse_loss)

            
            # Compute R2
            r2 = r2_score(y_val.numpy(), y_pred.detach().numpy().flatten())
            r2_scores.append(r2)
            # print(f"MSE: {mse_loss}, R2: {r2}")

    print(f"Average MSE on test set: {np.mean(losses)}")
    print(f"Average R2 on test set: {np.mean(r2_scores)}")
    return test_preds


def main():
    sol = load_data()
    lag = 10
    train_data, test_data = data_preperation(sol, train_test_split=0.85)
    train_data, test_data, scaler = normalize_data(train_data, test_data)

    X_train, y_train = generate_xy(train_data, lag=lag)
    X_test, y_test = generate_xy(test_data, lag=lag)

    criterion = torch.nn.MSELoss()

    model_path = ""

    if os.path.exists(model_path):
        print("Loading saved model...")
        model = torch.load(model_path, weights_only=False)
    else: 
        input_window_size = 12
        hidden_size = 32
        num_layers = 2
        model = LSTM(input_window_size, hidden_size, num_layers)

        
        optimizer = torch.optim.Adam(model.parameters(), lr=1e-4, weight_decay=1e-4)

        trainer = Trainer(model, learning_rate=0.001, optimizer=optimizer)
        trainer.train(
            trainX=X_train,
            trainY=y_train,
            epochs=20
        )

        trainer.plot_losses()

    test_preds = validate_test_data(model, X_test, y_test, criterion)

    print('generating timeseries')
    steps = 5000

    generated = X_test[:19]  
    generated = generated[:, -1, :]

    output = model.generate_timeseries(steps=steps, generated=generated, Y_test=y_test[20:], criterion=criterion, sliding_window_size=lag)
    plot_density(output)
    plot_trajectories(scaler.inverse_transform(y_test[0:steps]), scaler.inverse_transform(output))
    plot_trajectories(scaler.inverse_transform(y_test), scaler.inverse_transform(test_preds))


    save_message = save_model(model)
    print(save_message)
    

if __name__ == "__main__":
    main()
