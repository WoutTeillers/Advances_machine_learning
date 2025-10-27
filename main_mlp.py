import numpy as np
import pandas as pd
from data.data_creation import get_trajectories, plot_trajectories
from src.mlp import MLP
from src.trainer_mlp import Trainer, EarlyStopping
import os
import pickle
import torch
from sklearn.preprocessing import RobustScaler, MinMaxScaler
from sklearn.pipeline import make_pipeline
from sklearn.model_selection import KFold
from data.data_creation import plot_trajectories
import torch.nn as nn
import math


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


def generate_xy(data, lag=1, history=False):
    n = len(data)
    if history > lag:
        raise ValueError("History cannot be greater than lag.")
    if history:
        X = torch.stack([torch.flatten(torch.tensor(data[i:i+history])) for i in range(n - lag)])
    else:
        X = torch.stack([torch.tensor(X[i]) for i in range(n - lag)])
    y = torch.tensor(data[lag:])
    print(y.shape)
    # change y to only the x, y positions so first 6 columns
    y = y[:, :6]
    return X, y


def normalize_data(train_data, test_data, scaler=None):
    if scaler is None:
        #scaler = MinMaxScaler(feature_range=(0, 1))
        scaler = RobustScaler()
    scaler.fit(train_data)
    normalized_train_data = scaler.transform(train_data)
    normalized_test_data = scaler.transform(test_data)
    return normalized_train_data, normalized_test_data, scaler


def cross_validation_split(X, y, n_splits=5):
    # Note; split could harm the timeseries
    kf = KFold(n_splits=n_splits, shuffle=True, random_state=44)
    splits = []
    for train_index, val_index in kf.split(X):
        X_train, X_val = torch.tensor(X[train_index]), torch.tensor(X[val_index])
        y_train, y_val = torch.tensor(y[train_index]), torch.tensor(y[val_index])
        splits.append((X_train, y_train, X_val, y_val))
    return splits


def save_model(model, path='models/model', i=0):
    # check if model directory exists
    dir_name = os.path.dirname(path+i+'.pt')
    try:
        if not os.path.exists(dir_name):
            torch.save(model.state_dict(), path)
            return f'Model saved successfully at path {dir_name}.'
        else:
            return save_model(model, path, i+1)
    except RecursionError:
        return 'Failed to save model after multiple attempts.'


def log_cosh_loss(y_pred: torch.Tensor, y_true: torch.Tensor) -> torch.Tensor:
    def _log_cosh(x: torch.Tensor) -> torch.Tensor:
        return x + torch.nn.functional.softplus(-2. * x) - math.log(2.0)
    return torch.mean(_log_cosh(y_pred - y_true))


def relative_mse(y_pred, y_true, eps=1e-8):
    rel_mse = ((y_pred - y_true)**2) / (y_true**2 + eps)    
    return torch.mean(rel_mse)


def main():
    sol = load_data()
    train_data, test_data = data_preperation(sol, train_test_split=0.85)

    scaler = make_pipeline(
        RobustScaler(),
        MinMaxScaler(feature_range=(0, 1))
        )

    train_data, test_data, scaler = normalize_data(train_data, test_data)
    splits = cross_validation_split(train_data, train_data, n_splits=5)

    model = MLP(input_size=12*1, layers=[256 for i in range(10)], output_size=6, initializer_method='xavier', activation=nn.ReLU)
    earlystopping = EarlyStopping(patience=2) # delta could be 1e-4/5/6/
    trainer = Trainer(model, learning_rate=0.00001, early_stopping=earlystopping)
    #X_train, y_train, X_test, y_test = splits[0]

    X_train, y_train = generate_xy(train_data, lag=1, history=1)
    X_test, y_test = generate_xy(test_data, lag=1, history=1)
    trainer.train(X_train, y_train, X_test, y_test, epochs=500)


    device = next(model.parameters()).device
    dtype = next(model.parameters()).dtype
    X_test = X_test.to(device).to(dtype)


    output = model.forward(X_test)
    output = output.detach().numpy()
    # change output back to full 12 dimensions by adding zeros for vx, vy
    out_full = np.zeros((output.shape[0], 12))
    out_full[:, :6] = output
    output = scaler.inverse_transform(out_full)
    out = output

    # change y_test back to full 12 dimensions by adding zeros for vx, vy
    y_test_full = np.zeros((y_test.shape[0], 12))
    y_test_full[:, :6] = y_test

    true = scaler.inverse_transform(y_test_full)
    true = true
    print(out.shape)
    print(true.shape)
    print(type(true), type(out))
    
    plot_trajectories(true, out)

    




    '''
    model = LSTM(input_size=12, hidden_size=128, output_size=12, initializer_method='xavier')
    earlystopping = EarlyStopping(patience=1, verbose=True) # delta could be 1e-4/5/6/
    trainer = Trainer(model, learning_rate=0.001, early_stopping=earlystopping)
    trainer.train(X_train, y_train, X_test, y_test, epochs=100)'''


if __name__ == "__main__":
    main()