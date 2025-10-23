import numpy as np
import pandas as pd
from data.data_creation import get_trajectories, plot_trajectories
from src.lstm import LSTM
from src.trainer import Trainer, EarlyStopping
import os
import pickle
import torch
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import KFold

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


def generate_xy(data, lag=1):
    X = torch.tensor(data[:-lag])
    y = torch.tensor(data[lag:])
    return X, y


def normalize_data(train_data, test_data, scaler=None):
    if scaler is None:
        scaler = MinMaxScaler(feature_range=(0, 1))
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


def main():
    sol = load_data()
    train_data, test_data = data_preperation(sol, train_test_split=0.85)
    train_data, test_data, scaler = normalize_data(train_data, test_data)
    X_train, y_train = generate_xy(train_data, lag=50)
    X_test, y_test = generate_xy(test_data, lag=50)
    splits = cross_validation_split(train_data, train_data, n_splits=5)

    model = LSTM(input_size=12, hidden_size=1028, output_size=12, initializer_method='xavier')
    earlystopping = EarlyStopping(patience=1, verbose=True) # delta could be 1e-4/5/6/
    trainer = Trainer(model, learning_rate=0.001, early_stopping=earlystopping)
    trainer.train(X_train, y_train, X_test, y_test, epochs=100)

    trainer.plot_losses()

    initializtion_steps = 1000
    steps = 2000
    true = y_test[initializtion_steps:steps]
    output = model.generate_timeseries(X_test[:initializtion_steps], steps=steps)


    true = true.detach().numpy()
    true = scaler.inverse_transform(true)
    output = output.detach().numpy()
    print(output.shape)
    output = scaler.inverse_transform(output)
    print(type(true), type(output))
    plot_trajectories(true, output)

    model_save_path = 'models/lstm_model'
    save_message = save_model(model, path=model_save_path)
    print(save_message)


if __name__ == "__main__":
    main()
