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
from sklearn.metrics import r2_score
from src.timeseriesdataloader import TimeSeriesDataset
from torch.utils.data import DataLoader
from itertools import product
import pandas as pd



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
    from sklearn.model_selection import TimeSeriesSplit
    # Note; split could harm the timeseries
    kf = TimeSeriesSplit(n_splits=n_splits)
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


def append_velocities(t, t_n, scaler, deltat, lag):

    t = t.unsqueeze(0)
    t_unscaled = scaler.inverse_transform(t.numpy())

    t_n_scaled = np.zeros((1, 12))
    t_n_scaled[:, :6] = t_n.numpy()
    t_n_unscaled = scaler.inverse_transform(t_n_scaled)
    t_n_positions = t_n_unscaled[:, :6]
    # print(t_n_positions.shape)

    t_positions = t_unscaled[:, :6]
    difference = t_n_positions - t_positions
    velocities = difference / (deltat*lag)
    # print(velocities.shape, t_n.shape)
    t_new_vel = np.hstack((t_n_positions, velocities))
    # print(t_new_vel.shape)
    t_new_scaled = scaler.transform(t_new_vel)
    # print(t_new_scaled.shape)
    return torch.tensor(t_new_scaled)

def evaluate_predict_velocities_fd(X_train, scaler, deltat, lag, offset=None):
    from sklearn.metrics import mean_squared_error
    """
    NumPy version: predicts velocity at t+offset as (pos_{t+offset} - pos_t) / (deltat*lag)
    Compares to actual velocity at t+offset (after inverse scaling) using sklearn MSE.
    If offset is None, uses offset = lag (recommended).
    """
    if offset is None:
        offset = lag

    n, dim = X_train.shape
    assert dim == 12, "X_train must have 12 columns"
    mse_list = []

    for t in range(n - offset):
        x_t_scaled = X_train[t].reshape(1, -1)
        x_t_off_scaled = X_train[t + offset].reshape(1, -1)

        x_t_unscaled = scaler.inverse_transform(x_t_scaled)         # shape (1,12)
        x_t_off_unscaled = scaler.inverse_transform(x_t_off_scaled)

        pos_t = x_t_unscaled[0, :6]
        pos_t_off = x_t_off_unscaled[0, :6]

        pred_vel = (pos_t_off - pos_t) / (deltat * lag)   # (6,)
        actual_vel = x_t_off_unscaled[0, 6:12]            # (6,)

        mse = mean_squared_error(actual_vel, pred_vel)
        mse_list.append(mse)

    return {
        "num_samples": len(mse_list),
        "mse_list": mse_list,
        "mean_mse": float(np.mean(mse_list)) if mse_list else float("nan"),
        "mean_rmse": float(np.mean(np.sqrt(mse_list))) if mse_list else float("nan")
    }


def generate_timeseries(model, steps, generated, Y_test, criterion, scaler, device='cpu', deltat=1e-3, lag=10):
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

    device = next(model.parameters()).device
    dtype = next(model.parameters()).dtype
    generated = generated.to(device).to(dtype)
    
    with torch.no_grad():
        for i in range(0,steps+1):
            input_t = generated[i]
            input_t = input_t.unsqueeze(0)  # add batch dimension
            output = model(input_t)
            print(output.shape)
            output = append_velocities(input_t.squeeze(0), output, scaler, deltat, lag)
            output = output.to(device).to(dtype)
            generated = torch.cat((generated, output), dim=0)           
            y_val = torch.tensor(Y_test[i], dtype=torch.float32)

    print(generated.shape)
    return np.array(generated)


def grid_search(X_train, y_train, X_test, y_test):


    # define the grid search parameters
    param_grid = {
        'num_layers': [3, 5, 10, 12],
        'num_nodes': [64, 256, 512],
        'learning_rate': [1e-2, 1e-3, 1e-4, 1e-5],
        'epochs': [50, 100, 500],
        'optimizer': ['SGD', 'RMSprop', "ADAM"],
        'num_splits': [5, 10]
    }


    # generate combinations
    keys = list(param_grid.keys())
    combinations = [dict(zip(keys, values)) for values in product(*param_grid.values())]
    print(f"Total combinations to try: {len(combinations)}")

    results_path = "grid_search_results_mlp.csv"
    if os.path.exists(results_path):
        results_df = pd.read_csv(results_path)
    else:
        results_df = pd.DataFrame(columns=keys + ["test_loss"])

    best_val_loss = results_df["test_loss"].min() if not results_df.empty else float('inf')
    best_params = None
    best_model_state = None

    for combo in combinations:
        mask = (results_df[list(combo.keys())] == pd.Series(combo)).all(axis=1)
        if mask.any():
            print(f"Skipping already completed combo: {combo}")
            continue

        print(f"Training with combo: {combo}")


        model = MLP(
            input_size=12*1, 
            layers=[combo['num_nodes'] for i in range(combo['num_layers'])],
            output_size=6,
            initializer_method='xavier',
            activation=nn.ReLU
        )
        
        if combo['optimizer'] == 'SGD':
            optimizer = torch.optim.SGD(model.parameters(), lr=combo['learning_rate'])
        elif combo['optimizer'] == 'RMSprop':
            optimizer = torch.optim.RMSprop(model.parameters(), lr=combo['learning_rate'], weight_decay=1e-4)
        else:
            optimizer = torch.optim.Adam(model.parameters(), lr=combo['learning_rate'])
        
        criterion = nn.MSELoss()
        earlystopping = EarlyStopping(patience=3)

        num_epochs = combo['epochs']
        trainer = Trainer(
            model=model,
            learning_rate=combo['learning_rate'],
            criterion=criterion,
            early_stopping=earlystopping,
            optimizer=optimizer,
            epochs=combo['epochs']
        )
        
        trainer.train(
            X_train=X_train,
            y_train=y_train,
            n_splits=combo['num_splits']
        )

        device = next(model.parameters()).device
        dtype = next(model.parameters()).dtype
        X_test = X_test.to(device).to(dtype)


        output = model.forward(X_test)
        
        criterion = torch.nn.MSELoss()
        test_loss = criterion(output, y_test.to(device).to(dtype)).item()

        # save result immediately
        results_df.loc[len(results_df)] = {**combo, "test_loss": test_loss}
        results_df.to_csv(results_path, index=False)

        if test_loss < best_val_loss:
            best_val_loss = test_loss
            best_params = combo
            best_model_state = model.state_dict()

    print(f"Best combo: {best_params} with val loss: {best_val_loss:.4f}")
    torch.save(best_model_state, "best_lstm_model.pt")
    print("Best model saved to best_lstm_model.pt")


def main():
    sol = load_data()
    train_data, test_data = data_preperation(sol, train_test_split=0.85)

    '''
    scaler = make_pipeline(
        RobustScaler(),
        MinMaxScaler(feature_range=(0, 1))
        )'''
    


    lag = 10
    train_data, test_data, scaler = normalize_data(train_data, test_data)

    model = MLP(
        input_size=12*1,
        layers=[256 for i in range(10)],
        output_size=6,
        initializer_method='xavier',
        activation=nn.ReLU
    )

    earlystopping = EarlyStopping(patience=3) # delta could be 1e-4/5/6/
    trainer = Trainer(
        model,
        learning_rate=0.0001,
        early_stopping=earlystopping,
        epochs=100)

    X_train, y_train = generate_xy(train_data, lag=10, history=1)
    X_test, y_test = generate_xy(test_data, lag=10, history=1)

    print(evaluate_predict_velocities_fd(X_test, scaler, 0.001, 10, 10)['mean_mse'])

    trainer.train(X_train, y_train)

    '''
    grid_search(
        X_train=X_train,
        y_train=y_train,
        X_test=X_test,
        y_test=y_test
    )'''

    # run on test data
    device = next(model.parameters()).device
    dtype = next(model.parameters()).dtype
    X_test = X_test.to(device).to(dtype)
    output = model.forward(X_test)
    
    criterion = torch.nn.MSELoss()
    mse_error = criterion(output, y_test.to(device).to(dtype))
    output = output.detach().numpy()

    r2 = r2_score(y_test, output)
    print(f"r2 on test data = {r2}, MSE: {mse_error}")

    # change output back to full 12 dimensions by adding zeros for vx, vy
    out_full = np.zeros((output.shape[0], 12))
    out_full[:, :6] = output
    output = scaler.inverse_transform(out_full)

    
    # change y_test back to full 12 dimensions by adding zeros for vx, vy
    y_test_full = np.zeros((y_test.shape[0], 12))
    y_test_full[:, :6] = y_test
    true = scaler.inverse_transform(y_test_full)

    plot_trajectories(true[:5000], output[:5000])



    
    generated = X_test[:10]
    y_pred = generate_timeseries(model, 5000, generated, y_test, criterion, scaler)
    output = scaler.inverse_transform(y_pred)
     # change y_test back to full 12 dimensions by adding zeros for vx, vy
    y_test_full = np.zeros((y_test.shape[0], 12))
    y_test_full[:, :6] = y_test
    true = scaler.inverse_transform(y_test_full)

    plot_trajectories(true[:5000], output[:5000])



    '''
    model = LSTM(input_size=12, hidden_size=128, output_size=12, initializer_method='xavier')
    earlystopping = EarlyStopping(patience=1, verbose=True) # delta could be 1e-4/5/6/
    trainer = Trainer(model, learning_rate=0.001, early_stopping=earlystopping)
    trainer.train(X_train, y_train, X_test, y_test, epochs=100)'''


if __name__ == "__main__":
    main()