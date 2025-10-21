import numpy as np
import pandas as pd
from data.data_creation import get_trajectories, plot_trajectories
from src.lstm import LSTM
from src.trainer import Trainer, EarlyStopping
import os
import pickle
import torch
from sklearn.preprocessing import MinMaxScaler


if not os.path.exists('data/trajectories.npy'):
    sol = get_trajectories()  # x,y,vx,vy,t
    pickle.dump(sol, open('data/trajectories.npy', 'wb'))
else:
    sol = np.load('data/trajectories.npy', allow_pickle=True)

x,y,vx,vy,t = sol


data = np.vstack((x, y, vx, vy)).T
print(data.dtype)
print(f"Data shape: {data.shape}")
split_point = int(0.8 * len(data))
train_data = data[:split_point]
test_data = data[split_point:]
print(f"Train data shape: {train_data.shape}\nTest data shape: {test_data.shape}")

scaler = MinMaxScaler(feature_range=(0, 1))  
scaler.fit(train_data) 
train_data = scaler.transform(train_data)  
test_data = scaler.transform(test_data)

lag = 1
X_train, y_train = torch.tensor(train_data[:-lag]), torch.tensor(train_data[lag:])
print(X_train.dtype)
X_test, y_test = torch.tensor(test_data[:-lag]), torch.tensor(test_data[lag:])
print(f"X_train shape: {X_train.shape}, y_train shape: {y_train.shape}")


model = LSTM(input_size=12, hidden_size=50, output_size=12, initializer_method='xavier')
earlystopping = EarlyStopping(patience=1, verbose=True) # delta could be 1e-4/5/6/
trainer = Trainer(model, learning_rate=0.01, early_stopping=earlystopping)
trainer.train(X_train, y_train, X_test, y_test, epochs=100)

trainer.plot_losses()


steps = 2000
true = y_test[:steps]
output = model.generate_timeseries(X_test[:1000], steps=steps)


true = true.detach().numpy()
true = scaler.inverse_transform(true)
output = output.detach().numpy()
print(output.shape)
output = scaler.inverse_transform(output)
print(type(true), type(output))
plot_trajectories(true, output)
