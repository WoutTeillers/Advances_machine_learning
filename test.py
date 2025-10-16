from data_creation import get_trajectories

import numpy as np

sol = get_trajectories()

x,y,vx,vy,t = sol

print((x.shape))

# Applied sigmoid but does not seem to do much
# We can first try without sigmoid


# custom function
def sigmoid(x):
  return 1 / (1 + np.exp(-x))


# define vectorized sigmoid
sigmoid_v = np.vectorize(sigmoid)

# # test
x_sig = sigmoid(x)  
y_sig = sigmoid(y)
vx_sig = sigmoid(vx)
vy_sig = sigmoid(vy)



sol = get_trajectories()

x,y,vx,vy,t = sol

print((x.shape))
import numpy as np
import pandas as pd

data = np.vstack((x, y, vx, vy)).T

from data_creation import transform_data
X_train, y_train, X_test, y_test, scaler = transform_data(data, window_size=10, test_size=0.2)


import os
import torch

model_path = "models\model_20251016-174704.pt"

if os.path.exists(model_path):
    print("Loading saved model...")
    model = torch.load(model_path, weights_only=False)

from data_creation import plot_trajectories
import torch.nn as nn
steps = 100

sliding_window = 10
print(X_test.shape)
generated = X_test[:19]
generated = generated[:, 0, :]
criterion = nn.MSELoss()
output = model.generate_timeseries(steps=steps, generated=generated, Y_test= y_test, criterion=criterion, sliding_window_size=sliding_window)
print(output[19:].shape)

# remove middle dimension
X_test_selected = X_test[:, 0, :]
print(X_test_selected[19:steps+19].shape)

plot_trajectories(X_test_selected[19:steps+19], output[19:])