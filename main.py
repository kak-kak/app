import torch
import torch.nn as nn
import numpy as np
from typing import Tuple, List
import matplotlib.pyplot as plt
import numpy as np
from typing import Tuple
from torchvision import transforms

from dataloader.timeSeriesDatasetFromNumpy import TimeSeriesDatasetFromNumpy
from model.lstm import LSTMModel
from sklearn.preprocessing import MinMaxScaler, StandardScaler

def generate_time_series(num_points: int, num_trend_changes: int, min=0, max=1000) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    x = np.linspace(min, max, num_points)
    trend_points = np.linspace(min, max, num_trend_changes + 2)[1:-1]
    slopes = [np.random.uniform(-2, 2) for _ in range(num_trend_changes + 1)]
    slopes_list = []
    y = np.zeros(num_points)
    start_idx = 0
    for i, point in enumerate(trend_points, start=1):
        end_idx = np.searchsorted(x, point, side='right')
        y[start_idx:end_idx] = slopes[i-1] * (x[start_idx:end_idx] - x[start_idx]) + y[start_idx-1]
        slopes_list += [slopes[i-1] for _ in range(end_idx - start_idx)]
        start_idx = end_idx
    y[start_idx:] = slopes[-1] * (x[start_idx:] - x[start_idx]) + y[start_idx-1]
    slopes_list += [slopes[-1] for _ in range(num_points - start_idx)]

    second_derivative = np.gradient(np.gradient(y, x), x)
    y += np.random.normal(scale=1, size=y.shape)
    return x, y, np.array(slopes_list).flatten(), second_derivative

time_series_length = 10000
num_trend_changes = int(time_series_length / 100)
sequence_length = 10

X, Data, Slope, second_derivatives = generate_time_series(time_series_length, num_trend_changes)

scaler = StandardScaler()
Data_normalized = scaler.fit_transform(Data.reshape(-1, 1)).flatten()
Slope_normalized = scaler.fit_transform(Slope.reshape(-1, 1)).flatten()

fig = plt.figure(figsize=(10, 10))
ax1 = fig.add_subplot(211)
ax1.plot(X, Data_normalized, label='data')
ax2 = fig.add_subplot(212)
ax2.plot(X, second_derivatives, label='second derivative') 

plt.legend()
plt.savefig('time_series.png')
dataset = TimeSeriesDatasetFromNumpy(data=Data_normalized, label=Slope_normalized, sequence_length=sequence_length)
dataloader = torch.utils.data.DataLoader(dataset, batch_size=32, shuffle=True)

model = LSTMModel(input_size=sequence_length, hidden_size=32, num_layers=5)
criterion = nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

num_epochs = 10
for epoch in range(num_epochs):
    for i, (data, label) in enumerate(dataloader):
        data = data.float().reshape((-1, 1, sequence_length))
        label = label.float().unsqueeze(-1)
        optimizer.zero_grad()
        output = model(data)
        loss = criterion(output, label)
        loss.backward()
        optimizer.step()
    print('Epoch [{}/{}], Loss: {:.4f}'.format(epoch+1, num_epochs, loss.item()))



test_time_series_length = 1000
test_num_trend_changes = int(time_series_length / 100)
test_sequence_length = 10

test_X, test_Data, test_Slope, test_second_derivatives = generate_time_series(test_time_series_length, test_num_trend_changes)

test_Data_normalized = scaler.fit_transform(test_Data.reshape(-1, 1)).flatten()
test_Slope_normalized = scaler.fit_transform(test_Slope.reshape(-1, 1)).flatten()

test_dataset = TimeSeriesDatasetFromNumpy(data=test_Data_normalized, label=test_Slope_normalized, sequence_length=test_sequence_length)
test_dataloader = torch.utils.data.DataLoader(test_dataset, batch_size=1, shuffle=False)

predictions = []
model.eval()
with torch.no_grad():
    for i, (data, label) in enumerate(test_dataloader):
        data = data.float().reshape((-1, 1, test_sequence_length))
        label = label.float().unsqueeze(-1)
        output = model(data)
        predictions.append(output.item())
        loss = criterion(output, label)
        print('Loss: {:.4f}'.format(loss.item()))

plt.close()
fig = plt.figure(figsize=(10, 10))
ax1 = fig.add_subplot(311)
ax1.plot(test_X[:test_time_series_length - test_sequence_length], test_Data_normalized[:test_time_series_length - test_sequence_length], label='data')
ax2 = fig.add_subplot(312)
ax2.plot(test_X[:test_time_series_length - test_sequence_length], test_Slope[:test_time_series_length - test_sequence_length], label='answer')
ax3 = fig.add_subplot(313)
ax3.plot(test_X[:test_time_series_length - test_sequence_length], predictions, label='prediction')
plt.legend()
plt.savefig('test_time_series.png')
