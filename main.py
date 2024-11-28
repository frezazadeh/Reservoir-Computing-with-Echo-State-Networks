import kagglehub
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import pandas as pd
import os

path = kagglehub.dataset_download("arashabbasi/mackeyglass-time-series")

files = os.listdir(path)

dataset_file = os.path.join(path, "Mackey-Glass Time Series(taw17).xlsx")
data = pd.read_excel(dataset_file)
mackey_glass_data = data.iloc[:, 1].values
mackey_glass_data = (mackey_glass_data - np.min(mackey_glass_data)) / (
    np.max(mackey_glass_data) - np.min(mackey_glass_data)
)

plt.figure(figsize=(10, 5))
plt.plot(mackey_glass_data, label="Mackey-Glass Time Series")
plt.xlabel("Time Step")
plt.ylabel("Normalized Value")
plt.title("Mackey-Glass Time Series Dataset")
plt.legend()
plt.savefig("mackey_glass_time_series.png", dpi=300)
plt.show()


class RC_ESN:
    def __init__(self, input_size, reservoir_size, output_size, spectral_radius=0.9, sparsity=0.1):
        self.reservoir_size = reservoir_size
        self.spectral_radius = spectral_radius
        self.W_res = np.random.uniform(-1, 1, (reservoir_size, reservoir_size))
        self.W_res[np.random.rand(*self.W_res.shape) > sparsity] = 0
        max_eigenvalue = np.max(np.abs(np.linalg.eigvals(self.W_res)))
        self.W_res *= spectral_radius / max_eigenvalue
        self.W_in = np.random.uniform(-1, 1, (reservoir_size, input_size))
        self.W_out = None
        self.states = np.zeros((reservoir_size,))

    def forward(self, input_data):
        self.states = np.tanh(
            np.dot(self.W_res, self.states) + np.dot(self.W_in, input_data)
        )
        return self.states

    def train(self, input_seq, target_seq, reg_param=1e-6):
        states = [self.forward(u) for u in input_seq]
        states = np.array(states)
        reg = Ridge(alpha=reg_param, fit_intercept=False)
        reg.fit(states, target_seq)
        self.W_out = reg.coef_

    def predict(self, input_seq):
        return np.array([np.dot(self.W_out, self.forward(u)) for u in input_seq])


train_ratio = 0.8
n_train = int(len(mackey_glass_data) * train_ratio)
train_data = mackey_glass_data[:n_train]
test_data = mackey_glass_data[n_train:]

train_inputs = train_data[:-1].reshape(-1, 1)
train_targets = train_data[1:].reshape(-1, 1)
test_inputs = test_data[:-1].reshape(-1, 1)
test_targets = test_data[1:].reshape(-1, 1)

input_size = 1
reservoir_size = 500
output_size = 1

esn = RC_ESN(input_size, reservoir_size, output_size)
esn.train(train_inputs, train_targets)

test_predictions = esn.predict(test_inputs)

if test_predictions.size > 0 and test_targets.size > 0:
    mse = mean_squared_error(test_targets, test_predictions)
    mae = mean_absolute_error(test_targets, test_predictions)
    r2 = r2_score(test_targets, test_predictions)

    plt.figure(figsize=(10, 5))
    plt.plot(test_targets, label="Actual")
    plt.plot(test_predictions, label="Predicted")
    plt.xlabel("Time Step")
    plt.ylabel("Normalized Value")
    plt.title(
        f"RC-ESN on Mackey-Glass Dataset\nMSE: {mse:.4f}, MAE: {mae:.4f}, R²: {r2:.4f}"
    )
    plt.legend()
    plt.savefig("mackey_glass_time_series-a-p.png", dpi=300)
    plt.show()
else:
    print("Empty predictions or targets. Check your data pipeline.")
