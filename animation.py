import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import pandas as pd
import os
from matplotlib.animation import FFMpegWriter


# Download dataset using kagglehub
path = kagglehub.dataset_download("arashabbasi/mackeyglass-time-series")
files = os.listdir(path)

# Specify the dataset file path
dataset_file = os.path.join(path, "Mackey-Glass Time Series(taw17).xlsx")

# Read the Excel file using openpyxl
data = pd.read_excel(dataset_file, engine='openpyxl')
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

# Split the dataset into training and testing sets
train_ratio = 0.7
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


def combined_animation(esn, train_inputs, train_targets, test_inputs, test_targets, interval=50, output_file="combined_animation.mp4"):
    fig, axes = plt.subplots(2, 1, figsize=(12, 12))

    # Training and Testing Plot
    axes[0].set_xlim(0, max(len(train_targets), len(test_targets)))
    axes[0].set_ylim(-1, 1)
    axes[0].set_title("Training and Testing Procedure")
    axes[0].set_xlabel("Time Steps")
    axes[0].set_ylabel("Normalized Value")
    train_line, = axes[0].plot([], [], label="Training Target", color="blue", marker="*", markevery=20 )
    train_pred_line, = axes[0].plot([], [], label="Training Prediction", color="green")
    test_line, = axes[0].plot([], [], label="Testing Target", color="orange", marker="s", markevery=20)
    test_pred_line, = axes[0].plot([], [], label="Testing Prediction", color="red")
    step_text = axes[0].text(0.05, 0.95, "", transform=axes[0].transAxes, fontsize=12, verticalalignment='top')
    axes[0].legend()

    # Reservoir Activations Plot
    axes[1].set_xlim(0, esn.reservoir_size)
    axes[1].set_ylim(-1, 1)
    axes[1].set_title("Reservoir Activations")
    axes[1].set_xlabel("Reservoir Neurons")
    axes[1].set_ylabel("Activation Value")
    scatter = axes[1].scatter([], [], s=10, color="blue", label="Neuron Activations")
    activation_text = axes[1].text(0.95, 0.95, '', transform=axes[1].transAxes, ha='right', fontsize=12, color='red')

    def init():
        train_line.set_data([], [])
        train_pred_line.set_data([], [])
        test_line.set_data([], [])
        test_pred_line.set_data([], [])
        step_text.set_text("")
        scatter.set_offsets(np.zeros((esn.reservoir_size, 2)))
        activation_text.set_text("")
        return train_line, train_pred_line, test_line, test_pred_line, step_text, scatter, activation_text

    def update(frame):
      if frame < len(train_targets):
          # Training phase
          step_text.set_text(f"Training Phase: Step {frame + 1}/{len(train_targets)}")
          train_line.set_data(np.arange(frame + 1), train_targets[:frame + 1].flatten())
          train_predictions = esn.predict(train_inputs[:frame + 1])
          train_pred_line.set_data(np.arange(frame + 1), train_predictions.flatten())

          input_data = train_inputs[frame]
          activations = esn.forward(input_data).flatten()

          # Filter active and inactive neuron indices
          active_indices = np.where((activations < -0.1) | (activations > 0.1))[0]
          inactive_indices = np.where((activations >= -0.1) & (activations <= 0.1))[0]

          # Create offsets and colors for scatter plot
          active_offsets = np.c_[active_indices, activations[active_indices]]
          inactive_offsets = np.c_[inactive_indices, activations[inactive_indices]]

          offsets = np.vstack((active_offsets, inactive_offsets))
          colors = ['green'] * len(active_indices) + ['black'] * len(inactive_indices)

          scatter.set_offsets(offsets)
          scatter.set_color(colors)

          num_active_neurons = len(active_indices)
          num_inactive_neurons = esn.reservoir_size - num_active_neurons

          activation_text.set_text(f"Step {frame + 1}\n"
                                    f"Active: {num_active_neurons}\n"
                                    f"Inactive: {num_inactive_neurons}")

      elif frame < len(train_targets) + len(test_targets):
          # Testing phase
          test_frame = frame - len(train_targets)
          step_text.set_text(f"Testing Phase: Step {test_frame + 1}/{len(test_targets)}")
          test_line.set_data(np.arange(test_frame + 1), test_targets[:test_frame + 1].flatten())
          test_predictions = esn.predict(test_inputs[:test_frame + 1])
          test_pred_line.set_data(np.arange(test_frame + 1), test_predictions.flatten())

          input_data = test_inputs[test_frame]
          activations = esn.forward(input_data).flatten()

          # Filter active and inactive neuron indices
          active_indices = np.where((activations < -0.1) | (activations > 0.1))[0]
          inactive_indices = np.where((activations >= -0.1) & (activations <= 0.1))[0]

          # Create offsets and colors for scatter plot
          active_offsets = np.c_[active_indices, activations[active_indices]]
          inactive_offsets = np.c_[inactive_indices, activations[inactive_indices]]

          offsets = np.vstack((active_offsets, inactive_offsets))
          colors = ['green'] * len(active_indices) + ['black'] * len(inactive_indices)

          scatter.set_offsets(offsets)
          scatter.set_color(colors)

          num_active_neurons = len(active_indices)
          num_inactive_neurons = esn.reservoir_size - num_active_neurons

          activation_text.set_text(f"Step {test_frame + 1}\n"
                                    f"Active: {num_active_neurons}\n"
                                    f"Inactive: {num_inactive_neurons}")

      return train_line, train_pred_line, test_line, test_pred_line, step_text, scatter, activation_text


    total_frames = len(train_targets) + len(test_targets)
    ani = animation.FuncAnimation(
        fig, update, frames=total_frames, init_func=init, interval=interval, blit=True
    )

    writer = FFMpegWriter(fps=1000 // interval, metadata={"artist": "Matplotlib"})
    ani.save(output_file, writer=writer)
    print(f"Animation saved as {output_file}")
    plt.close(fig)


combined_animation(esn, train_inputs, train_targets, test_inputs, test_targets)
