import csv
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path
from matplotlib.ticker import FuncFormatter
from scipy.stats import linregress

def list_directories(path: Path):
    return [ path for path in path.iterdir() if path.is_dir() ]

def list_files(path: Path):
    return [ file for file in path.iterdir() if path.is_file() ]

def read_train_loss(experiment_path: Path):
    file = experiment_path / "train_loss.csv"
    if not file.exists():
        return None
    return np.genfromtxt(file, delimiter=',')

def read_test_loss(experiment_path: Path):
    file = experiment_path / "test_loss.csv"
    if not file.exists():
        return None
    return np.genfromtxt(file, delimiter=',')

def power_series_regression(sample_counts, losses):
    # Do not consider samples <10^6 because training has not gotten up to speed yet
    start_index = np.where(sample_counts >= 1e6)[0][0]
    sample_counts = sample_counts[start_index:]
    losses = losses[start_index:]

    log_sample_counts = np.log(sample_counts)
    log_losses = np.log(losses)

    slope, intercept, r_value, p_value, std_err = linregress(log_sample_counts, log_losses)
    
    return slope, intercept


def plot_train_losses(name: str, experiment_path: Path, xaxis):
    train_logs = read_train_loss(experiment_path)
    if train_logs is None:
        print(f"Train logs not found for {name}")
        return

    steps = train_logs[:, 0]
    losses = train_logs[:, 1]
    wall_clock = train_logs[:, 2]
    num_samples = train_logs[:, 3]
    learning_rates = None
    if train_logs.shape[1] > 4:
        # We have learning rate information
        learning_rates = train_logs[:, 4]

    xaxis.loglog(num_samples, losses, base=10, color="b", label="Train loss")

    xaxis.set_xlabel("# seen samples")
    xaxis.set_ylabel("Loss")
    xaxis.set_yticks(np.array([5, 3, 1, 0.5, 0.2, 0.1, 0.01]))
    xaxis.set_xlim(left=1e6)
    def custom_format(x, _pos):
        return f'{x:.2f}'
    xaxis.yaxis.set_major_formatter(FuncFormatter(custom_format))

    if learning_rates is not None:
        ax2 = xaxis.twinx()
        ax2.plot(num_samples, learning_rates, 'orange')
        ax2.set_ylabel("Learning rate")
        ax2.set_yticks(np.array([0, 0.0001, 0.0002, 0.0003, 0.0004, 0.001]))


def plot_test_losses(name: str, experiment_path: Path, xaxis):
    test_logs = read_test_loss(experiment_path)
    if test_logs is None:
        print(f"Test logs not found for {name}")
        return

    steps = test_logs[:, 0]
    losses = test_logs[:, 1]
    wall_clock = test_logs[:, 2]
    num_samples = test_logs[:, 3]

    slope, intercept = power_series_regression(num_samples, losses)
    trendline = np.exp(intercept) * (num_samples ** slope)
    xaxis.loglog(num_samples, trendline, 'g--', base=10, label=f'Trendline (y = {np.exp(intercept):.2f}x^{slope:.2f})')

    xaxis.loglog(num_samples, losses, base=10, color="r", label="Test loss")
    xaxis.set_xlabel("# seen samples")
    xaxis.set_ylabel("Loss")
    xaxis.set_yticks(np.array([5, 3, 1, 0.5, 0.2, 0.1, 0.01]))
    xaxis.set_xlim(left=1e6)
    def custom_format(x, _pos):
        return f'{x:.2f}'
    xaxis.yaxis.set_major_formatter(FuncFormatter(custom_format))

if __name__ == "__main__":
    base_path = Path("saved_checkpoints")
    experiment_paths = list_directories(base_path)

    for i, experiment_path in enumerate(experiment_paths):
        name = experiment_path.name
        fig, plot = plt.subplots()
        plt.title(f"Losses - {name}")
        plot_train_losses(name, experiment_path, plot)
        plot_test_losses(name, experiment_path, plot)
        plot.legend()

    plt.show()

