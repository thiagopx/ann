import numpy as np
from sklearn.datasets import make_blobs
import matplotlib.pyplot as plt


def make_linear_regression_dataset(n_samples=100, x_min=-10, x_max=10, w=None, noise=0.1, random_state=None):
    x = np.random.uniform(x_min, x_max, n_samples)
    w = np.random.rand() if w is None else w
    b = np.random.rand()
    y = w * x + b + noise * np.random.randn(n_samples)
    return x, y


def make_classification_dataset(n_samples=100, n_features=2, n_classes=2, cluster_std=1.5, random_state=None):
    X, y = make_blobs(
        n_samples=n_samples,
        centers=n_classes,
        cluster_std=cluster_std,
        n_features=n_features,
        random_state=random_state,
    )
    return X, y


def plot_regression_dataset_1D(x, y):
    plt.scatter(x, y)
    plt.xlabel("$x$")
    plt.ylabel("$y$")


def plot_loss_history(loss_history, iterations=None):
    if iterations is None:
        iterations = np.arange(len(loss_history))
    plt.plot(iterations, loss_history)
    plt.xlabel("Iteration")
    plt.ylabel("Loss")


def plot_classification_dataset_2D(X, y, negative_label=-1):
    plt.scatter(X[y == negative_label][:, 0], X[y == negative_label][:, 1], color="orange")
    plt.scatter(X[y == 1][:, 0], X[y == 1][:, 1], color="blue")
    plt.xlabel("$x_1$")
    plt.ylabel("$x_2$")
    plt.legend(["negative class", "positive class"])
