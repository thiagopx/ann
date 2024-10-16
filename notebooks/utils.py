import numpy as np
from sklearn.datasets import make_blobs
import matplotlib.pyplot as plt


def min_max_scaler(X, min_val=None, max_val=None):
    """Min-max scaler.

    Args:
        X: input data matrix (n_samples, n_features)
        min_val: minimum value for scaling
        max_val: maximum value for scaling

    Returns:
        X_scaled: scaled data matrix
        min_val: minimum value for scaling
        max_val: maximum value for scaling
    """
    if min_val is None:
        min_val = X.min(axis=0)
    if max_val is None:
        max_val = X.max(axis=0)
    X_scaled = (X - min_val) / (max_val - min_val)
    return X_scaled, min_val, max_val


def standard_scaler(X, mean=None, std=None):
    """Standard scaler.

    Args:
        X: input data matrix (n_samples, n_features)
        mean: mean value for scaling
        std: standard deviation for scaling

    Returns:
        X_scaled: scaled data matrix
        mean: mean value for scaling
        std: standard deviation for scaling
    """
    if mean is None:
        mean = X.mean(axis=0)
    if std is None:
        std = X.std(axis=0)
    X_scaled = (X - mean) / std
    return X_scaled, mean, std


def onehot(y, num_classes):
    """
    Convert integer labels to one-hot encoding.
    """
    y_onehot = np.zeros((len(y), num_classes))
    y_onehot[np.arange(len(y)), y] = 1
    return y_onehot


def make_linear_regression_dataset(
    n_samples=100, x_min=-10, x_max=10, w=None, noise=0.1, random_state=None
):
    x = np.random.uniform(x_min, x_max, n_samples)
    w = np.random.rand() if w is None else w
    b = np.random.rand()
    y = w * x + b + noise * np.random.randn(n_samples)
    return x, y


def make_classification_dataset(
    n_samples=100, n_features=2, n_classes=2, cluster_std=1.5, random_state=None
):
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


def plot_loss_history(loss_history, iterations=None, interval=1):
    if iterations is None:
        iterations = np.arange(len(loss_history))
    plt.plot(iterations, loss_history)
    plt.xticks(iterations[::interval])
    plt.xlabel("Iteration")
    plt.ylabel("Loss")


def plot_classification_dataset_2D(
    X,
    y,
    negative_label=-1,
    ax=None,
    colors=["orange", "blue"],
    alpha=1,
    labels=["negative class", "positive class"],
    loc="upper right",
):
    if ax is None:
        ax = plt.gca()

    # Retrieve the existing legend handles and labels
    handles_ax, labels_ax = ax.get_legend_handles_labels()

    X_neg = X[y == negative_label]
    X_pos = X[y == 1]
    X = np.vstack([X_neg, X_pos])
    scatter_neg = ax.scatter(
        X_neg[:, 0], X_neg[:, 1], c=colors[0], edgecolors="gray", alpha=alpha, label=labels[0]
    )
    scatter_pos = ax.scatter(
        X_pos[:, 0], X_pos[:, 1], c=colors[1], edgecolors="gray", alpha=alpha, label=labels[1]
    )
    ax.set_xlabel("$x_1$")
    ax.set_ylabel("$x_2$")

    labels = labels_ax + labels
    handles_ax += [scatter_neg, scatter_pos]
    ax.legend(handles=handles_ax, labels=labels + labels_ax, loc=loc)


def plot_decision_boundary_2D(X, w, ax=None, colors=None):
    if colors is None:
        colors = ["orange", "blue"]

    x1 = np.array([X[:, 0].min(), X[:, 0].max()])
    x2 = (-w[2] - w[0] * x1) / w[1]

    # Switch colors if the positive class is above the decision boundary
    x_above = np.array([x1[0], x2[0] + 1, 0])
    if np.dot(x_above, w) > 0:
        colors = colors[::-1]

    if ax is None:
        ax = plt.gca()

    # Plot the decision boundary line
    ax.plot(x1, x2, color="black")

    # Fill the region above the decision boundary (shade in one color)
    ax.fill_between(x1, x2, max(x2.max(), X[:, 1].max()), color=colors[0], alpha=0.2)

    # Fill the region below the decision boundary (shade in another color)
    ax.fill_between(x1, min(x2.min(), X[:, 1].min()), x2, color=colors[1], alpha=0.2)


def plot_xor(ax=None, colors=None):
    if ax is None:
        ax = plt.gca()

    if colors is None:
        colors = ["orange", "blue"]

    ax.scatter([0, 1], [0, 1], color=colors[0], edgecolors="gray")
    ax.scatter([0, 1], [1, 0], color=colors[1], edgecolors="gray")
    ax.set_xlabel("$x_1$")
    ax.set_ylabel("$x_2$")
    ax.legend(["0", "1"])
