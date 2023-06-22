import numpy as np
import dataclasses
from sklearn import datasets
from typing import List, Callable, Tuple
from matplotlib import pyplot as plt


def plot_labels(inputs: np.ndarray, labels: np.ndarray):

    plt.scatter(inputs[labels == 0, 0],
                inputs[labels == 0, 1], label="class 0")

    plt.scatter(inputs[labels == 1, 0],
                inputs[labels == 1, 1], label="class 1")

    plt.legend()
    plt.show()


def plot_predictions(inputs: np.ndarray, true_labels: np.ndarray, predicted_labels: np.ndarray):
    plt.title('True labels')
    plot_labels(inputs, true_labels)
    plt.title('Predicted labels')
    plot_labels(inputs, predicted_labels)


def main():
    inputs, labels = datasets.make_moons(100)
    plot_labels(inputs, labels)


if __name__ == "__main__":
    main()
