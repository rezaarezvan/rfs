import numpy as np
import dataclasses

from sklearn import datasets
from matplotlib import pyplot as plt
from typing import List, Callable, Tuple

LEARNING_RATE = 1e-3
NUM_EPOCHS = 300


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


def _relu(inputs: np.ndarray) -> np.ndarray:
    return np.maximum(inputs, np.zeros_like(inputs))


def _softmax(inputs: np.ndarray) -> np.ndarray:
    exp_inputs = np.exp(inputs - np.max(inputs))
    return exp_inputs / np.sum(exp_inputs)


def _relu_derivative(inputs: np.ndarray) -> np.ndarray:
    return np.where(inputs > 0, np.ones_like(inputs), np.zeros_like(inputs))


def _softmax_derivative(inputs: np.ndarray) -> np.ndarray:
    softmax = _softmax(inputs)
    return softmax * (1 - softmax)


def _get_activation(function_name: str) -> Callable[[np.ndarray], np.ndarray]:
    if function_name == 'relu':
        return _relu
    elif function_name == 'softmax':
        return _softmax
    else:
        raise ValueError(f'Unknown function_name: {function_name=}.')


def _get_activation_derivative(function_name: str) -> Callable[[np.ndarray], np.ndarray]:
    if function_name == 'relu':
        return _relu_derivative
    elif function_name == 'softmax':
        return _softmax_derivative
    else:
        raise ValueError(f'Unknown function_name: {function_name=}.')


def cross_entropy(prediction: np.ndarray, label: np.ndarray) -> float:
    return -1 * np.log(prediction)[label]


def cross_entropy_derivative(prediction: np.ndarray, label: int) -> float:
    return -1. / prediction[label]


def mean_squared_error(prediction: np.ndarray, label: int) -> float:
    one_hot = np.zeros_like(prediction)
    one_hot[label] = 1
    return np.mean((prediction - one_hot) ** 2)


def mean_squared_error_derivative(prediction: np.ndarray, label: int) -> float:
    one_hot = np.zeros_like(prediction)
    one_hot[label] = 1
    return prediction - one_hot


@dataclasses.dataclass
class LayerParams:
    weights: np.ndarray
    biases: np.ndarray
    activation: Callable[[np.ndarray], np.ndarray]


def _glorot_scale(in_dims: int, out_dims: int) -> float:
    return np.sqrt(2 / (in_dims + out_dims))


def _initialize_weights(in_dims: int, out_dims: int) -> np.ndarray:
    return np.random.normal(loc=0.,
                            scale=_glorot_scale(in_dims, out_dims),
                            size=(in_dims, out_dims))


def init_params(sizes: List[int]) -> List[LayerParams]:
    params = []
    for size_index in range(len(sizes) - 1):
        in_dims = sizes[size_index]
        out_dims = sizes[size_index + 1]
        if size_index == len(sizes) - 2:
            activation = 'softmax'
        else:
            activation = 'relu'
        params.append(LayerParams(weights=_initialize_weights(in_dims, out_dims),
                                  biases=0.01 * np.ones(out_dims),
                                  activation=activation))
    return params


def forward_with_activations(
        params: List[LayerParams],
        image: np.ndarray) -> Tuple[np.ndarray, List[np.ndarray]]:
    layer_input = image
    activations = []
    for _, layer_params in enumerate(params):
        layer_output = np.matmul(layer_input, layer_params.weights)
        layer_output += layer_params.biases
        layer_output = _get_activation(layer_params.activation)(layer_output)
        layer_input = layer_output
        activations.append(layer_output)
    return layer_output, activations


def backward(params: List[LayerParams], image: np.ndarray,
             label: int) -> List[LayerParams]:
    probabilities, activations = forward_with_activations(params, image)
    accumulated_gradient = mean_squared_error_derivative(probabilities, label)
    new_params = [None for _ in params]
    for reverse_layer_index, layer_params in enumerate(reversed(params)):

        layer_index = len(params) - 1 - reverse_layer_index
        activation_derivative_func = _get_activation_derivative(
            layer_params.activation)
        prev_activation = activations[layer_index - 1]
        activation_derivative = activation_derivative_func(
            activations[layer_index])
        bias_gradient = accumulated_gradient * activation_derivative
        weight_gradient = np.outer(prev_activation, bias_gradient.T)

        new_biases = layer_params.biases - LEARNING_RATE * bias_gradient
        new_weights = layer_params.weights - LEARNING_RATE * weight_gradient
        new_params[layer_index] = LayerParams(weights=new_weights,
                                              biases=new_biases,
                                              activation=layer_params.activation)
        accumulated_gradient = np.dot(bias_gradient, layer_params.weights.T)
    return new_params


def calculate_loss(params: List[LayerParams], image: np.ndarray, label: int) -> float:
    prediction, _ = forward_with_activations(params, image)
    return cross_entropy(prediction, label)


def accuracy(params: List[LayerParams],
             images: np.ndarray, labels: np.ndarray) -> float:
    num_correct = 0
    for data_index in range(len(labels)):
        image, label = images[data_index, :], labels[data_index]
        prediction = np.argmax(forward_with_activations(params, image)[0])
        num_correct += prediction == label
    return num_correct / len(images)


def predict_labels(params: List[LayerParams], inputs: np.ndarray):
    predicted_labels = []
    for image in inputs:
        probabilities, _ = forward_with_activations(params, image)
        prediction = np.argmax(probabilities)
        predicted_labels.append(prediction)
    return np.array(predicted_labels)


def main():
    inputs, labels = datasets.make_moons(100)
    params = init_params([2, 128, 128, 128, 128, 2])

    for epoch_index in range(NUM_EPOCHS):
        loss = sum([calculate_loss(params, inputs[data_index, :], labels[data_index])
                   for data_index in range(len(labels))])/len(labels)
        print(
            f'Running epoch {epoch_index}, mean loss: {loss}, accuracy: {accuracy(params, inputs, labels)}')
        for data_index in range(len(labels)):
            image, label = inputs[data_index, :], labels[data_index]
            params = backward(params, image, label)

    plot_predictions(inputs, labels, predict_labels(params, inputs))


if __name__ == "__main__":
    main()
