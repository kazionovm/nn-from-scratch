### The goal is to build MultiLayerPerceptron
### An MLP consists of at least three layers of nodes: an input layer, a hidden layer and an output layer.

from re import X
from typing import Tuple

import joblib
import numpy as np
from sklearn import datasets
import joblib
from sklearn.model_selection import train_test_split
from tqdm import tqdm

from src.nn.functions import Sigmoid, Softmax, SquareLoss
from src.utils.general import (
    accuracy_score,
    initialize_weights,
    normalize,
    onehot_encoding,
)

from imageio import imread
from PIL import Image


class MultiLayerPerceptron:
    def __init__(
        self,
        sizes: Tuple[int, int],
        n_iterations: int = 3000,
        learning_rate: float = 0.01,
    ):
        self.sizes = sizes
        self.n_hidden = sum(sizes[1:-1])
        self.n_iterations = n_iterations
        self.learning_rate = learning_rate
        self.activation_function = Sigmoid()
        self.output_function = Softmax()
        self.loss_function = SquareLoss()

    def _forward(self, X):
        assert X.shape[-1] == self.h_weights.shape[0]

        self.hidden_input = X.dot(self.h_weights)
        self.hidden_output = self.activation_function(self.hidden_input)

        self.output_layer_input = self.hidden_output.dot(self.out_weights)
        y_pred = self.output_function(self.output_layer_input)

        return y_pred

    def _backprop(self, X, y, y_pred):
        grad_out = self.loss_function.gradient(
            y, y_pred
        ) * self.output_function.gradient(self.output_layer_input)
        grad_out_weights = self.hidden_output.T.dot(grad_out)

        grad_hidden = grad_out.dot(
            self.out_weights.T
        ) * self.activation_function.gradient(self.hidden_input)
        grad_hidden_weights = X.T.dot(grad_hidden)

        self.out_weights -= self.learning_rate * grad_out_weights
        self.h_weights -= self.learning_rate * grad_hidden_weights

    def fit(self, X, y):
        self.h_weights, self.out_weights = initialize_weights(self.sizes, self.n_hidden)

        for _ in tqdm(range(self.n_iterations)):
            y_pred = self._forward(X)
            self._backprop(X, y, y_pred)

    def predict(self, X):
        return self._forward(X)


def train():
    data = datasets.load_digits()
    X = normalize(data.data)
    y = data.target

    y = onehot_encoding(y.reshape(-1, 1))

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.4)

    clf = MultiLayerPerceptron([X.shape[1], 16, y.shape[1]], 16000, learning_rate=0.01)

    clf.fit(X_train, y_train)
    y_pred = np.argmax(clf.predict(X_test), axis=1)
    y_test = np.argmax(y_test, axis=1)

    accuracy = accuracy_score(y_test, y_pred)
    print("Accuracy:", accuracy)

    joblib.dump(clf, "src/mlp_pretrained.pkl")

    print("✅ Succesfully saved fitted model")


def predict_digit(X):
    clf = joblib.load("src/mlp_pretrained.pkl")
    return np.argmax(clf.predict(X), axis=1)


if __name__ == "__main__":
    # train()

    x = imread("./images/output.png", pilmode="L")
    x = np.invert(x)
    x = np.array(Image.fromarray(x).resize((8, 8)))

    x = normalize(x.reshape(-1))

    print(predict_digit(x))
