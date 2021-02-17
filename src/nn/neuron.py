import numpy as np
import sys
from typing import Tuple


class Neuron:
    def __init__(
        self, weights: Tuple[float, ...], activation_function: str, limit: int
    ):
        try:
            self.weights = weights
            self.input_nodes = np.arange(1, len(weights) + 1)
            self.activation_function = getattr(self, activation_function.lower())
            self.step_limit = limit if limit else 0
        except Exception as ex:
            print(f"🔴 Exception occured while initializing neuron.\n{ex}")
            raise sys.exit(1)

    def step(self, x: float) -> int:
        """Step function

        Args:
            x (float): input value

        Returns:
            int: calculated result (0/1)
        """

        return 1 * (x >= self.step_limit)

    # take a look at relu!?
    def linear(self, x: float) -> int:
        """Linear limited function

        Args:
            x (float): input value

        Returns:
            int: calculated result (0/x/1)
        """
        if x >= 0 and x < self.step_limit:
            return x

        return 0 if x < 0 else 1

    # https://en.wikipedia.org/wiki/Sigmoid_function
    def sigmoid(self, x: float, c: float = 1.0) -> float:
        """Sigmoid function

        Args:
            x (float): input value
            c (float, optional): sigmoid width coef. Defaults to 1.

        Returns:
            float: calculated result
        """

        return 1.0 / (1.0 + np.exp(c * -x))

    # https://en.wikipedia.org/wiki/Hyperbolic_functions
    def tanh(self, x: float) -> float:
        """Tanh function

        Args:
            x (float): input value

        Returns:
            float: [description]
        """

        # 2.0/(1.0+np.exp(2*(c * -x))) - 1.0
        return (np.exp(2.0 * x) - 1.0) / (np.exp(2.0 * x) + 1.0)

    def forward(self) -> float:
        """Performs neuron calculations

        Returns:
            float: calculated neuron result -> (activation_function(input nodes sum))
        """
        inputs_sum = np.dot(self.input_nodes, self.weights)
        neuron_output = self.activation_function(inputs_sum)
        return neuron_output
