import base64
import re
import sys

import numpy as np
import plotly.graph_objects as go
from sklearn.preprocessing import OneHotEncoder

from imageio import imread
from PIL import Image

from skimage.filters import threshold_mean
from matplotlib import pyplot as plt
import matplotlib.cm as cm


def get_user_inputs():
    """Gets user inputs for building neuron"""
    try:
        weights = tuple(
            float(x.strip()) for x in input("Enter synapses weights: ").split(",")
        )

        activation_function = input("Enter activation function: ")

        if activation_function in ("step", "linear"):
            step_limit = int(input("Enter step limit (default = 0): "))
        else:
            step_limit = None

        print("✅ Succesfully got user inputs")

        return weights, activation_function, step_limit
    except Exception as ex:
        print(f"🔴 Exception occured while getting user inputs.\n{ex}")
        raise sys.exit(1)


def plot_activation_function(function, n: int, step: int = 0.5):
    """Activation function visualization

    Args:
        function: activation function to visualize
        n (int): interval max range
        step (int, optional): numbers interval step. Defaults to 0.5.

    Raises:
        sys.exit: exits if error while visualization process
    """
    try:

        x = np.arange(-n, n + 1, step)
        res = [function(i) for i in np.arange(-n, n + 1, step)]

        if function.__name__ in ("step"):
            shape = "hv"
        elif function.__name__ in ("linear"):
            shape = "linear"
        else:
            shape = "spline"

        fig = go.Figure(
            go.Scatter(x=x, y=res, line=dict(color="#f88f01", width=2, shape=shape))
        )

        # https://plotly.com/python/line-charts/
        fig.update_layout(
            title=f"Activation function {function.__name__} visualization",
            xaxis_title="Input parameters",
            yaxis_title="Function output",
            xaxis=dict(
                showline=True,
                showgrid=False,
                showticklabels=True,
                linecolor="rgb(204, 204, 204)",
                linewidth=2,
                ticks="outside",
                tickfont=dict(family="Arial", size=12, color="rgb(82, 82, 82)",),
            ),
            yaxis=dict(
                showgrid=False, zeroline=False, showline=False, showticklabels=False,
            ),
            autosize=False,
            margin=dict(autoexpand=False, l=100, r=20, t=110,),
            showlegend=False,
            plot_bgcolor="white",
        )

        fig.show()

    except Exception as ex:
        print(f"🔴 Exception occured while getting visualizing function.\n{ex}")
        raise sys.exit(1)


def initialize_weights(sizes: list, n_hidden: int) -> list:
    """Initializes neural network weights

    Args:
        sizes (list): nn size
            Example: [2, 4, 1]

    Returns:
        list: initialized weights
            Example: [0.08333, 0.16667, ... , 1.0]
    """

    assert isinstance(sizes, list)

    weights_limit = sum(x * y for x, y in zip(sizes, sizes[1:]))
    print(f"🔍 Number of weights: {weights_limit}")

    weights = [i / weights_limit for i in range(1, weights_limit + 1)]

    return (
        np.array(weights[: -n_hidden * sizes[-1]]).reshape(sizes[0], n_hidden),
        np.array(weights[-n_hidden * sizes[-1] :]).reshape(n_hidden, sizes[-1]),
    )


def accuracy_score(y_true, y_pred):
    """ Compare y_true to y_pred and return the accuracy """
    accuracy = np.sum(y_true == y_pred, axis=0) / len(y_true)
    return accuracy


def normalize(X, axis=-1, order=2):
    """ Normalize the dataset X """
    l2 = np.atleast_1d(np.linalg.norm(X, order, axis))
    l2[l2 == 0] = 1
    return X / np.expand_dims(l2, axis)


def onehot_encoding(x):
    """ One-hot encoding of nominal values """
    enc = OneHotEncoder(handle_unknown="ignore")
    return enc.fit_transform(x).toarray()


def convert_image(img_data):
    """Save img for future prediction"""

    imgstr = re.search(r"base64,(.*)", str(img_data)).group(1)

    with open("images/output.png", "wb") as output:
        output.write(base64.b64decode(imgstr))


def save_img(predicted: list):
    plt.imsave(
        "static/images/predicted.png", [reshape(d) for d in predicted][0], cmap=cm.gray
    )


def reshape(data):
    dim = int(np.sqrt(len(data)))
    data = np.reshape(data, (dim, dim))
    return data


def preprocessing(img):
    w, h = img.shape
    thresh = threshold_mean(img)
    binary = img > thresh
    shift = 2 * (binary * 1) - 1

    flatten = np.reshape(shift, (w * h))
    return flatten


def preprocess_image(image_data, shape):
    convert_image(image_data)

    x = imread("./images/output.png", pilmode="L")
    x = np.invert(x)
    x = np.array(Image.fromarray(x).resize(shape))

    return [preprocessing(d) for d in x.reshape(-1, shape[0], shape[1])][0]

