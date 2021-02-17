import numpy as np
import plotly.graph_objects as go
import sys


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
