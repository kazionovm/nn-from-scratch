import argparse
from nn.neuron import Neuron
from utils.general import *

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Simulates neuron")

    parser.add_argument(
        "--visualize",
        action="store_true",
        help="Visualize activation function",
        default=False,
    )

    args = parser.parse_args()

    try:
        neu = Neuron(*get_user_inputs())
        print(f"🔍 Neuron (axon) output: {neu.forward()}")

        if args.visualize:
            try:
                function = getattr(
                    neu, input("\n🖼 Choose activation function to-visualize: ").lower()
                )
                print("✅Success\n")
                plot_activation_function(function, 100, 1)

            except Exception as ex:
                print(
                    f"🔴 Exception occured while visualizing activation function.\n{ex}"
                )
                raise sys.exit(1)

    except Exception as ex:
        print(f"🔴 Exception occured while executing main.py\n{ex}")
        raise sys.exit(1)

