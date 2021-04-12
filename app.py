import os
import sys

sys.path.append(os.path.abspath("./src"))

import numpy as np
from flask import Flask, render_template, request

from imageio import imread
from PIL import Image

from src.mlp import *
from src.utils.general import normalize, convertImage

app = Flask(__name__)


@app.route("/")
def index():
    return render_template("index.html")


@app.route("/predict/", methods=["POST"])
def predict():
    image_data = request.get_data()
    convertImage(image_data)

    x = imread("./images/output.png", pilmode="L")
    x = np.invert(x)
    x = np.array(Image.fromarray(x).resize((8, 8)))

    x = normalize(x.reshape(-1))

    return "{}".format(*predict_digit(x))


if __name__ == "__main__":
    app.run(debug=False, threaded=False)

