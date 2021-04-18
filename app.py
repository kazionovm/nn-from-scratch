import os
import sys

sys.path.append(os.path.abspath("./src"))

from flask import Flask, render_template, request

from src.mlp import *
from src.utils.general import preprocess_image, save_img
from src.nn.hopfield import HopfieldNetwork

app = Flask(__name__)

data = []


@app.route("/")
def index():
    return render_template("index.html")


@app.route("/train/", methods=["POST"])
def train():
    image_data = request.get_data()
    x = preprocess_image(image_data, (100, 100))
    data.append(x)

    return "Successfuly saved training sample"


@app.route("/predict/", methods=["POST"])
def predict():
    model = HopfieldNetwork()

    image_data = request.get_data()
    model.fit(data)

    x = [preprocess_image(image_data, (100, 100))]
    predicted = model.predict(x, num_iter=100, threshold=25)
    save_img(predicted)

    return "<img src='static/images/predicted.png' alt='Predicted'>"


if __name__ == "__main__":
    app.run(debug=False, threaded=False)

