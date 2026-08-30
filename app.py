from flask import Flask, request, jsonify, render_template
import base64
from io import BytesIO
from PIL import Image
import numpy as np
from neural_network.nn.network import Network
from neural_network.nn.layers import Dense
from neural_network.nn.activations import ReLU, Softmax
from cnn.model import create_model

app = Flask(__name__)

# Initalize network with weights
network = Network([Dense(784, 128), ReLU(), Dense(128, 64), ReLU(), Dense(64, 10), Softmax()])
network.load_weights("neural_network/model/weights.pkl")


model = create_model()
model.load_weights("cnn/model/weights.pkl")

@app.route("/")
def home():
    return render_template("index.html")

@app.route("/predict_nn", methods=["POST"])
def predict():
    """
    Receives a base64 encoded image from the frontend
    Preprocesses it, and returns the predicted digit.

    Returns:
    JSON Object
        Predicted digit (0-9)
    """

    data = request.json # Receive image

    # Preprocess
    image_base64 = data["image"].split(",")[-1] + "=="
    image_data = base64.b64decode(image_base64)
    image = Image.open(BytesIO(image_data))
    image = image.convert("L")
    image = image.resize((28,28))
    image = np.array(image)
    image = image.reshape(1,784)
    image = image/255

    # Reeturn predicted digit
    predicted_digit = int(network.predict(image)[0])
    return jsonify({"prediction": predicted_digit})


@app.route("/predict_cnn", methods=["POST"])
def predict_cnn():
    """
    Receives a base64 encoded image from the frontend
    Preprocesses it, and returns the predicted digit using CNN.

    Returns:
    JSON Object
        Predicted digit (0-9)
    """

    data = request.json # Receive image

    # Preprocess
    image_base64 = data["image"].split(",")[-1] + "=="
    image_data = base64.b64decode(image_base64)
    image = Image.open(BytesIO(image_data))
    image = image.convert("L")
    image = image.resize((28,28))
    image = np.array(image)
    image = image.reshape(1,1,28,28)
    image = image/255

    # Return predicted digit
    predicted_digit = int(model.predict(image)[0])
    
    return jsonify({"prediction": predicted_digit})

if __name__ == "__main__":
    app.run()