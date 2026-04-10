# Digit Recognition System

A web app that identifies handwritten digits using a three-layer neural network built from scratch with numpy (no TensorFlow or PyTorch).

## Live Demo
### [Demo Web App](https://codeaarav10-digit-recognizer.hf.space)

![App Screenshot](assets/screenshot.png)

### How to Use
1. Draw any digit (0-9) on the canvas
2. Click **Predict** to see the result
3. Click **Clear** to reset and try again

## Overview
This project builds a handwritten digit recognition system entirely from scratch. The neural network is implemented using only numpy. Every component including the forward pass, activation functions, loss computation, backpropagation, and weight updates is written manually without any machine learning frameworks.

The network is trained on the MNIST dataset and learns to classify 28x28 pixel images of handwritten digits into one of 10 classes (0-9). To improve robustness on real handwritten input (which looks different from clean MNIST images), random augmentations are applied to every training image on every epoch.

The trained model is served through a Flask web app where users draw a digit on a canvas, the image is preprocessed and passed through the network, and the predicted digit is returned in real time. The full pipeline (from raw pixel values to a prediction) runs in under a second.

The project also includes cross-validation to find optimal hyperparameters and a benchmarking script to visualize the training loss curve.

## Technical Highlights
- Neural network implemented entirely from scratch using only numpy, without ML frameworks
- Full backpropagation pipeline written manually, including gradient computation and weight updates
- Cross-validation used to find optimal hyperparameters (learning rate, batch size, epochs)
- Data augmentation pipeline with random shifts, rotations, blur, and noise applied per epoch
- Deployed as a Flask web app with a canvas drawing interface, deployed on Hugging Face Spaces

## Dataset
The model is trained on [MNIST](http://yann.lecun.com/exdb/mnist/) — one of the most well-known datasets in machine learning. It contains 70,000 grayscale images of handwritten digits:
- 60,000 training images
- 10,000 test images
- Each image is 28x28 pixels, flattened to a vector of 784 values
- Pixel values are normalized from [0, 255] to [0, 1] before training

MNIST is automatically downloaded the first time you run `train.py` — no manual setup required.

## Results
- Test accuracy: **98.52%**
- Training: 30 epochs with data augmentation

![Loss Curve](assets/loss_curve.png)

## How It Works
1. User draws a digit on a canvas
2. The image is sent to a Flask backend as a base64 string
3. The image is preprocessed — converted to grayscale, resized to 28x28, normalized to [0, 1]
4. A three-layer neural network predicts the digit
5. The prediction is returned and displayed instantly

## Architecture
- Input layer: 784 neurons (28x28 pixels)
- Hidden layer 1: 128 neurons, ReLU activation
- Hidden layer 2: 64 neurons, ReLU activation
- Output layer: 10 neurons, Softmax activation

## Data Augmentation
To make the model robust to real-world handwriting, each training image is randomly augmented during training:
- **Random shift** — moves the digit up to 3 pixels in any direction
- **Random rotation** — rotates up to 15 degrees
- **Gaussian blur** — softens edges to simulate different pen pressures
- **Gaussian noise** — adds random pixel noise to prevent overfitting

Each augmentation is applied independently with 50% probability per image per epoch, meaning the model rarely sees the same image twice.

## Dependencies
| Library | Purpose |
|---|---|
| numpy | Neural network, forward/backward pass |
| Flask | Web backend |
| Pillow | Image preprocessing |
| scipy | Data augmentation |
| matplotlib | Loss curve plot |

## Setup

### 1. Clone the repository
```bash
git clone https://github.com/codeAarav10/digit_recognition_system
cd digit_recognition_system
```

### 2. Install dependencies
Install all required Python packages:
```bash
pip install -r requirements.txt
```

### 3. Train the model
This downloads MNIST, trains the network for 30 epochs, and saves the weights to `model/weights.pkl`:
```bash
python train.py
```

### 4. (Optional) Generate loss curve
Plots and saves the training loss curve to `assets/loss_curve.png`:
```bash
python benchmark.py
```

### 5. Run the app
Starts the Flask server locally at `http://127.0.0.1:5000`:
```bash
python app.py
```