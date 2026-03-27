import urllib.request
import numpy as np
import os
from nn.network import Network
from nn.layers import Dense
from nn.activations import ReLU, Softmax
from nn.loss import CrossEntropy

def download_mnist():
    """
    Download MNIST dataset if not already present

    Returns:
    string
        Filepath to the downloaded .npz dataset file
    """
    url = "https://storage.googleapis.com/tensorflow/tf-keras-datasets/mnist.npz"
    filepath = "data/mnist.npz"

    if not os.path.exists("data"):
        os.makedirs("data")
    
    if not os.path.exists(filepath):
        print("Downloading MNIST Dataset...")
        urllib.request.urlretrieve(url, filepath)
        print("Done.")
    
    return filepath

def load_mnist(filepath):
    """
    Load the MNIST dataset and flatten images

    Parameters:
    filepath: string
        Filepath where the dataset is stored
    
    Returns:
    numpy array
        4 arrays (training images and labels, test images and labels)
    """
    data = np.load(filepath)
    x_train, y_train = data['x_train'], data['y_train']
    x_test, y_test = data['x_test'], data['y_test']

    # Flatten images
    x_train = x_train.reshape(-1, 784)
    x_test = x_test.reshape(-1, 784)

    # Normalize pixels with values 0-255 to 0-1
    x_train = x_train / 255
    x_test = x_test / 255

    # One-hot encoding (Convert each label to array of 10 probabilities)
    y_train_oh = np.zeros((len(y_train), 10))
    y_train_oh[np.arange(len(y_train)), y_train] = 1
    y_test_oh = np.zeros((len(y_test), 10))
    y_test_oh[np.arange(len(y_test)), y_test] = 1

    return x_train, y_train_oh, x_test, y_test_oh

def train(network, x_train, y_train, epochs, batch_size, learning_rate):
    """
    Train the network on the MNIST dataset.

    Parameters:
    network: Network
        The neural network to train.
    x_train: numpy array with shape (60000, 784)
        Training images.
    y_train: numpy array with shape (60000, 10)
        One-hot encoded training labels.
    epochs: int
        Number of full passes through the training data.
    batch_size: int
        Number of images per batch.
    learning_rate: float
        Step size for gradient descent.

    Returns:
    List
         Loss history
    """
    loss_history = []   
    loss_fn = CrossEntropy()

    for epoch in range(epochs):
        for i in range(0, len(x_train), batch_size):
            # Forward pass
            x_batch = x_train[i:i+batch_size]
            y_batch = y_train[i:i+batch_size]
            predictions = network.forward(x_batch)

            loss = loss_fn.forward(predictions, y_batch) # Compute loss

            # Backpropagation
            gradient = loss_fn.backward(predictions, y_batch)
            network.backward(gradient)
            network.update(learning_rate)

        loss_history.append(loss)
        print(f"Epoch {epoch+1} - Loss: {loss:.4f}")
    
    return loss_history

def main():
    # Download and load mnist
    filepath = download_mnist()
    x_train, y_train, x_test, y_test = load_mnist(filepath)

    network = Network([Dense(784, 128), ReLU(), Dense(128, 64), ReLU(), Dense(64, 10), Softmax()]) # Build our network

    # Train
    loss_history = train(network, x_train, y_train, epochs=20, batch_size=32, learning_rate=0.01)
    print("\n") # Add space after the losses printed during training

    # Save weights
    os.makedirs("model", exist_ok=True)
    network.save_weights("model/weights.pkl")

    # Predict on test set
    predictions = network.predict(x_test)
    true_labels = np.argmax(y_test, axis=1)
    accuracy = np.mean(predictions == true_labels)
    print(f"Test accuracy: {accuracy * 100:.2f}%")

if __name__ == "__main__":
    main()