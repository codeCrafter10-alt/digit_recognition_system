import urllib.request
import numpy as np
import os

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