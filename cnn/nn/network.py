import numpy as np
import pickle

class Network:
    """
    Fully connected neural network that stacks layers.
    Runs forward and backward passes, update, predict, and save and load weights
    """

    def __init__(self, layers):
        """ Initialize the network with a list of layers"""
        self.layers = layers

    def forward(self, x):
        """
        Run forward pass through all layers

        Parameters:
        x: numpy array with shape (batch_size, neurons)
            Input data
        
        Returns:
        numpy array with shape (batch_size, 10)
            Outputs the last layer (the probabilities of each digit)
        """
        output = x
        for layer in self.layers:
            output = layer.forward(output)
        
        return output

    def backward(self, grad):
        """
        Run backward pass over the full network of layers

        Parameters:
        grad: numpy array with shape (batch_size, 10)
            Gradient from the loss function

        Returns:
        None
        """
        for layer in reversed(self.layers):
            grad = layer.backprop(grad)


    def update(self, learning_rate):
        """
        Update the model's weights and biases

        Parameters:
        learning_rate: float
            Step size when updating the weights and biases

        Returns:
        None
        """
        for layer in self.layers:
            if hasattr(layer, 'd_weights'):
                layer.weights -= learning_rate * layer.d_weights
                layer.biases -= learning_rate * layer.d_biases

    def predict(self, x):
        """
        Predict the label of the image based on the current model

        Parameters:
        x: numpy array with shape (batch_size, 784)
            Input data of the images

        Returns:
        numpy array with shape (batch_size,)
            Predicted digit (0-9) for each image
        """
        output = self.forward(x)
        return np.argmax(output, axis=1)

    def save_weights(self, filepath):
        """
        Save the updated weights to the file

        Parameters:
        filepath: string
            Filepath to the file where weights are to be stored

        Returns:
        None
        """
        with open(filepath, 'wb') as f:
            pickle.dump(self.layers, f)

    def load_weights(self, filepath):
        """
        Load the model's weights from the file

        Parameters:
        filepath: string
            Filepath to the file where weights are stored
        
        Returns:
        None
        """
        with open(filepath, 'rb') as f:
            self.layers = pickle.load(f)