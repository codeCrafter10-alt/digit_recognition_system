import numpy as np

class Network:
    """
    Fully connected neural network taht stacks layers.
    Runs forward and backward passes.
    """

    def __init__(self, layers):
        """ Initialize the network with a list of layers"""
        self.layers = layers

    def forward(self, x):
        """
        Run forward pass through all layers

        Parameters:
        x: numpy array with shape (batch_size, 784)
            Input data
        
        Returns:
        numpy array with shape (batch_size, 10)
            Output of the last layer (the probabilities of each digit)
        """
        pass

    def backward(self, grad, true_labels):
        """
        Run backward pass over the full network of layers

        Parameters:
        grad: numpy array with shape (batch_size, 10)
            Gradient from the loss function
        
        true_labels: numpy array with shape (batch_size, 10)
            The correct labels for each image

        Returns:
        None
        """
        pass

    def update(self, learning_rate):
        """
        Update the model's weights and biases

        Parameters:
        learning_rate: float
            Step size when updating the weights and biases

        Returns:
        None
        """
        pass

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
        pass

    def save_weights(self, filepath):
        """
        Save the updated weights to the file

        Parameters:
        filepath: string
            Filepath to the file where weights are to be stored

        Returns:
        None
        """
        pass

    def load_weights(self, filepath):
        """
        Load the model's weights from the file

        Parameters:
        filepath: string
            Filepath to the file where weights are stored
        
        Returns:
        None
        """
        pass