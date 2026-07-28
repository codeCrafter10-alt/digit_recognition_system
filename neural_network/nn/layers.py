import numpy as np

class Dense:
    """
    A fully connected (dense) layer
    
    Parameters:
    input_size: int
        Number of input features
    output_size: int
        Number of neurons in this layer
    """

    def __init__(self, input_size, output_size):
        """ Initalize weights randomly and biases to 0"""
        self.weights = np.random.randn(input_size, output_size) * 0.01
        self.biases = np.zeros(output_size)
    
    def forward(self, x):
        """
        Compute forward pass: output = x @ weights + biases
        
        Parameters:
        x: numpy array with shape (batch_size, input_size)
            Input to this layer
        
        Returns:
        numpy array with shape (batch_size, output_size)
        """
        self.input = x
        return x @ self.weights + self.biases

    def backprop(self, incoming_grad):
        """
        Compute backward pass and store gradients

        Parameters:
        incoming_grad: numpy array with shape (batch_size, output_size)
            Gradient flowing in from the next layer
        
        Returns:
            numpy array with shape (batch_size, input_size)
                Gradient to pass to the previous layer
        """
        self.d_weights = self.input.T @ incoming_grad
        self.d_biases = incoming_grad.sum(axis=0)
        dx = incoming_grad @ self.weights.T
        return dx