import numpy as np

class ReLU:
    """
    ReLU activation function applied to each element
    Makes negative values 0 and passes positive values unchanged
    """
    def forward(self, x):
        """
        Apply RelU activation: max(0,x)

        Parameters:
        x: numpy array
            Input array
        
        Returns:
        numpy array
            Same shape as input, with negative values zeroed out
        """
        self.input = x
        return np.maximum(0,x)

    def backprop(self, incoming_grad):
        """
        Compute backward pass for ReLU

        Parameters:
        incoming_grad: numpy array
            Gradient flowing in from the next layer
        
        Returns:
        numpy array
            Gradient passed through if input > 0 and 0 otherwise
        """
        return incoming_grad * (self.input > 0)

class Softmax:
    """
    Softmax activation function for the output layer
    Converts raw scores into probabilites that sum to 1
    """
    def forward(self, x):
        """
        Apply softmax activation

        Parameters:
        x: numpy array with shape (batch_size, 10)
            Raw score from the ouput layer
        
        Returns:
        numpy array with shape (batch_size, 10)
            Probabilities summing to 1 across each row
        """
        shifted = x - np.max(x, axis=1, keepdims=True) # Make values <= 0 to prevent overflow
        exp = np.exp(shifted)
        self.output = exp / np.sum(exp, axis=1, keepdims=True)
        return self.output

    def backprop(self, predictions, true_labels):
        """
        Backward pass for Softmax 

        Parameters:
        predictions: numpy array with shape (batch_size, 10)
            Softmax output probabilities
        true_labels: numpy array with shape (batch_size, 10)
            True labels for each image (supervised learning)
        
        Returns:
        numpy array with shape (batch_size, 10)
            Gradient to pass to previous layer
        """
        return predictions - true_labels