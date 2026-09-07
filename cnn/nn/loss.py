import numpy as np

class CrossEntropy:
    """
    Cross entropy loss function
    Measures how wrong the model's predictions are (Error function)
    """

    def forward(self, predictions, true_labels):
        """
        Compute the cross entropy loss of the current model

        Parameters:
        predictions: numpy array with shape (batch_size,10)
            The model's predictions as an array of probabilites
        
        true_labels: numpy array with shape (batch_size, 10)
            The correct value with that value having 1 and other digits having 0
        
        Returns:
        float
            The average loss over the batch
        """
        predictions = np.clip(predictions, 1e-10, 1.0) # Prevent log(0)
        return -np.sum(true_labels * np.log(predictions)) / predictions.shape[0]

    def backward(self, predictions, true_labels):
        """
        Compute the gradient of the loss to start backpropagation

        predictions: numpy array with shape (batch_size,10)
            The model's predictions as an array of probabilites
        
        true_labels: numpy array with shape (batch_size, 10)
            The correct value with that value having 1 and other digits having 0
        
        Returns:
        numpy array with shape (batch_size, 10)
            Gradient to pass to the previous layer
        """
        predictions = np.clip(predictions, 1e-10, 1.0) # Prevent log(0)
        return (predictions - true_labels) / predictions.shape[0]
