class Node:
    """
    Represents a single value in the computational graph

    Attributes:
    data: float
        The value of this node
    grad: float
        The gradient of the loss with respect to this node.
        Accumulated during backpropagation
    """
    def __init__(self, value):
        # Initialize node with a value and zero gradient
        self.data = value
        self.grad = 0

    def backprop(self, incoming_grad):
        # Accumulate incoming gradient during backpropagation
        self.grad += incoming_grad

    def null_gradients(self):
        # Reset gradients to zero before each training step
        self.grad = 0