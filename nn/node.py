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
        self.data = value
        self.grad = 0