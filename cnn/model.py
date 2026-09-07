from cnn.nn.layers import Conv2D, MaxPool2D, Flatten, Dense
from cnn.nn.activations import ReLU, Softmax
from cnn.nn.network import Network


def create_model():
    """
    Create and return the CNN used for MNIST classification.
    """

    return Network([
        Conv2D(
            in_channels=1,
            out_channels=8,
            kernel_size=3
        ),

        ReLU(),

        MaxPool2D(
            pool_size=2,
            stride=2
        ),

        Conv2D(
            in_channels=8,
            out_channels=16,
            kernel_size=3
        ),

        ReLU(),

        MaxPool2D(
            pool_size=2,
            stride=2
        ),

        Flatten(),

        Dense(
            input_size=400,
            output_size=128
        ),

        ReLU(),

        Dense(
            input_size=128,
            output_size=10
        ),

        Softmax()
    ])