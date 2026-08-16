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

class Flatten:
    """
    Flatten each sample from (channels, height, width)
    into a single vector.
    """

    def forward(self, x):
        """
        Parameters:
        x: (batch_size, channels, height, width)

        returns:
            (batch_size, channels * height * width)
        """
        self.input_shape = x.shape

        batch_size = x.shape[0]
        return x.reshape(batch_size, -1)

    def backprop(self, incoming_grad):
        """
        Reshape gradient back to the original input shape.
        """
        return incoming_grad.reshape(self.input_shape)

class MaxPool2D:
    """
    2D max pooling layer.

    Default:
        pool_size = 2
        stride = 2
    """

    def __init__(self, pool_size=2, stride=2):
        self.pool_size = pool_size
        self.stride = stride

    def forward(self, x):
        """
        Parameters:
        x: (batch_size, channels, height, width)

        returns:
            (batch_size, channels, output_height, output_width)
        """
        self.input = x

        batch_size, channels, height, width = x.shape

        output_height = (height - self.pool_size) // self.stride + 1
        output_width = (width - self.pool_size) // self.stride + 1

        output = np.zeros(
            (batch_size, channels, output_height, output_width)
        )

        for n in range(batch_size):
            for c in range(channels):
                for i in range(output_height):
                    for j in range(output_width):

                        h_start = i * self.stride
                        h_end = h_start + self.pool_size

                        w_start = j * self.stride
                        w_end = w_start + self.pool_size

                        region = x[
                            n,
                            c,
                            h_start:h_end,
                            w_start:w_end
                        ]

                        output[n, c, i, j] = np.max(region)

        return output

    def backprop(self, incoming_grad):
        """
        Backpropagate gradient through max pooling.

        Parameters:
        incoming_grad: (batch_size, channels, output_height, output_width)
            Gradient from the next layer.

        Returns:
        (batch_size, channels, height, width)
            Gradient to pass to the previous layer.
        """
        x = self.input

        batch_size, channels, height, width = x.shape

        dx = np.zeros_like(x)

        output_height = incoming_grad.shape[2]
        output_width = incoming_grad.shape[3]

        for n in range(batch_size):
            for c in range(channels):
                for i in range(output_height):
                    for j in range(output_width):

                        h_start = i * self.stride
                        h_end = h_start + self.pool_size

                        w_start = j * self.stride
                        w_end = w_start + self.pool_size

                        region = x[
                            n,
                            c,
                            h_start:h_end,
                            w_start:w_end
                        ]

                        max_index = np.unravel_index(
                            np.argmax(region),
                            region.shape
                        )

                        dx[
                            n,
                            c,
                            h_start + max_index[0],
                            w_start + max_index[1]
                        ] += incoming_grad[n, c, i, j]

        return dx


class Conv2D:
    """
    2D convolution layer.

    Parameters:
    in_channels: int
        Number of input channels.

    out_channels: int
        Number of filters / output channels.

    kernel_size: int
        Height and width of the convolution kernel.

    stride: int
        Convolution stride.

    Returns:
    numpy array: (batch_size, out_channels, output_height, output_width)
    """

    def __init__(self, in_channels, out_channels, kernel_size, stride=1):
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride

        # He initialization
        scale = np.sqrt(2 / (in_channels * kernel_size * kernel_size))

        self.weights = np.random.randn(
            out_channels,
            in_channels,
            kernel_size,
            kernel_size
        ) * scale

        self.biases = np.zeros(out_channels)

    def forward(self, x):
        """
        x:
            (batch_size, in_channels, height, width)

        returns:
            (batch_size, out_channels,
             output_height, output_width)
        """

        self.input = x

        batch_size, _, height, width = x.shape

        output_height = (
            (height - self.kernel_size) // self.stride
        ) + 1

        output_width = (
            (width - self.kernel_size) // self.stride
        ) + 1

        output = np.zeros(
            (
                batch_size,
                self.out_channels,
                output_height,
                output_width
            )
        )

        for n in range(batch_size):
            for f in range(self.out_channels):

                kernel = self.weights[f]

                for i in range(output_height):
                    for j in range(output_width):

                        h_start = i * self.stride
                        h_end = h_start + self.kernel_size

                        w_start = j * self.stride
                        w_end = w_start + self.kernel_size

                        region = x[
                            n,
                            :,
                            h_start:h_end,
                            w_start:w_end
                        ]

                        output[n, f, i, j] = (
                            np.sum(region * kernel)
                            + self.biases[f]
                        )

        return output

    def backprop(self, incoming_grad):
        """
        Backpropagate through convolution.

        Returns:
            gradient with respect to input
        """

        x = self.input

        batch_size, _, height, width = x.shape

        dx = np.zeros_like(x)

        self.d_weights = np.zeros_like(self.weights)
        self.d_biases = np.zeros_like(self.biases)

        output_height = incoming_grad.shape[2]
        output_width = incoming_grad.shape[3]

        for n in range(batch_size):
            for f in range(self.out_channels):

                kernel = self.weights[f]

                for i in range(output_height):
                    for j in range(output_width):

                        h_start = i * self.stride
                        h_end = h_start + self.kernel_size

                        w_start = j * self.stride
                        w_end = w_start + self.kernel_size

                        region = x[
                            n,
                            :,
                            h_start:h_end,
                            w_start:w_end
                        ]

                        grad = incoming_grad[n, f, i, j]

                        # Gradient with respect to filter
                        self.d_weights[f] += region * grad

                        # Gradient with respect to input
                        dx[
                            n,
                            :,
                            h_start:h_end,
                            w_start:w_end
                        ] += kernel * grad

                        # Gradient with respect to bias
                        self.d_biases[f] += grad

        return dx