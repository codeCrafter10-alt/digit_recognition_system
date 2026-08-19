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
        scale = np.sqrt(2 / input_size)  
        self.weights = np.random.randn(input_size, output_size) * scale
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

        pool = self.pool_size
        stride = self.stride

        output_height = (
            (height - pool) // stride
        ) + 1

        output_width = (
            (width - pool) // stride
        ) + 1

        # Get all pooling regions at once.
        windows = np.lib.stride_tricks.sliding_window_view(
            x,
            (pool, pool),
            axis=(2, 3)
        )

        # Apply the pooling stride.
        windows = windows[
            :,
            :,
            ::stride,
            ::stride,
            :,
            :
        ]

        # Save the windows for backpropagation.
        self.windows = windows

        # Take the maximum value from each pooling region.
        output = np.max(
            windows,
            axis=(4, 5)
        )

        return output

    def backprop(self, incoming_grad):
        """
        Backpropagate gradient through max pooling.

        Parameters:
        incoming_grad:
            (batch_size, channels,
             output_height, output_width)

        Returns:
            (batch_size, channels, height, width)
        """

        x = self.input

        batch_size, channels, height, width = x.shape

        pool = self.pool_size
        stride = self.stride

        output_height = incoming_grad.shape[2]
        output_width = incoming_grad.shape[3]

        dx = np.zeros_like(x)

        # Find which position in each pooling window
        # contained the maximum value.
        max_positions = np.argmax(
            self.windows.reshape(
                batch_size,
                channels,
                output_height,
                output_width,
                pool * pool
            ),
            axis=4
        )

        # Put the incoming gradient at the position
        # that contained the maximum.
        for i in range(output_height):

            h_start = i * stride

            for j in range(output_width):

                w_start = j * stride

                positions = max_positions[
                    :,
                    :,
                    i,
                    j
                ]

                row = positions // pool
                col = positions % pool

                batch_indices = np.arange(batch_size)[:, None]
                channel_indices = np.arange(channels)[None, :]

                dx[
                    batch_indices,
                    channel_indices,
                    h_start + row,
                    w_start + col
                ] += incoming_grad[
                    :,
                    :,
                    i,
                    j
                ]

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
        scale = np.sqrt(
            2 / (in_channels * kernel_size * kernel_size)
        )

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

        batch_size, channels, height, width = x.shape

        kernel = self.kernel_size
        stride = self.stride

        output_height = (
            (height - kernel) // stride
        ) + 1

        output_width = (
            (width - kernel) // stride
        ) + 1

        # Extract all kernel-sized regions at once.
        windows = np.lib.stride_tricks.sliding_window_view(
            x,
            (kernel, kernel),
            axis=(2, 3)
        )

        # Apply the convolution stride.
        windows = windows[
            :,
            :,
            ::stride,
            ::stride,
            :,
            :
        ]

        # Save the windows for backpropagation.
        self.windows = windows

        # Multiply every window by every filter.
        output = np.einsum(
            "ncijkl,fckl->nfij",
            windows,
            self.weights
        )

        # Add the bias for each output channel.
        output += self.biases[
            None,
            :,
            None,
            None
        ]

        return output

    def backprop(self, incoming_grad):
        """
        Backpropagate through convolution.

        Returns:
            Gradient with respect to input.
        """

        x = self.input

        batch_size, channels, height, width = x.shape

        kernel = self.kernel_size
        stride = self.stride

        output_height = incoming_grad.shape[2]
        output_width = incoming_grad.shape[3]

        # Gradient with respect to the filters.
        self.d_weights = np.einsum(
            "nfij,ncijkl->fckl",
            incoming_grad,
            self.windows
        )

        # Gradient with respect to the biases.
        self.d_biases = np.sum(
            incoming_grad,
            axis=(0, 2, 3)
        )

        # Gradient with respect to the input.
        dx = np.zeros_like(x)

        for i in range(output_height):

            h_start = i * stride
            h_end = h_start + kernel

            for j in range(output_width):

                w_start = j * stride
                w_end = w_start + kernel

                grad = incoming_grad[
                    :,
                    :,
                    i,
                    j
                ]

                dx[
                    :,
                    :,
                    h_start:h_end,
                    w_start:w_end
                ] += np.einsum(
                    "nf,fckl->nckl",
                    grad,
                    self.weights
                )

        return dx