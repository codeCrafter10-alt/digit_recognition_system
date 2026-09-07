"""
Data augmentation function for improving model
Applies random transformation to training immages
Simulates messy real-world handwriteen digit inputs.
"""

import numpy as np
from scipy.ndimage import rotate, shift, gaussian_filter


def shift_image(image):
    """
    Randomly shift an image in x and y direction

    Parameters:
    image: numpy array with shape (784,)
        Flattened input image with pixel values between 0 and 1
    Returns:
    numpy array with shape (784,)
        Shifted image
    """
    img = image.reshape(28,28)
    dx, dy = np.random.randint(-3,3), np.random.randint(-3,3)
    img = shift(img, [dy,dx])
    return img.reshape(784)

def add_noise(image):
    """
    Add random Gaussian noise to an image

    Parameters:
    image: numpy array with shape (784,)
        Flattened input image with pixel values between 0 and 1
    Returns:
    numpy array with shape (784,)
        Noisy image
    """
    noise = np.random.normal(0,0.1, image.shape)
    return np.clip(image+noise, 0, 1)

def blur_image(image):
    """
    Apply Gaussian blur to an image

    Parameters:
    image: numpy array with shape (784,)
        Flattened input image with pixel values between 0 and 1
    Returns:
    numpy array with shape (784,)
        Blurred image
    """
    img = image.reshape(28, 28)
    img = gaussian_filter(img, sigma=0.8)
    return img.reshape(784)

def rotate_image(image):
    """
    Randomly rotate an image between -15 and 15 degrees

    Parameters:
    image: numpy array with shape (784,)
        Flattened input image with pixel values between 0 and 1
    Returns:
    numpy array with shape (784,)
        Rotated image
    """
    img = image.reshape(28, 28)
    angle = np.random.uniform(-15,15)
    img = rotate(img, angle, reshape=False)
    return img.reshape(784)

def augment(image):
    """
    Randomly apply a combination of transformations to an image

    Parameters:
    image: numpy array with shape (784,)
        Flattened input image with pixel values between 0 and 1
    Returns:
    numpy array with shape (784,)
        Augmented image with random transformations applied
    """
    image = image.reshape(784)

    if np.random.random() < 0.5: # 50% chance to apply transformation
        image = shift_image(image)
    if np.random.random() < 0.5: # 50% chance to apply transformation
        image = add_noise(image)
    if np.random.random() < 0.5: # 50% chance to apply transformation
        image = blur_image(image)
    if np.random.random() < 0.5: # 50% chance to apply transformation
        image = rotate_image(image)
    return image.reshape(1, 28, 28)