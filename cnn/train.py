import numpy as np
import os
import urllib.request
import pickle

from cnn.model import create_model
from cnn.nn.loss import CrossEntropy
from cnn.augment import augment

# Settings
BATCH_SIZE = 32
EPOCHS = 20
LEARNING_RATE = 0.05

def download_mnist():
    """
    Download MNIST dataset if not already present

    Returns:
    string
        Filepath to the downloaded .npz dataset file
    """
    url = "https://storage.googleapis.com/tensorflow/tf-keras-datasets/mnist.npz"
    filepath = "data/mnist.npz"

    if not os.path.exists("data"):
        os.makedirs("data")
    
    if not os.path.exists(filepath):
        print("Downloading MNIST Dataset...")
        urllib.request.urlretrieve(url, filepath)
        print("Finished downloading MNIST Dataset.")
    
    return filepath

def load_data():
    """
    Download and load the MNIST dataset.

    Returns:
        x_train: training images
        y_train: training labels
        x_test: test images
        y_test: test labels
    """

    filepath = download_mnist()

    data = np.load(filepath)

    x_train = data["x_train"]
    y_train = data["y_train"]

    x_test = data["x_test"]
    y_test = data["y_test"]

    # Convert pixel values from 0-255 to 0-1
    x_train = x_train.astype(np.float32) / 255.0
    x_test = x_test.astype(np.float32) / 255.0

    # Add the channel dimension
    x_train = x_train[:, np.newaxis, :, :]
    x_test = x_test[:, np.newaxis, :, :]

    return x_train, y_train, x_test, y_test


def one_hot_encode(labels):
    """
    Convert integer labels into one-hot encoded vectors.

    Example:
        3 -> [0, 0, 0, 1, 0, 0, 0, 0, 0, 0]
    """

    encoded = np.zeros(
        (len(labels), 10),
        dtype=np.float32
    )

    encoded[np.arange(len(labels)), labels] = 1.0

    return encoded


def shuffle_data(x, y):
    """
    Shuffle images and labels using the same random order.
    """

    indices = np.random.permutation(len(x))

    return x[indices], y[indices]


def evaluate(model, x, y):
    """
    Evaluate the model on a dataset.

    Returns:
        average_loss
        accuracy
    """

    correct = 0
    total = len(x)

    for start in range(0, total, BATCH_SIZE):

        end = min(
            start + BATCH_SIZE,
            total
        )

        x_batch = x[start:end]
        x_batch = np.array([augment(img) for img in x_batch])

        y_batch = y[start:end]

        # Forward pass
        predictions = model.forward(x_batch)

        # Calculate accuracy
        predicted_labels = np.argmax(
            predictions,
            axis=1
        )

        actual_labels = np.argmax(
            y_batch,
            axis=1
        )

        correct += np.sum(
            predicted_labels == actual_labels
        )

    accuracy = correct / total

    return accuracy


def train(model, loss_function, x_train, y_train, x_test, y_test):
    """
    Train the CNN using batch gradient descent.
    """

    num_samples = len(x_train)

    
    history = []


    for epoch in range(EPOCHS):

        # Shuffle training data at the beginning
        # of every epoch
        x_train, y_train = shuffle_data(
            x_train,
            y_train
        )

        total_loss = 0.0
        correct = 0

        for start in range(0, num_samples, BATCH_SIZE):
            end = min(
                start + BATCH_SIZE,
                num_samples
            )

            x_batch = x_train[start:end]
            x_batch = np.array([augment(img) for img in x_batch])

            y_batch = y_train[start:end]

            # Forward pass
            predictions = model.forward(x_batch)

            # Calculate loss
            loss = loss_function.forward(
                predictions,
                y_batch
            )

            total_loss += loss * len(x_batch)

            # Calculate training accuracy
            predicted_labels = np.argmax(
                predictions,
                axis=1
            )

            actual_labels = np.argmax(
                y_batch,
                axis=1
            )

            correct += np.sum(
                predicted_labels == actual_labels
            )

            # Backward pass
            gradient = loss_function.backward(
                predictions,
                y_batch
            )

            model.backward(gradient)

            # Update weights
            model.update(LEARNING_RATE)


        history.append(loss)

        # Average training loss
        average_loss = total_loss / num_samples

        # Training accuracy
        train_accuracy = correct / num_samples

        # Evaluate on test set
        test_accuracy = evaluate(
            model,
            x_test,
            y_test
        )

        print(
            f"Epoch {epoch + 1}/{EPOCHS} "
            f"- Loss: {average_loss:.4f} "
            f"- Accuracy: {train_accuracy:.4f} "
            f"- Test Accuracy: {test_accuracy:.4f}"
        )

    return history



def main():

    print("Loading MNIST data...")

    x_train, y_train, x_test, y_test = load_data()

    print("Training data:", x_train.shape)
    print("Test data:", x_test.shape)

    # Convert labels to one-hot vectors
    y_train = one_hot_encode(y_train)
    y_test = one_hot_encode(y_test)

    print("Creating model...")

    model = create_model()
    loss_function = CrossEntropy()

    print("Starting training...")
    print()

    history = train(
        model,
        loss_function,
        x_train,
        y_train,
        x_test,
        y_test
    )

    print()
    print("Training finished.")
    print("Saving model...")

    # Save the training history
    os.makedirs("cnn/model", exist_ok=True)
    with open("cnn/model/history.pkl", "wb") as f:
        pickle.dump(history, f)

    os.makedirs("cnn/model", exist_ok=True)
    model.save_weights("cnn/model/weights.pkl")

    print("Model saved to:", "cnn/model/weights.pkl")


if __name__ == "__main__":
    main()