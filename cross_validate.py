"""
Cross-validation script for hyperparameter tuning.
Trains the network with different hyperparameter combinations
Finds the best performing configuration using early stopping.
"""

from train import load_mnist, download_mnist
import numpy as np
from nn.network import Network
from nn.activations import ReLU, Softmax
from nn.loss import CrossEntropy
from nn.layers import Dense

hyperparams = [
    {'learning_rate': 0.1,   'batch_size': 32},
    {'learning_rate': 0.1,   'batch_size': 64},
    {'learning_rate': 0.01,  'batch_size': 32},
    {'learning_rate': 0.01,  'batch_size': 64},
    {'learning_rate': 0.01,  'batch_size': 128},
    {'learning_rate': 0.001, 'batch_size': 32},
    {'learning_rate': 0.001, 'batch_size': 64},
    {'learning_rate': 0.001, 'batch_size': 128},
    {'learning_rate': 0.005, 'batch_size': 32},
    {'learning_rate': 0.005, 'batch_size': 64},
]
def cross_validate(x_train, y_train, hyperparams, epochs=50, patience=5):
    """
    Train the network with different hyperparameter combinations
    and return the best performing configuration.

    Parameters:
    x_train: numpy array with shape (60000, 784)
        Training images.
    y_train: numpy array with shape (60000, 10)
        One-hot encoded training labels.
    hyperparams: list of dicts
        Each dict contains learning_rate and batch_size.
    epochs: int
        Maximum number of epochs per combo.
    patience: int
        Number of epochs with no improvement before early stopping.

    Returns:
    dict
        Best hyperparameter combination and its validation accuracy.
    """
    best_combo = None
    best_accuracy = 0

    # Split into train/val (10% validation)
    split = int(len(x_train) * 0.9)
    x_val, y_val = x_train[split:], y_train[split:]
    x_tr, y_tr = x_train[:split], y_train[:split]

    for combo in hyperparams:
        learning_rate = combo['learning_rate']
        batch_size = combo['batch_size']
        print(f"\nTrying learning_rate={learning_rate}, batch_size={batch_size}")

        # Build fresh network
        network = Network([
            Dense(784, 128), ReLU(),
            Dense(128, 64), ReLU(),
            Dense(64, 10), Softmax()
        ])

        loss_fn = CrossEntropy()
        prev_val_loss = float('inf')
        epochs_no_improve = 0

        for epoch in range(epochs):
            # Training loop
            for i in range(0, len(x_tr), batch_size):
                x_batch = x_tr[i:i+batch_size]
                y_batch = y_tr[i:i+batch_size]
                predictions = network.forward(x_batch)
                loss = loss_fn.forward(predictions, y_batch)
                gradient = loss_fn.backward(predictions, y_batch)
                network.backward(gradient)
                network.update(learning_rate)

            # Validation loss
            val_predictions = network.forward(x_val)
            val_loss = loss_fn.forward(val_predictions, y_val)
            print(f"Epoch {epoch+1} - Val Loss: {val_loss:.4f}")

            # Early stopping
            if val_loss < prev_val_loss:
                epochs_no_improve = 0
            else:
                epochs_no_improve += 1
                if epochs_no_improve >= patience:
                    print(f"Early stopping at epoch {epoch+1}")
                    break
            
            prev_val_loss = val_loss

        # Record accuracy
        preds = network.predict(x_val)
        true = np.argmax(y_val, axis=1)
        accuracy = np.mean(preds == true)
        print(f"Accuracy: {accuracy * 100:.2f}%")

        if accuracy > best_accuracy:
            best_accuracy = accuracy
            best_combo = combo

    print(f"\nBest combo: {best_combo} with accuracy {best_accuracy * 100:.2f}%")
    return best_combo

def main():
    filepath = download_mnist()
    x_train, y_train, x_test, y_test = load_mnist(filepath)
    best = cross_validate(x_train, y_train, hyperparams)
    print(f"\nUse these hyperparameters: {best}")

if __name__ == "__main__":
    main()