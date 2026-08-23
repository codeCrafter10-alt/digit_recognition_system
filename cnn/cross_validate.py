"""
Cross-validation script for hyperparameter tuning.
Trains the network with different hyperparameter combinations
Finds the best performing configuration using early stopping.
"""

import os
import numpy as np

from train import load_data, one_hot_encode, shuffle_data
from model import create_model
from nn.loss import CrossEntropy


HYPERPARAMS = [
    {"learning_rate": 0.1,   "batch_size": 32},
    {"learning_rate": 0.1,   "batch_size": 64},
    {"learning_rate": 0.05,  "batch_size": 32},
    {"learning_rate": 0.05,  "batch_size": 64},
    {"learning_rate": 0.01,  "batch_size": 32},
    {"learning_rate": 0.01,  "batch_size": 64},
    {"learning_rate": 0.01,  "batch_size": 128},
    {"learning_rate": 0.005, "batch_size": 64},
    {"learning_rate": 0.005, "batch_size": 128},
    {"learning_rate": 0.001, "batch_size": 64},
]

EPOCHS = 10
PATIENCE = 3
VALIDATION_RATIO = 0.10



def evaluate(model, x, y, batch_size):
    correct = 0
    total = len(x)

    for start in range(0, total, batch_size):

        end = min(
            start + batch_size,
            total
        )

        x_batch = x[start:end]
        y_batch = y[start:end]

        predictions = model.forward(x_batch)

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

    return correct / total


def validation_loss(model, loss_function, x, y, batch_size):

    total_loss = 0.0
    total = len(x)

    for start in range(0, total, batch_size):

        end = min(
            start + batch_size,
            total
        )

        x_batch = x[start:end]
        y_batch = y[start:end]

        predictions = model.forward(x_batch)

        loss = loss_function.forward(
            predictions,
            y_batch
        )

        total_loss += loss * len(x_batch)

    return total_loss / total


def train_model(
    x_train,
    y_train,
    x_val,
    y_val,
    learning_rate,
    batch_size
):

    model = create_model()
    loss_function = CrossEntropy()

    best_val_loss = float("inf")
    best_val_accuracy = 0.0

    epochs_without_improvement = 0

    num_samples = len(x_train)

    for epoch in range(EPOCHS):

        x_train, y_train = shuffle_data(
            x_train,
            y_train
        )

        total_loss = 0.0
        correct = 0

        for start in range(0, num_samples, batch_size):

            end = min(
                start + batch_size,
                num_samples
            )

            x_batch = x_train[start:end]
            y_batch = y_train[start:end]

            predictions = model.forward(x_batch)

            loss = loss_function.forward(
                predictions,
                y_batch
            )

            total_loss += loss * len(x_batch)

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

            gradient = loss_function.backward(
                predictions,
                y_batch
            )

            model.backward(gradient)

            model.update(learning_rate)

        train_loss = total_loss / num_samples

        val_loss = validation_loss(
            model,
            loss_function,
            x_val,
            y_val,
            batch_size
        )

        val_accuracy = evaluate(
            model,
            x_val,
            y_val,
            batch_size
        )

        print(
            f"Epoch {epoch + 1}/{EPOCHS} "
            f"- Train Loss: {train_loss:.4f} "
            f"- Val Loss: {val_loss:.4f} "
            f"- Val Acc: {val_accuracy:.4f}"
        )

        if val_loss < best_val_loss:

            best_val_loss = val_loss
            best_val_accuracy = val_accuracy
            epochs_without_improvement = 0

        else:

            epochs_without_improvement += 1

            if epochs_without_improvement >= PATIENCE:

                print("Early stopping.")

                break

    return model, best_val_accuracy


def cross_validate(x_train, y_train):

    x_train, y_train = shuffle_data(
        x_train,
        y_train
    )

    split = int(
        len(x_train) * (1 - VALIDATION_RATIO)
    )

    x_tr = x_train[:split]
    y_tr = y_train[:split]

    x_val = x_train[split:]
    y_val = y_train[split:]

    best_combo = None
    best_accuracy = 0.0

    for i, combo in enumerate(HYPERPARAMS):

        learning_rate = combo["learning_rate"]
        batch_size = combo["batch_size"]

        print()
        print(
            f"Trial {i + 1}/{len(HYPERPARAMS)}"
        )
        print(
            f"Learning rate: {learning_rate}"
        )
        print(
            f"Batch size: {batch_size}"
        )

        model, accuracy = train_model(
            x_tr,
            y_tr,
            x_val,
            y_val,
            learning_rate,
            batch_size
        )

        print(
            f"Validation accuracy: "
            f"{accuracy * 100:.2f}%"
        )

        if accuracy > best_accuracy:

            best_accuracy = accuracy
            best_combo = combo
            best_model = model

            print("NEW BEST MODEL")


    print()
    print("BEST HYPERPARAMETERS:")

    print(
        f"Learning rate: "
        f"{best_combo['learning_rate']}"
    )

    print(
        f"Batch size: "
        f"{best_combo['batch_size']}"
    )

    print(
        f"Validation accuracy: "
        f"{best_accuracy * 100:.2f}%"
    )

    return best_combo, best_accuracy


def main():

    print("Loading MNIST...")

    x_train, y_train, x_test, y_test = load_data()

    y_train = one_hot_encode(y_train)
    y_test = one_hot_encode(y_test)

    best_combo, best_accuracy = cross_validate(
        x_train,
        y_train
    )

    print()
    print("Best configuration:")
    print(best_combo)

    print(
        f"Validation accuracy: "
        f"{best_accuracy * 100:.2f}%"
    )

    #   BEST HYPERPARAMETERS:
    #   Learning rate: 0.05
    #   Batch size: 32
    #   Validation accuracy: 98.98%


if __name__ == "__main__":
    main()