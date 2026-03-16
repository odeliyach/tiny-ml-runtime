"""
Training script for Iris classification model.

This module trains a simple feedforward neural network on the Iris dataset
using PyTorch. The trained model is exported for use with the C inference engine.

Architecture: 4 → 8 → 3 (input → hidden → output classes)
"""

import logging
import sys
from pathlib import Path
from typing import Tuple

import numpy as np
import torch
import torch.nn as nn
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger(__name__)


class IrisNet(nn.Module):
    """Neural network for Iris flower classification.

    Architecture:
        - Input layer: 4 features (sepal length/width, petal length/width)
        - Hidden layer: 8 neurons with ReLU activation
        - Output layer: 3 classes (Setosa, Versicolor, Virginica)
    """

    def __init__(self, input_size: int = 4, hidden_size: int = 8, num_classes: int = 3):
        """Initialize the network architecture.

        Args:
            input_size: Number of input features. Defaults to 4.
            hidden_size: Number of neurons in hidden layer. Defaults to 8.
            num_classes: Number of output classes. Defaults to 3.
        """
        super().__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(hidden_size, num_classes)

        logger.info(
            f"Initialized IrisNet: {input_size} → {hidden_size} → {num_classes}"
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass through the network.

        Args:
            x: Input tensor of shape (batch_size, input_size).

        Returns:
            Output logits of shape (batch_size, num_classes).
        """
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        return x


def load_and_preprocess_data(
    test_size: float = 0.2,
    random_state: int = 42
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, StandardScaler]:
    """Load Iris dataset and split into train/test sets.

    Args:
        test_size: Fraction of data to use for testing. Defaults to 0.2.
        random_state: Random seed for reproducibility. Defaults to 42.

    Returns:
        Tuple of (X_train, X_test, y_train, y_test, scaler).

    Raises:
        RuntimeError: If dataset loading fails.
    """
    try:
        logger.info("Loading Iris dataset...")
        iris = load_iris()
        X = iris.data.astype(np.float32)
        y = iris.target

        logger.info(f"Dataset shape: {X.shape}, Classes: {len(np.unique(y))}")

        # Normalize features using StandardScaler
        logger.info("Applying StandardScaler normalization...")
        scaler = StandardScaler()
        X = scaler.fit_transform(X)

        # Train/test split
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, random_state=random_state
        )

        logger.info(
            f"Split: train={len(X_train)} samples, test={len(X_test)} samples"
        )

        # Convert to PyTorch tensors
        X_train = torch.tensor(X_train)
        X_test = torch.tensor(X_test)
        y_train = torch.tensor(y_train, dtype=torch.long)
        y_test = torch.tensor(y_test, dtype=torch.long)

        return X_train, X_test, y_train, y_test, scaler

    except Exception as e:
        logger.error(f"Failed to load dataset: {e}")
        raise RuntimeError("Dataset loading failed") from e


def train_model(
    model: nn.Module,
    X_train: torch.Tensor,
    y_train: torch.Tensor,
    epochs: int = 500,
    learning_rate: float = 0.01,
    log_interval: int = 100
) -> nn.Module:
    """Train the neural network model.

    Args:
        model: Neural network model to train.
        X_train: Training input features.
        y_train: Training labels.
        epochs: Number of training epochs. Defaults to 500.
        learning_rate: Learning rate for Adam optimizer. Defaults to 0.01.
        log_interval: Log progress every N epochs. Defaults to 100.

    Returns:
        Trained model.
    """
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    loss_fn = nn.CrossEntropyLoss()

    logger.info(f"Starting training for {epochs} epochs...")

    for epoch in range(epochs):
        model.train()

        # Forward pass
        outputs = model(X_train)
        loss = loss_fn(outputs, y_train)

        # Backward pass
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # Log progress
        if (epoch + 1) % log_interval == 0:
            logger.info(f"Epoch {epoch + 1}/{epochs}, Loss: {loss.item():.4f}")

    logger.info("Training completed successfully")
    return model


def evaluate_model(
    model: nn.Module,
    X_test: torch.Tensor,
    y_test: torch.Tensor
) -> float:
    """Evaluate model accuracy on test set.

    Args:
        model: Trained model to evaluate.
        X_test: Test input features.
        y_test: Test labels.

    Returns:
        Test accuracy as a float between 0 and 1.
    """
    model.eval()
    with torch.no_grad():
        outputs = model(X_test)
        predicted = outputs.argmax(dim=1)
        accuracy = (predicted == y_test).float().mean().item()

    logger.info(f"Test Accuracy: {accuracy * 100:.2f}%")
    return accuracy


def save_weights(
    model: nn.Module,
    scaler: StandardScaler,
    output_path: str = "weights.npy"
) -> None:
    """Save trained model weights and scaler parameters to file.

    Args:
        model: Trained model.
        scaler: Fitted StandardScaler.
        output_path: Path to save weights file. Defaults to "weights.npy".

    Raises:
        IOError: If weight saving fails.
    """
    try:
        weights = {
            'w1': model.fc1.weight.detach().numpy(),
            'b1': model.fc1.bias.detach().numpy(),
            'w2': model.fc2.weight.detach().numpy(),
            'b2': model.fc2.bias.detach().numpy(),
            'scaler_mean': scaler.mean_.astype(np.float32),
            'scaler_std': scaler.scale_.astype(np.float32)
        }

        output_file = Path(output_path)
        np.save(output_file, weights)

        logger.info(f"Weights saved successfully to: {output_file.absolute()}")

    except Exception as e:
        logger.error(f"Failed to save weights: {e}")
        raise IOError("Weight saving failed") from e


def main() -> None:
    """Main training pipeline execution."""
    try:
        # Load and preprocess data
        X_train, X_test, y_train, y_test, scaler = load_and_preprocess_data()

        # Initialize model
        model = IrisNet(input_size=4, hidden_size=8, num_classes=3)

        # Train model
        model = train_model(model, X_train, y_train, epochs=500, learning_rate=0.01)

        # Evaluate model
        accuracy = evaluate_model(model, X_test, y_test)

        # Save weights
        save_weights(model, scaler, output_path="weights.npy")

        logger.info("Training pipeline completed successfully")
        logger.info(f"Final test accuracy: {accuracy * 100:.2f}%")

    except Exception as e:
        logger.error(f"Training pipeline failed: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
