"""Unit tests for training pipeline."""

import numpy as np
import pytest
import torch

from src.python.train import IrisNet, load_and_preprocess_data, train_model, evaluate_model


class TestIrisNet:
    """Test cases for IrisNet neural network."""

    def test_network_initialization(self):
        """Test that network initializes with correct architecture."""
        model = IrisNet(input_size=4, hidden_size=8, num_classes=3)

        assert model.fc1.in_features == 4
        assert model.fc1.out_features == 8
        assert model.fc2.in_features == 8
        assert model.fc2.out_features == 3

    def test_forward_pass_shape(self):
        """Test that forward pass produces correct output shape."""
        model = IrisNet()
        batch_size = 10
        input_tensor = torch.randn(batch_size, 4)

        output = model(input_tensor)

        assert output.shape == (batch_size, 3)

    def test_forward_pass_single_sample(self):
        """Test forward pass with single sample."""
        model = IrisNet()
        input_tensor = torch.randn(1, 4)

        output = model(input_tensor)

        assert output.shape == (1, 3)
        assert not torch.isnan(output).any()
        assert not torch.isinf(output).any()


class TestDataLoading:
    """Test cases for data loading and preprocessing."""

    def test_load_and_preprocess_data(self):
        """Test that data loading returns correct shapes."""
        X_train, X_test, y_train, y_test, scaler = load_and_preprocess_data(
            test_size=0.2, random_state=42
        )

        # Check data split (80/20)
        total_samples = len(X_train) + len(X_test)
        assert total_samples == 150  # Iris dataset size

        # Check shapes
        assert X_train.shape[1] == 4  # 4 features
        assert X_test.shape[1] == 4

        # Check normalization (mean ≈ 0, std ≈ 1)
        train_mean = X_train.mean().item()
        train_std = X_train.std().item()
        assert abs(train_mean) < 0.5  # Close to 0
        assert abs(train_std - 1.0) < 0.5  # Close to 1

        # Check labels are valid
        assert y_train.min() >= 0
        assert y_train.max() <= 2
        assert y_test.min() >= 0
        assert y_test.max() <= 2

    def test_data_reproducibility(self):
        """Test that same random_state produces same split."""
        X_train1, X_test1, _, _, _ = load_and_preprocess_data(random_state=42)
        X_train2, X_test2, _, _, _ = load_and_preprocess_data(random_state=42)

        assert torch.allclose(X_train1, X_train2)
        assert torch.allclose(X_test1, X_test2)


class TestTraining:
    """Test cases for model training."""

    def test_train_model_reduces_loss(self):
        """Test that training reduces loss over epochs."""
        X_train, _, y_train, _, _ = load_and_preprocess_data()
        model = IrisNet()

        # Get initial loss
        initial_loss = torch.nn.functional.cross_entropy(
            model(X_train), y_train
        ).item()

        # Train model
        trained_model = train_model(
            model, X_train, y_train, epochs=50, learning_rate=0.01, log_interval=50
        )

        # Get final loss
        final_loss = torch.nn.functional.cross_entropy(
            trained_model(X_train), y_train
        ).item()

        # Loss should decrease
        assert final_loss < initial_loss

    def test_evaluate_model_returns_valid_accuracy(self):
        """Test that evaluation returns valid accuracy score."""
        X_train, X_test, y_train, y_test, _ = load_and_preprocess_data()
        model = IrisNet()

        # Train briefly
        trained_model = train_model(
            model, X_train, y_train, epochs=100, learning_rate=0.01, log_interval=100
        )

        # Evaluate
        accuracy = evaluate_model(trained_model, X_test, y_test)

        # Accuracy should be between 0 and 1
        assert 0.0 <= accuracy <= 1.0

        # After 100 epochs, should achieve reasonable accuracy
        assert accuracy > 0.5  # Better than random (33%)


class TestNumericalStability:
    """Test cases for numerical stability."""

    def test_no_nans_in_training(self):
        """Test that training doesn't produce NaN values."""
        X_train, _, y_train, _, _ = load_and_preprocess_data()
        model = IrisNet()

        trained_model = train_model(
            model, X_train, y_train, epochs=10, learning_rate=0.01, log_interval=10
        )

        # Check all parameters are finite
        for param in trained_model.parameters():
            assert not torch.isnan(param).any()
            assert not torch.isinf(param).any()

    def test_extreme_inputs_dont_crash(self):
        """Test that model handles extreme input values gracefully."""
        model = IrisNet()

        # Test with very large values
        large_input = torch.ones(1, 4) * 1000
        output = model(large_input)
        assert not torch.isnan(output).any()

        # Test with very small values
        small_input = torch.ones(1, 4) * 0.001
        output = model(small_input)
        assert not torch.isnan(output).any()

        # Test with zeros
        zero_input = torch.zeros(1, 4)
        output = model(zero_input)
        assert not torch.isnan(output).any()


@pytest.mark.parametrize("hidden_size", [4, 8, 16, 32])
def test_different_hidden_sizes(hidden_size):
    """Test that model works with various hidden layer sizes."""
    model = IrisNet(input_size=4, hidden_size=hidden_size, num_classes=3)
    input_tensor = torch.randn(5, 4)

    output = model(input_tensor)

    assert output.shape == (5, 3)
    assert not torch.isnan(output).any()
