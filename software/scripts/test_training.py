#!/usr/bin/env python3
"""Test script to verify training loop works with synthetic data.

This script tests:
1. Loss functions
2. Metrics computation
3. Trainer with synthetic batches
"""

import sys
from pathlib import Path

import torch
from torch.utils.data import DataLoader, TensorDataset

# Add software directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from loaf.model import MPNN, VisionTransformer
from loaf.training import (
    Trainer,
    TrainingConfig,
    Metrics,
    MSELoss,
    WindSpeedLoss,
    CombinedLoss,
    wind_speed_error,
)


def test_wind_speed_error():
    """Test wind speed error calculation."""
    print("=" * 60)
    print("Testing wind_speed_error")
    print("=" * 60)

    # Perfect prediction (epsilon is added for numerical stability, so not exactly 0)
    output = torch.tensor([[1.0, 2.0], [3.0, 4.0]])
    target = torch.tensor([[1.0, 2.0], [3.0, 4.0]])
    error = wind_speed_error(output, target)
    # Error should be very small (just epsilon)
    assert error.max() < 1e-3
    print("  Perfect prediction: PASSED")

    # Known error - should be approximately 1.0 (sqrt(1^2 + 0^2 + eps))
    output = torch.tensor([[1.0, 0.0]])
    target = torch.tensor([[0.0, 0.0]])
    error = wind_speed_error(output, target)
    assert torch.abs(error - 1.0).max() < 1e-3
    print("  Unit error: PASSED")

    print("\nwind_speed_error: PASSED")


def test_loss_functions():
    """Test loss function implementations."""
    print("\n" + "=" * 60)
    print("Testing Loss Functions")
    print("=" * 60)

    batch_size = 4
    n_stations = 10
    n_vars = 2

    output = torch.randn(batch_size, n_stations, n_vars)
    target = torch.randn(batch_size, n_stations, n_vars)

    # MSE Loss
    mse_loss = MSELoss()
    loss = mse_loss(output, target)
    assert loss.dim() == 0  # Scalar
    assert loss.item() >= 0
    print(f"  MSELoss: {loss.item():.4f} (shape: {loss.shape})")

    # Wind Speed Loss
    ws_loss = WindSpeedLoss()
    loss = ws_loss(output, target)
    assert loss.dim() == 0
    assert loss.item() >= 0
    print(f"  WindSpeedLoss: {loss.item():.4f}")

    # Combined Loss
    combined_loss = CombinedLoss(["u", "v"])
    loss = combined_loss(output, target)
    assert loss.dim() == 0
    assert loss.item() >= 0
    print(f"  CombinedLoss: {loss.item():.4f}")

    # Test with mask
    mask = torch.rand(batch_size, n_stations) > 0.3
    loss = mse_loss(output, target, mask)
    print(f"  MSELoss with mask: {loss.item():.4f}")

    print("\nLoss functions: PASSED")


def test_metrics():
    """Test metrics computation."""
    print("\n" + "=" * 60)
    print("Testing Metrics")
    print("=" * 60)

    metrics = Metrics(["u", "v"])

    # Generate some batches
    for _ in range(5):
        output = torch.randn(4, 10, 2)
        target = torch.randn(4, 10, 2)
        metrics.update(output, target)

    results = metrics.compute()

    print("  Computed metrics:")
    for name, value in results.items():
        print(f"    {name}: {value:.4f}")

    assert "mse_u" in results
    assert "mse_v" in results
    assert "rmse_u" in results
    assert "rmse_v" in results
    assert "mae_u" in results
    assert "mae_v" in results
    assert "wind_speed_error" in results

    # Reset should clear all
    metrics.reset()
    results = metrics.compute()
    assert len([v for v in results.values() if v > 0]) == 0

    print("\nMetrics: PASSED")


def create_synthetic_dataloader(n_samples, n_stations, n_hours, n_vars, batch_size):
    """Create a dataloader with synthetic weather data."""
    # Create synthetic tensors matching expected dataset format
    # For simplicity, we create a TensorDataset that yields dicts

    class SyntheticDataset(torch.utils.data.Dataset):
        def __init__(self, n_samples, n_stations, n_hours, n_vars):
            self.n_samples = n_samples
            self.n_stations = n_stations
            self.n_hours = n_hours
            self.n_vars = n_vars

            # Pre-generate some data
            self.station_lon = torch.rand(n_stations) * 3 - 124  # -124 to -121
            self.station_lat = torch.rand(n_stations) * 2.5 + 46.5  # 46.5 to 49

            # Create edge index (simple ring graph)
            edges = []
            for i in range(n_stations):
                edges.append([i, (i + 1) % n_stations])
                edges.append([(i + 1) % n_stations, i])
            self.edge_index = torch.tensor(edges, dtype=torch.long).t()

        def __len__(self):
            return self.n_samples

        def __getitem__(self, idx):
            return {
                "u": torch.randn(self.n_hours, self.n_stations),
                "v": torch.randn(self.n_hours, self.n_stations),
                "temp": torch.randn(self.n_hours, self.n_stations),
                "dewpoint": torch.randn(self.n_hours, self.n_stations),
                "station_lon": self.station_lon,
                "station_lat": self.station_lat,
                "k_edge_index": self.edge_index,
            }

    dataset = SyntheticDataset(n_samples, n_stations, n_hours, n_vars)
    return DataLoader(dataset, batch_size=batch_size, shuffle=True)


def test_trainer_gnn():
    """Test trainer with GNN model."""
    print("\n" + "=" * 60)
    print("Testing Trainer with MPNN")
    print("=" * 60)

    # Model parameters
    n_stations = 10
    n_hours = 24
    n_vars = 4
    hidden_dim = 32
    n_out = 2

    # Create model
    model = MPNN(
        n_passing=2,
        lead_hrs=6,
        n_node_features_m=n_hours * n_vars,
        n_node_features_e=n_hours * n_vars,
        n_out_features=n_out,
        hidden_dim=hidden_dim,
    )
    print(f"  Model parameters: {sum(p.numel() for p in model.parameters()):,}")

    # Create synthetic dataloaders
    train_loader = create_synthetic_dataloader(
        n_samples=32, n_stations=n_stations, n_hours=n_hours, n_vars=n_vars, batch_size=8
    )
    val_loader = create_synthetic_dataloader(
        n_samples=16, n_stations=n_stations, n_hours=n_hours, n_vars=n_vars, batch_size=8
    )

    # Create trainer
    config = TrainingConfig(
        epochs=3,
        learning_rate=1e-3,
        batch_size=8,
        patience=5,
        device="cpu",
        output_vars=["u", "v"],
    )

    trainer = Trainer(model, config, model_type="gnn")

    # Train for a few epochs
    print("  Training for 3 epochs...")
    history = trainer.train(train_loader, val_loader)

    assert len(history["train_loss"]) == 3
    assert len(history["val_loss"]) == 3
    print(f"  Final train loss: {history['train_loss'][-1]:.4f}")
    print(f"  Final val loss: {history['val_loss'][-1]:.4f}")

    print("\nTrainer with MPNN: PASSED")


def test_trainer_vit():
    """Test trainer with ViT model."""
    print("\n" + "=" * 60)
    print("Testing Trainer with VisionTransformer")
    print("=" * 60)

    # Model parameters
    n_stations = 10
    n_hours = 24
    n_vars = 4
    n_out = 2

    # Create model
    model = VisionTransformer(
        n_stations=n_stations,
        madis_len=n_hours,
        madis_n_vars_i=n_vars,
        madis_n_vars_o=n_out,
        dim=32,
        attn_dim=16,
        mlp_dim=64,
        num_heads=2,
        num_layers=2,
    )
    print(f"  Model parameters: {sum(p.numel() for p in model.parameters()):,}")

    # Create synthetic dataloaders
    train_loader = create_synthetic_dataloader(
        n_samples=32, n_stations=n_stations, n_hours=n_hours, n_vars=n_vars, batch_size=8
    )
    val_loader = create_synthetic_dataloader(
        n_samples=16, n_stations=n_stations, n_hours=n_hours, n_vars=n_vars, batch_size=8
    )

    # Create trainer
    config = TrainingConfig(
        epochs=3,
        learning_rate=1e-3,
        batch_size=8,
        patience=5,
        device="cpu",
        output_vars=["u", "v"],
    )

    trainer = Trainer(model, config, model_type="vit")

    # Train for a few epochs
    print("  Training for 3 epochs...")
    history = trainer.train(train_loader, val_loader)

    assert len(history["train_loss"]) == 3
    assert len(history["val_loss"]) == 3
    print(f"  Final train loss: {history['train_loss'][-1]:.4f}")
    print(f"  Final val loss: {history['val_loss'][-1]:.4f}")

    print("\nTrainer with VisionTransformer: PASSED")


def main():
    """Run all tests."""
    print("\n" + "#" * 60)
    print("# LOAF Training Pipeline Tests")
    print("#" * 60)

    test_wind_speed_error()
    test_loss_functions()
    test_metrics()
    test_trainer_gnn()
    test_trainer_vit()

    print("\n" + "=" * 60)
    print("ALL TRAINING TESTS PASSED!")
    print("=" * 60 + "\n")


if __name__ == "__main__":
    main()
