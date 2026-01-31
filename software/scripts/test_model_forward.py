#!/usr/bin/env python3
"""Test script to verify model forward pass works with sample data.

This script creates synthetic data and tests:
1. MPNN forward pass
2. VisionTransformer forward pass
3. Network construction
"""

import numpy as np
import torch

from loaf.model import (
    MPNN,
    VisionTransformer,
    StationNetwork,
    GridNetwork,
    NetworkConstructionMethod,
    build_networks,
)


def test_network_construction():
    """Test graph network construction."""
    print("=" * 60)
    print("Testing Network Construction")
    print("=" * 60)

    # Create synthetic station locations (Seattle area)
    n_stations = 10
    station_lons = np.random.uniform(-122.5, -121.5, n_stations).astype(np.float32)
    station_lats = np.random.uniform(47.0, 48.0, n_stations).astype(np.float32)

    # Test methods that don't require torch-cluster
    methods_to_test = [
        NetworkConstructionMethod.DELAUNAY,
        NetworkConstructionMethod.FULLY_CONNECTED,
    ]

    # Try KNN if torch-cluster is available
    knn_available = False
    try:
        import torch_cluster
        methods_to_test.insert(0, NetworkConstructionMethod.KNN)
        knn_available = True
    except ImportError:
        print("\nNote: torch-cluster not available, skipping KNN tests")

    for method in methods_to_test:
        print(f"\nMethod: {method.name}")
        network = StationNetwork(
            station_lons=station_lons,
            station_lats=station_lats,
            n_neighbors=3,
            method=method,
        )
        print(f"  Stations: {network.n_stations}")
        print(f"  Edge index shape: {network.edge_index.shape}")
        print(f"  Number of edges: {network.edge_index.shape[1]}")

    # Test grid network only if KNN is available
    if knn_available:
        print("\nGrid Network (ERA5-like):")
        grid_lons = np.linspace(-124.0, -121.0, 10).astype(np.float32)
        grid_lats = np.linspace(46.5, 49.0, 8).astype(np.float32)
        grid_lons, grid_lats = np.meshgrid(grid_lons, grid_lats)

        station_network = StationNetwork(
            station_lons=station_lons,
            station_lats=station_lats,
            n_neighbors=3,
            method=NetworkConstructionMethod.KNN,
        )

        grid_network = GridNetwork(
            grid_lons=grid_lons,
            grid_lats=grid_lats,
            station_network=station_network,
            n_neighbors=4,
        )
        print(f"  Grid points: {grid_network.n_grid}")
        print(f"  Edge index shape: {grid_network.edge_index.shape}")
    else:
        print("\nGrid Network: Skipped (requires torch-cluster)")

    print("\nNetwork construction: PASSED")


def test_mpnn_forward():
    """Test MPNN forward pass."""
    print("\n" + "=" * 60)
    print("Testing MPNN Forward Pass")
    print("=" * 60)

    # Model parameters
    n_batch = 2
    n_stations = 10
    n_hours = 24
    n_features = 4  # u, v, temp, dewpoint
    n_grid = 20
    hidden_dim = 64
    n_passing = 2
    n_out = 2  # u, v predictions

    # Create model
    model = MPNN(
        n_passing=n_passing,
        lead_hrs=6,
        n_node_features_m=n_hours * n_features,
        n_node_features_e=n_hours * n_features,
        n_out_features=n_out,
        hidden_dim=hidden_dim,
    )
    print(f"Model created with {sum(p.numel() for p in model.parameters())} parameters")

    # Create synthetic inputs
    madis_x = torch.randn(n_batch, n_stations, n_hours, n_features)
    madis_lon = torch.randn(n_batch, n_stations, 1)
    madis_lat = torch.randn(n_batch, n_stations, 1)

    # Create edge index for stations (simple ring graph for testing)
    edges = []
    for i in range(n_stations):
        edges.append([i, (i + 1) % n_stations])
        edges.append([(i + 1) % n_stations, i])
    edge_index = torch.tensor(edges, dtype=torch.long).t()
    edge_index = edge_index.unsqueeze(0).expand(n_batch, -1, -1)

    # External (grid) inputs
    ex_x = torch.randn(n_batch, n_grid, n_hours, n_features)
    ex_lon = torch.randn(n_batch, n_grid, 1)
    ex_lat = torch.randn(n_batch, n_grid, 1)

    # Create edge index for grid-to-station (each station connects to 2 grid points)
    edges_e2m = []
    for i in range(n_stations):
        for j in range(2):
            edges_e2m.append([i % n_grid, i])
    edge_index_e2m = torch.tensor(edges_e2m, dtype=torch.long).t()
    edge_index_e2m = edge_index_e2m.unsqueeze(0).expand(n_batch, -1, -1)

    # Forward pass
    print("\nRunning forward pass...")
    with torch.no_grad():
        output = model(
            madis_x=madis_x,
            madis_lon=madis_lon,
            madis_lat=madis_lat,
            edge_index=edge_index,
            ex_lon=ex_lon,
            ex_lat=ex_lat,
            ex_x=ex_x,
            edge_index_e2m=edge_index_e2m,
        )

    print(f"Input shape: {madis_x.shape}")
    print(f"Output shape: {output.shape}")
    assert output.shape == (n_batch, n_stations, n_out)
    print("\nMPNN forward pass: PASSED")


def test_vit_forward():
    """Test VisionTransformer forward pass."""
    print("\n" + "=" * 60)
    print("Testing VisionTransformer Forward Pass")
    print("=" * 60)

    # Model parameters
    n_batch = 2
    n_stations = 10
    madis_len = 24
    n_vars_i = 4
    n_vars_o = 2
    dim = 64
    attn_dim = 32
    mlp_dim = 128
    num_heads = 2
    num_layers = 2

    # Create model
    model = VisionTransformer(
        n_stations=n_stations,
        madis_len=madis_len,
        madis_n_vars_i=n_vars_i,
        madis_n_vars_o=n_vars_o,
        dim=dim,
        attn_dim=attn_dim,
        mlp_dim=mlp_dim,
        num_heads=num_heads,
        num_layers=num_layers,
    )
    print(f"Model created with {sum(p.numel() for p in model.parameters())} parameters")

    # Create synthetic input
    madis_x = torch.randn(n_batch, n_stations, madis_len, n_vars_i)

    # Forward pass
    print("\nRunning forward pass...")
    with torch.no_grad():
        output, attns = model(madis_x, return_attn=True)

    print(f"Input shape: {madis_x.shape}")
    print(f"Output shape: {output.shape}")
    print(f"Attention shape: {attns.shape}")
    assert output.shape == (n_batch, n_stations, n_vars_o)
    assert attns.shape == (n_batch, num_layers, num_heads, n_stations, n_stations)
    print("\nVisionTransformer forward pass: PASSED")


def test_build_networks():
    """Test the build_networks convenience function."""
    print("\n" + "=" * 60)
    print("Testing build_networks Function")
    print("=" * 60)

    # Check if torch-cluster is available
    try:
        import torch_cluster
    except ImportError:
        print("Skipping (requires torch-cluster for KNN)")
        print("\nbuild_networks: SKIPPED")
        return

    # Station locations
    n_stations = 15
    station_lons = np.random.uniform(-122.5, -121.5, n_stations).astype(np.float32)
    station_lats = np.random.uniform(47.0, 48.0, n_stations).astype(np.float32)

    # ERA5 grid (meshgrid)
    era5_lons_1d = np.linspace(-124.0, -121.0, 8).astype(np.float32)
    era5_lats_1d = np.linspace(46.5, 49.0, 6).astype(np.float32)
    era5_lons, era5_lats = np.meshgrid(era5_lons_1d, era5_lats_1d)

    # HRRR grid (denser, meshgrid)
    hrrr_lons_1d = np.linspace(-124.0, -121.0, 20).astype(np.float32)
    hrrr_lats_1d = np.linspace(46.5, 49.0, 15).astype(np.float32)
    hrrr_lons, hrrr_lats = np.meshgrid(hrrr_lons_1d, hrrr_lats_1d)

    station_net, era5_net, hrrr_net = build_networks(
        station_lons=station_lons,
        station_lats=station_lats,
        era5_lons=era5_lons,
        era5_lats=era5_lats,
        hrrr_lons=hrrr_lons,
        hrrr_lats=hrrr_lats,
        n_neighbors_m2m=5,
        n_neighbors_e2m=4,
        n_neighbors_h2m=4,
    )

    print(f"Station network: {station_net.n_stations} stations, {station_net.edge_index.shape[1]} edges")
    print(f"ERA5 network: {era5_net.n_grid} grid points, {era5_net.edge_index.shape[1]} edges")
    print(f"HRRR network: {hrrr_net.n_grid} grid points, {hrrr_net.edge_index.shape[1]} edges")
    print("\nbuild_networks: PASSED")


def main():
    """Run all tests."""
    print("\n" + "#" * 60)
    print("# LOAF Model Forward Pass Tests")
    print("#" * 60)

    test_network_construction()
    test_mpnn_forward()
    test_vit_forward()
    test_build_networks()

    print("\n" + "=" * 60)
    print("ALL TESTS PASSED!")
    print("=" * 60 + "\n")


if __name__ == "__main__":
    main()
