"""The GNN layers' per-layer normalization must be selectable, default off.

Regression coverage for the 3-month Arlington run that trained fine, loaded
fine, and forecast nothing: the ported layers end with torch_geometric's
InstanceNorm, which standardizes every feature channel across the station
nodes of each graph. With 5 stations that removes the regional wind level
every station shares, so the output barely depends on the grid input at all.
See loaf.model.gnn.norm.
"""

import pytest
import torch
from loaf.model.gnn import MPNN
from loaf.model.gnn.norm import GNNNorm
from loaf.training import build_model
from torch_geometric.nn import InstanceNorm

N_STATIONS, N_GRID, HRS, GRID_HRS, N_VARS = 5, 12, 4, 6, 2


def _inputs(seed: int = 0):
    g = torch.Generator().manual_seed(seed)
    madis_x = torch.randn(1, N_STATIONS, HRS, N_VARS, generator=g)
    lon = torch.rand(1, N_STATIONS, 1, generator=g)
    lat = torch.rand(1, N_STATIONS, 1, generator=g)
    src = torch.arange(N_STATIONS).repeat_interleave(2)
    dst = (src + torch.tensor([1, 2]).repeat(N_STATIONS)) % N_STATIONS
    edge_index = torch.stack([src, dst]).unsqueeze(0)
    ex_x = torch.randn(1, N_GRID, GRID_HRS, N_VARS, generator=g)
    ex_lon = torch.rand(1, N_GRID, 1, generator=g)
    ex_lat = torch.rand(1, N_GRID, 1, generator=g)
    e_src = torch.arange(N_GRID) % N_GRID
    e_dst = torch.arange(N_GRID) % N_STATIONS
    e2m = torch.stack([e_src, e_dst]).unsqueeze(0)
    return madis_x, lon, lat, edge_index, ex_lon, ex_lat, ex_x, e2m


def _mpnn(norm: str) -> MPNN:
    torch.manual_seed(0)
    return MPNN(
        n_passing=2,
        lead_hrs=0,
        n_node_features_m=HRS * N_VARS,
        n_node_features_e=GRID_HRS * N_VARS,
        n_out_features=2,
        hidden_dim=16,
        norm=norm,
    )


class TestGNNNorm:
    def test_rejects_unknown_kind(self) -> None:
        with pytest.raises(ValueError, match="gnn_norm"):
            GNNNorm("batch", 8)

    @pytest.mark.parametrize("kind", ["none", "layer", "instance"])
    def test_every_layer_gets_the_requested_norm(self, kind: str) -> None:
        model = _mpnn(kind)
        norms = [m for m in model.modules() if isinstance(m, GNNNorm)]
        assert len(norms) == 4  # 2 external + 2 internal layers
        assert all(n.kind == kind for n in norms)
        has_instance = any(isinstance(m, InstanceNorm) for m in model.modules())
        assert has_instance == (kind == "instance")

    def test_instance_norm_erases_a_level_shared_by_all_stations(self) -> None:
        """The mechanism: a feature every station in a graph has in common
        (e.g. the regional wind level the grid layer just pulled in) is
        exactly what cross-node InstanceNorm subtracts. Per-node LayerNorm
        and no normalization keep it."""
        n_graphs, n_nodes, dim = 2, N_STATIONS, 8
        shared = torch.arange(1.0, dim + 1).repeat(n_graphs * n_nodes, 1)  # same on every node
        batch = torch.arange(n_graphs).repeat_interleave(n_nodes)

        gone = GNNNorm("instance", dim)(shared, batch)
        assert torch.allclose(gone, torch.zeros_like(shared), atol=1e-5)

        kept = GNNNorm("none", dim)(shared, batch)
        assert torch.equal(kept, shared)

        per_node = GNNNorm("layer", dim)(shared, batch)
        assert not torch.allclose(per_node, torch.zeros_like(shared), atol=1e-3)


class TestBuildModelNorm:
    def test_config_key_is_passed_through(self) -> None:
        model = build_model(
            {"type": "mpnn", "hidden_dim": 16, "num_gnn_layers": 1, "gnn_norm": "layer"},
            n_stations=N_STATIONS, in_hrs=HRS, n_station_vars=N_VARS, n_out_features=2,
        )
        kinds = {m.kind for m in model.modules() if isinstance(m, GNNNorm)}
        assert kinds == {"layer"}

    def test_checkpoint_without_key_rebuilds_with_original_instance_norm(self) -> None:
        model = build_model(
            {"type": "mpnn", "hidden_dim": 16, "num_gnn_layers": 1},
            n_stations=N_STATIONS, in_hrs=HRS, n_station_vars=N_VARS, n_out_features=2,
        )
        kinds = {m.kind for m in model.modules() if isinstance(m, GNNNorm)}
        assert kinds == {"instance"}
