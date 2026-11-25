
import json
from pathlib import Path

import torch
from torch_geometric.data import Data


def load_encoded_graph(json_path):
    """Load the numerically encoded graph."""
    with open(json_path, "r", encoding="utf-8") as file:
        return json.load(file)


def create_pyg_graph(graph):
    """Convert encoded graph data into a PyG Data object."""

    # Node feature matrix
    x = torch.tensor(
        graph["x"],
        dtype=torch.float32
    )

    # Graph connectivity
    edge_index = torch.tensor(
        graph["edge_index"],
        dtype=torch.long
    )

    # Create PyTorch Geometric graph
    data = Data(
        x=x,
        edge_index=edge_index
    )

    return data


def main():
    project_root = Path(__file__).resolve().parents[2]

    input_file = (
        project_root
        / "data"
        / "processed"
        / "features"
        / "app-debug_encoded.json"
    )

    graph = load_encoded_graph(input_file)

    data = create_pyg_graph(graph)

    print("\n========== PYTORCH GEOMETRIC GRAPH ==========")
    print(data)
    print("---------------------------------------------")
    print("Node features shape:", data.x.shape)
    print("Edge index shape:", data.edge_index.shape)
    print("Number of nodes:", data.num_nodes)
    print("Number of edges:", data.num_edges)
    print("Number of node features:", data.num_node_features)
    print("=============================================")

    # Basic structural validation
    assert data.x.shape[0] == data.num_nodes
    assert data.edge_index.shape[0] == 2
    assert data.edge_index.min() >= 0
    assert data.edge_index.max() < data.num_nodes

    print("\nGraph validation successful!")


if __name__ == "__main__":
    main()
