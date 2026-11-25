
import json
from pathlib import Path

import torch
import torch.nn.functional as F

from torch_geometric.data import Data
from torch_geometric.nn import GCNConv, global_mean_pool


class APKGCN(torch.nn.Module):

    def __init__(self, input_features, hidden_channels=128):
        super().__init__()

        # First graph convolution: 626 -> 128
        self.conv1 = GCNConv(
            input_features,
            hidden_channels
        )

        # Second graph convolution: 128 -> 64
        self.conv2 = GCNConv(
            hidden_channels,
            64
        )

        # APK-level classification layer
        self.classifier = torch.nn.Linear(64, 2)

    def encode(self, x, edge_index, batch):
        """Generate one graph representation per APK."""

        # First GCN layer
        x = self.conv1(x, edge_index)
        x = F.relu(x)

        # Second GCN layer
        x = self.conv2(x, edge_index)
        x = F.relu(x)

        # Combine method representations into APK representation
        graph_embedding = global_mean_pool(x, batch)

        return graph_embedding

    def forward(self, x, edge_index, batch):
        """Generate APK-level classification scores."""

        graph_embedding = self.encode(
            x,
            edge_index,
            batch
        )

        logits = self.classifier(graph_embedding)

        return logits


def main():

    project_root = Path(__file__).resolve().parents[2]

    input_file = (
        project_root
        / "data"
        / "processed"
        / "features"
        / "app-debug_encoded.json"
    )

    # Load encoded graph
    with open(input_file, "r", encoding="utf-8") as file:
        graph = json.load(file)

    # Convert graph data into PyTorch tensors
    data = Data(
        x=torch.tensor(graph["x"], dtype=torch.float32),
        edge_index=torch.tensor(
            graph["edge_index"],
            dtype=torch.long
        )
    )

    # All nodes belong to the same APK
    batch = torch.zeros(
        data.num_nodes,
        dtype=torch.long
    )

    # Initialize GCN model
    model = APKGCN(
        input_features=data.num_node_features
    )

    model.eval()

    # Test the model without calculating gradients
    with torch.no_grad():

        graph_embedding = model.encode(
            data.x,
            data.edge_index,
            batch
        )

        logits = model(
            data.x,
            data.edge_index,
            batch
        )

    print("\n========== GCN MODEL TEST ==========")
    print("Input features:", data.x.shape)
    print("Edge index:", data.edge_index.shape)
    print("First GCN output: [1926, 128]")
    print("Second GCN output: [1926, 64]")
    print("APK embedding:", graph_embedding.shape)
    print("Classification logits:", logits.shape)
    print("====================================")


if __name__ == "__main__":
    main()
