
import json
from pathlib import Path

import numpy as np


def load_graph(json_path):
    """Load an extracted APK graph."""
    with open(json_path, "r", encoding="utf-8") as file:
        return json.load(file)


def build_api_vocabulary(graph):
    """Create a shared index for every unique API call."""
    vocabulary = set()

    for method in graph["methods"]:
        vocabulary.update(method["api_calls"])

    return {
        api_call: index
        for index, api_call in enumerate(sorted(vocabulary))
    }


def encode_node_features(graph, vocabulary):
    """Convert each method's API calls into a multi-hot vector."""
    node_count = graph["node_count"]
    feature_count = len(vocabulary)

    x = np.zeros((node_count, feature_count), dtype=np.float32)

    for method in graph["methods"]:
        node_id = method["id"]

        for api_call in method["api_calls"]:
            feature_index = vocabulary[api_call]
            x[node_id][feature_index] = 1.0

    return x


def encode_edges(graph):
    """Convert graph relationships into source-target pairs."""
    edges = graph["edges"]

    if not edges:
        return np.empty((2, 0), dtype=np.int64)

    edge_index = np.array(
        [
            [edge["source"] for edge in edges],
            [edge["target"] for edge in edges]
        ],
        dtype=np.int64
    )

    return edge_index


def process_graph(json_path):
    """Create numerical node features and graph connectivity."""
    graph = load_graph(json_path)

    vocabulary = build_api_vocabulary(graph)

    x = encode_node_features(graph, vocabulary)
    edge_index = encode_edges(graph)

    # No APK label is assigned at this stage.
    encoded_data = {
        "apk_name": graph["apk_name"],
        "node_count": graph["node_count"],
        "edge_count": graph["edge_count"],
        "feature_count": len(vocabulary),
        "x": x.tolist(),
        "edge_index": edge_index.tolist()
    }

    project_root = Path(__file__).resolve().parents[2]

    output_dir = project_root / "data" / "processed" / "features"
    output_dir.mkdir(parents=True, exist_ok=True)

    output_file = output_dir / f"{Path(json_path).stem}_encoded.json"

    with open(output_file, "w", encoding="utf-8") as file:
        json.dump(encoded_data, file, indent=4)

    print("\n========== FEATURE ENCODING SUMMARY ==========")
    print(f"APK: {graph['apk_name']}")
    print(f"Nodes: {x.shape[0]}")
    print(f"Unique API features: {x.shape[1]}")
    print(f"Feature matrix shape: {x.shape}")
    print(f"Edge index shape: {edge_index.shape}")
    print(f"Encoded graph saved to: {output_file}")
    print("==============================================")

    return encoded_data


if __name__ == "__main__":
    project_root = Path(__file__).resolve().parents[2]

    input_file = (
        project_root
        / "data"
        / "processed"
        / "graphs"
        / "app-debug.json"
    )

    process_graph(input_file)
