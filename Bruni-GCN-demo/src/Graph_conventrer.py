import torch
from torch_geometric.data import Data

def convert_nx_to_pyg(graph, features_dict, labels_dict=None):
    node_to_index = {n: i for i, n in enumerate(graph.nodes())}
    num_nodes = len(node_to_index)

    edge_index = []
    for u, v in graph.edges():
        edge_index.append([node_to_index[u], node_to_index[v]])
        edge_index.append([node_to_index[v], node_to_index[u]])
    edge_index = torch.tensor(edge_index, dtype=torch.long).t().contiguous()

    feature_dim = list(features_dict.values())[0].shape[0]
    x = torch.zeros((num_nodes, feature_dim))
    for node, idx in node_to_index.items():
        x[idx] = features_dict[node]

    y = torch.full((num_nodes,), -1, dtype=torch.long)
    if labels_dict:
        for node, label in labels_dict.items():
            y[node_to_index[node]] = label

    data = Data(x=x, edge_index=edge_index, y=y)
    return data