import networkx as nx
import torch
import random
import numpy as np

def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

set_seed(42)  # 你可以自定义这个数字


def generate_bipartite_graph(num_left=10, num_right=6):
    """
    生成二分图，左侧为用户，右侧为商品
    """
    B = nx.Graph()
    left_nodes = [f"user_{i}" for i in range(num_left)]
    right_nodes = [f"item_{j}" for j in range(num_right)]

    B.add_nodes_from(left_nodes, bipartite=0)
    B.add_nodes_from(right_nodes, bipartite=1)

    for u in left_nodes:
        for v in random.sample(right_nodes, k=random.randint(1, 3)):
            B.add_edge(u, v)
    return B

def assign_node_features(G):
    features = {}
    for node in G.nodes():
        features[node] = torch.rand(2)
    return features

def assign_node_labels(G):
    labels = {}
    for i, node in enumerate(G.nodes()):
        labels[node] = i % 3  # 随机 3 类
    return labels