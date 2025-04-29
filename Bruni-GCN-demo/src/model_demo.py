import matplotlib.pyplot as plt
import torch
import torch.nn.functional as F
from torch_geometric.nn import GCNConv

loss_list = []


def get_data():
    from src.data_generator import generate_bipartite_graph, assign_node_features, assign_node_labels
    from src.Graph_conventrer import convert_nx_to_pyg

    G = generate_bipartite_graph()
    features = assign_node_features(G)
    labels = assign_node_labels(G)
    data = convert_nx_to_pyg(G, features, labels)
    return data


class GCN(torch.nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels):
        super().__init__()
        self.conv1 = GCNConv(in_channels, hidden_channels)
        self.conv2 = GCNConv(hidden_channels, out_channels)

    def forward(self, data):
        x, edge_index = data.x, data.edge_index
        x = F.relu(self.conv1(x, edge_index))
        x = self.conv2(x, edge_index)
        return F.log_softmax(x, dim=1)


def train(model, data):
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
    model.train()

    # 训练循环
    for epoch in range(201):  # 遍历 201 个 epoch
        optimizer.zero_grad()
        out = model(data)  # 前向传播
        loss = F.nll_loss(out, data.y)  # 损失计算
        loss.backward()  # 反向传播
        optimizer.step()  # 优化器更新

        # 记录 loss
        loss_list.append(loss.item())

        # 打印 loss 信息（每 50 轮输出一次）
        if epoch % 50 == 0:
            print(f"Epoch {epoch:03d}, Loss: {loss:.4f}")

    # 静态绘制最终的 Loss 曲线
    plt.figure(figsize=(8, 6))
    plt.plot(loss_list, label="Training Loss", color='blue')
    plt.title("Training Loss Curve")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.legend()
    plt.show()

def test(model, data):
    model.eval()
    pred = model(data).argmax(dim=1)
    acc = (pred == data.y).sum().item() / data.num_nodes
    print(f"Accuracy: {acc:.4f}")
