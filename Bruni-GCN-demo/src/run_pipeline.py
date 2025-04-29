from src.model_demo import get_data, GCN, train, test

print("✅ Running Bruni-GCN Demo")

data = get_data()
model = GCN(in_channels=data.num_node_features, hidden_channels=32, out_channels=3)
train(model, data)
test(model, data)

