"""
Graph Neural Network (GCN) on the Cora Dataset using PyTorch Geometric
"""

import os
import torch
import torch.nn.functional as F
from torch_geometric.datasets import Planetoid
from torch_geometric.nn import GCNConv
from sklearn.manifold import TSNE
import numpy as np
import matplotlib.pyplot as plt
from tqdm import trange

# Select device
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", DEVICE)

# Load Cora Dataset
dataset = Planetoid(root="data", name="Cora")
data = dataset[0].to(DEVICE)

# -----------------------------
# DEFINE GCN MODEL
# -----------------------------
class GCN(torch.nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels, dropout=0.5):
        super().__init__()
        self.conv1 = GCNConv(in_channels, hidden_channels)
        self.conv2 = GCNConv(hidden_channels, out_channels)
        self.dropout = dropout

    def forward(self, x, edge_index):
        x = self.conv1(x, edge_index)
        x = F.relu(x)
        x = F.dropout(x, p=self.dropout, training=self.training)
        x = self.conv2(x, edge_index)
        return x

# Initialize Model
model = GCN(
    in_channels=dataset.num_node_features,
    hidden_channels=16,
    out_channels=dataset.num_classes
).to(DEVICE)

optimizer = torch.optim.Adam(model.parameters(), lr=0.01, weight_decay=5e-4)

# Training function
def train():
    model.train()
    optimizer.zero_grad()
    out = model(data.x, data.edge_index)
    loss = F.cross_entropy(out[data.train_mask], data.y[data.train_mask])
    loss.backward()
    optimizer.step()
    return loss.item()

# Testing function
@torch.no_grad()
def test():
    model.eval()
    out = model(data.x, data.edge_index)
    pred = out.argmax(dim=1)

    accs = []
    for mask in [data.train_mask, data.val_mask, data.test_mask]:
        correct = pred[mask].eq(data.y[mask]).sum().item()
        acc = correct / mask.sum().item()
        accs.append(acc)
    return accs

# -----------------------------
# TRAINING LOOP
# -----------------------------
epochs = 200
loss_history = []
train_acc_history = []
val_acc_history = []
test_acc_history = []

for epoch in trange(1, epochs + 1):
    loss = train()
    train_acc, val_acc, test_acc = test()

    loss_history.append(loss)
    train_acc_history.append(train_acc)
    val_acc_history.append(val_acc)
    test_acc_history.append(test_acc)

    if epoch % 20 == 0:
        print(f"Epoch {epoch}, Loss={loss:.4f}, Train={train_acc:.4f}, Val={val_acc:.4f}, Test={test_acc:.4f}")


# PRINT FINAL ACCURACY (ADD THIS)
# -----------------------------
print("\nFinal Accuracy Results:")
print(f"Train Accuracy: {train_acc_history[-1]:.4f}")
print(f"Val Accuracy:   {val_acc_history[-1]:.4f}")
print(f"Test Accuracy:  {test_acc_history[-1]:.4f}")

# -----------------------------
# SAVE MODEL
# -----------------------------
os.makedirs("models", exist_ok=True)
torch.save(model.state_dict(), "models/gcn_cora.pt")
print("Model saved to models/gcn_cora.pt")

# -----------------------------
# PLOT TRAINING RESULTS
# -----------------------------
plt.figure(figsize=(10,4))
plt.subplot(1,2,1)
plt.plot(loss_history)
plt.title("Training Loss")

plt.subplot(1,2,2)
plt.plot(train_acc_history, label="train")
plt.plot(val_acc_history, label="val")
plt.plot(test_acc_history, label="test")
plt.legend()
plt.title("Accuracy")

plt.tight_layout()
plt.savefig("training_plots.png")
print("Saved training_plots.png")

plt.show()

# -----------------------------
# GENERATE t-SNE VISUALIZATION
# -----------------------------
model.eval()
with torch.no_grad():
    logits = model(data.x, data.edge_index)
    embeddings = logits.cpu().numpy()

tsne = TSNE(n_components=2, perplexity=30, learning_rate=200)
embed_2d = tsne.fit_transform(embeddings)

plt.figure(figsize=(6, 6))
plt.scatter(embed_2d[:, 0], embed_2d[:, 1], c=data.y.cpu(), cmap="tab10", s=8)
plt.title("t-SNE Plot of Node Embeddings")
plt.savefig("tsne_plot.png")
print("Saved tsne_plot.png")
plt.close()
