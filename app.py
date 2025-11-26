# app.py
import os
import time
import numpy as np
import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt
import streamlit as st

from tqdm import trange
from sklearn.manifold import TSNE

# PyG imports
from torch_geometric.datasets import Planetoid
from torch_geometric.nn import GCNConv, GATConv

# ----- Config -----
DATA_ROOT = "data"
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
os.makedirs("models", exist_ok=True)

st.set_page_config(page_title="GNN Cora Live Trainer", layout="wide")

# -------------------------------------------
# 📌 PROJECT INTRODUCTION (Shown at Top)
# -------------------------------------------

st.markdown("""
# 🔥 Graph Neural Network (GNN) — Cora Real-Time Trainer

This project demonstrates how **Graph Neural Networks (GCN + GAT)** can learn from  
the **Cora Citation Network**, where:

- **Nodes = Research papers**  
- **Edges = Citation links**  
- **Goal = Predict the topic/category of each research paper**  

### 🧠 Why GNN?
Traditional neural networks cannot understand relationships.  
GNNs allow each node to learn from its neighbors, enabling tasks like:

- Social network analysis  
- Fraud detection  
- Molecular drug discovery  
- Recommendation systems  
- Citation network classification  

### 🔷 Models Used
#### ✅ **GCN — Graph Convolutional Network**
GCN = *Graph Convolutional Network*  
✔ Learns by aggregating info from neighbors  
✔ Fast & simple  
✔ Great for structured datasets like Cora  

#### ✅ **GAT — Graph Attention Network**
GAT = *Graph Attention Network*  
✔ Uses **self-attention** to learn importance of neighbors  
✔ More expressive than GCN  
✔ Similar to Transformer attention  

### 🎯 What this App Can Do
- Train **GCN or GAT** in real time  
- Select **epochs, learning rate & hidden dim**  
- View **live metrics** (loss, accuracy per epoch)  
- Generate **t-SNE embedding visualization**  
- Perform **node-level prediction**  
- Save & load trained models  
- Full interactive dashboard for ML + Deep Learning interviews  

---
""")


# ----- Helper: Load dataset -----
@st.cache_data(show_spinner=False)
def load_data():
    dataset = Planetoid(root=DATA_ROOT, name="Cora")
    data = dataset[0]
    return dataset, data

dataset, data = load_data()
data = data.to(DEVICE)

# ----- Model definitions -----
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

class GAT(torch.nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels, heads=8, dropout=0.6):
        super().__init__()
        self.conv1 = GATConv(in_channels, hidden_channels, heads=heads, dropout=dropout)
        self.conv2 = GATConv(hidden_channels * heads, out_channels, heads=1, concat=False, dropout=dropout)
        self.dropout = dropout

    def forward(self, x, edge_index):
        x = self.conv1(x, edge_index)
        x = F.elu(x)
        x = F.dropout(x, p=self.dropout, training=self.training)
        x = self.conv2(x, edge_index)
        return x

# ----- UI: Sidebar controls -----
st.sidebar.title("Training Controls")
model_choice = st.sidebar.selectbox("Model", ["GCN", "GAT"])
epochs = st.sidebar.slider("Epochs", min_value=10, max_value=500, value=100, step=10)
lr = st.sidebar.select_slider("Learning rate", options=[0.001, 0.005, 0.01, 0.02], value=0.01)
hidden = st.sidebar.select_slider("Hidden dim", options=[8, 16, 32, 64], value=16)
batch_dummy = st.sidebar.checkbox("Use CPU mode (no GPU even if available)", value=False)
if batch_dummy:
    DEVICE = torch.device("cpu")

st.sidebar.markdown("---")
st.sidebar.write("Model files saved in `models/`")
if st.sidebar.button("Load last saved model"):
    model_path = "models/gcn_cora.pt" if model_choice == "GCN" else "models/gat_cora.pt"
    if os.path.exists(model_path):
        st.sidebar.success(f"Found {model_path}. It will be loaded after you press 'Start Training'.")
    else:
        st.sidebar.warning(f"No saved model at {model_path}")

# ----- Main layout -----
st.title("GNN on Cora — Live Trainer & Visualizer")
st.write("Interactive training demo. Choose model, epochs and press Start Training.")

col1, col2 = st.columns((1, 1))

with col1:
    st.subheader("Model & Metrics")
    st.write(f"Dataset: **Cora** — Nodes: {data.num_nodes}, Features: {data.num_node_features}, Classes: {dataset.num_classes}")
    model_info = st.empty()
    metrics_box = st.empty()
    progress_bar = st.empty()
    # place for live metrics
    train_loss_text = st.empty()
    train_acc_text = st.empty()
    val_acc_text = st.empty()
    test_acc_text = st.empty()

with col2:
    st.subheader("Visualizations")
    loss_chart_area = st.empty()
    tsne_area = st.empty()
    acc_chart_area = st.empty()

# ----- Utilities -----
def build_model(name):
    if name == "GCN":
        return GCN(in_channels=dataset.num_node_features, hidden_channels=hidden, out_channels=dataset.num_classes).to(DEVICE)
    else:
        return GAT(in_channels=dataset.num_node_features, hidden_channels=hidden, out_channels=dataset.num_classes).to(DEVICE)

@torch.no_grad()
def evaluate(model):
    model.eval()
    out = model(data.x.to(DEVICE), data.edge_index.to(DEVICE))
    pred = out.argmax(dim=1)
    accs = []
    for mask in [data.train_mask, data.val_mask, data.test_mask]:
        mask = mask.to(DEVICE)
        correct = pred[mask].eq(data.y[mask].to(DEVICE)).sum().item()
        acc = correct / int(mask.sum().item())
        accs.append(acc)
    return accs, out

def compute_tsne(embeddings_cpu, labels_cpu):
    try:
        tsne = TSNE(n_components=2, perplexity=30, learning_rate=200, init="pca")
        emb2 = tsne.fit_transform(embeddings_cpu)
        return emb2
    except Exception as e:
        st.error(f"t-SNE failed: {e}")
        return None

# ----- Training routine (runs in the app) -----
def train_and_visualize(model, epochs, lr):
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=5e-4)
    loss_history = []
    train_acc_history = []
    val_acc_history = []
    test_acc_history = []

    pbar = progress_bar.progress(0)
    model_info.info(f"Training {model_choice} — hidden={hidden}, lr={lr}, device={DEVICE}")

    for epoch in range(1, epochs + 1):
        model.train()
        optimizer.zero_grad()
        out = model(data.x.to(DEVICE), data.edge_index.to(DEVICE))
        loss = F.cross_entropy(out[data.train_mask.to(DEVICE)], data.y[data.train_mask.to(DEVICE)])
        loss.backward()
        optimizer.step()

        # evaluate
        (train_acc, val_acc, test_acc), logits = evaluate(model)

        loss_history.append(loss.item())
        train_acc_history.append(train_acc)
        val_acc_history.append(val_acc)
        test_acc_history.append(test_acc)

        # update live texts
        train_loss_text.markdown(f"**Epoch {epoch}/{epochs}** — Loss: `{loss.item():.4f}`")
        train_acc_text.markdown(f"**Train Acc:** {train_acc:.4f}")
        val_acc_text.markdown(f"**Val Acc:** {val_acc:.4f}")
        test_acc_text.markdown(f"**Test Acc:** {test_acc:.4f}")

        # update progress bar
        pbar.progress(int(epoch / epochs * 100))

        # update loss chart
        fig_loss, ax_loss = plt.subplots(figsize=(5, 3))
        ax_loss.plot(loss_history, label="loss")
        ax_loss.set_xlabel("Epoch")
        ax_loss.set_ylabel("Loss")
        ax_loss.legend()
        loss_chart_area.pyplot(fig_loss)
        plt.close(fig_loss)

        # update accuracy chart
        fig_acc, ax_acc = plt.subplots(figsize=(5, 3))
        ax_acc.plot(train_acc_history, label="train")
        ax_acc.plot(val_acc_history, label="val")
        ax_acc.plot(test_acc_history, label="test")
        ax_acc.set_xlabel("Epoch")
        ax_acc.set_ylabel("Accuracy")
        ax_acc.legend()
        acc_chart_area.pyplot(fig_acc)
        plt.close(fig_acc)

        # small sleep so UI remains responsive (and user can cancel)
        time.sleep(0.01)

    # finalize
    pbar.progress(100)
    st.success("Training completed")

    # Save model
    model_name = "gcn_cora.pt" if model_choice == "GCN" else "gat_cora.pt"
    save_path = os.path.join("models", model_name)
    torch.save(model.state_dict(), save_path)
    st.info(f"Model saved to `{save_path}`")

    # compute and show final t-SNE
    model.eval()
    with torch.no_grad():
        final_logits = model(data.x.to(DEVICE), data.edge_index.to(DEVICE))
        embeddings = final_logits.cpu().numpy()
        labels = data.y.cpu().numpy()

    emb2 = compute_tsne(embeddings, labels)
    if emb2 is not None:
        fig_tsne, ax_tsne = plt.subplots(figsize=(6, 6))
        sc = ax_tsne.scatter(emb2[:, 0], emb2[:, 1], c=labels, s=10, cmap="tab10")
        ax_tsne.set_title("t-SNE of Node Embeddings (final)")
        plt.colorbar(sc, ax=ax_tsne)
        tsne_area.pyplot(fig_tsne)
        fig_tsne.savefig("tsne_plot.png")
        plt.close(fig_tsne)
        st.success("t-SNE generated and saved as tsne_plot.png")

    # final metrics box
    metrics_box.markdown("### Final metrics")
    metrics_box.write({
        "train_acc": train_acc_history[-1],
        "val_acc": val_acc_history[-1],
        "test_acc": test_acc_history[-1],
        "final_loss": loss_history[-1]
    })

    return model, (train_acc_history, val_acc_history, test_acc_history, loss_history)

# ----- Run training on click -----
train_button = st.button("Start Training")

if train_button:
    # build model fresh
    model = build_model(model_choice)
    # Optionally load saved model weights if present
    model_path = os.path.join("models", "gcn_cora.pt" if model_choice == "GCN" else "gat_cora.pt")
    if st.sidebar.checkbox("Load existing model weights (if available)", value=False):
        if os.path.exists(model_path):
            model.load_state_dict(torch.load(model_path, map_location=DEVICE))
            st.sidebar.success(f"Loaded weights from {model_path}")
        else:
            st.sidebar.info("No saved weights found, starting from random init")

    model, histories = train_and_visualize(model, epochs, lr)

    st.balloons()

# ----- Node prediction panel -----
st.write("---")
st.subheader("Node prediction demo")
node_id = st.number_input("Select node id", min_value=0, max_value=data.num_nodes - 1, value=0, step=1)
predict_button = st.button("Predict for node")

if predict_button:
    # load best model (from selected model_choice) if exists, else use a fresh one
    model_path = os.path.join("models", "gcn_cora.pt" if model_choice == "GCN" else "gat_cora.pt")
    model = build_model(model_choice)
    if os.path.exists(model_path):
        model.load_state_dict(torch.load(model_path, map_location=DEVICE))
    model.to(DEVICE)
    model.eval()
    with torch.no_grad():
        out = model(data.x.to(DEVICE), data.edge_index.to(DEVICE))
        probs = torch.nn.functional.softmax(out, dim=1)
        node_probs = probs[node_id].cpu().numpy()
        pred = int(out[node_id].argmax().item())

    st.write(f"Predicted class for node **{node_id}**: **{pred}**")
    st.write("Class probabilities (first 10 shown):")
    st.write(node_probs[:min(len(node_probs), 10)])

# ----- Show saved artifacts -----
st.write("---")
st.write("Saved artifacts in project folder:")
files = os.listdir("models")
st.write(files)
if os.path.exists("tsne_plot.png"):
    st.image("tsne_plot.png", caption="Last t-SNE", use_column_width=True)
if os.path.exists("training_plots.png"):
    st.image("training_plots.png", caption="Training loss & accuracy", use_column_width=True)


# --------------------------------------
# 📌 Footer — Social Links
# --------------------------------------
st.write("---")
# Create a centered layout with columns for better visual appeal (adjusted for wider content)
col1, col2, col3 = st.columns([1, 3, 1])  # Increased middle column width for a less "big" background feel
with col2:
    st.markdown("""
    <div style="text-align: center; padding: 8px; background-color: #f0f2f6; border-radius: 9px; box-shadow: 0 4px 8px rgba(0,0,0,0.1); max-width: 600px; margin: 0 auto;">  <!-- Added max-width to control size -->
        <h3 style="color: #333; margin-bottom: 13px;">🔗 Connect With Me</h3>
        <div style="display: flex; justify-content: center; gap: 10px; flex-wrap: wrap;">  <!-- Horizontal layout for badges -->
            <a href="https://www.linkedin.com/in/riteshbhadana" target="_blank" style="text-decoration: none;">
                <img src="https://img.shields.io/badge/LinkedIn-0077B5?style=for-the-badge&logo=linkedin&logoColor=white" alt="LinkedIn">
            </a>
            <a href="https://github.com/riteshbhadana" target="_blank" style="text-decoration: none;">
                <img src="https://img.shields.io/badge/GitHub-100000?style=for-the-badge&logo=github&logoColor=white" alt="GitHub">
            </a>
        </div>
        <p style="margin-top: 17px; font-size: 10px; color: #666;">
            Made with ❤️ by <strong>Ritesh </strong>
        </p>
    </div>
    """, unsafe_allow_html=True)