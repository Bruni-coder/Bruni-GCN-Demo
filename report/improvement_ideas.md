# Ideas for Improvement and Future Work

This document outlines key limitations of the current GCN demo and proposes several research-oriented extensions to further enhance its academic and practical value.

---

## 1. Model Architecture

- Replace 2-layer GCN with **deeper or residual GCN**
- Add support for **GAT (Graph Attention Networks)** to analyze attention patterns
- Try **GraphSAGE**, **GIN**, or **ChebNet** to explore architecture diversity

---

## 2. Training Setup

- Add **validation split** and track val_loss, val_accuracy
- Implement **early stopping** based on validation performance
- Log metrics using **TensorBoard** or CSV export

---

## 3. Dataset & Input Graphs

- Extend beyond synthetic bipartite graphs:
  - Cora, Citeseer, PubMed from PyG datasets
  - Real-world bipartite networks (e.g., user–item)
- Inject **noise or imbalance** to simulate real-world data
- Add graph statistics visualization (degree, clustering)

---

## 4. Privacy & Federated Settings (for research goals)

- Simulate **multi-party federated training**
- Explore **differential privacy (DP)** or **homomorphic encryption (HE)** wrappers for edge nodes
- Train local subgraphs and aggregate gradients

---

## 5. Visualization

- Plot accuracy vs. loss in the same figure
- Add interactive graph exploration via NetworkX or D3.js
- Export graph snapshots before/after training

---

## 6. Academic Extensions

- Rewrite as a **modular research repo** for reuse in:
  - Federated GCN
  - Supply chain modeling
  - Privacy-preserving multi-agent RL
- Prepare short paper or slide deck for lab presentation or course project

---

*Maintained by Bruni. Last updated: 2025.*
