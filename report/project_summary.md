# Bruni-GCN Project Summary

## Problem Statement

This project explores the use of Graph Convolutional Networks (GCNs) to perform node classification on **custom-generated bipartite graphs**. Unlike traditional experiments on fixed datasets (like Cora or Citeseer), this work simulates a dynamic graph structure and feature-label assignment process from scratch.

## Objectives

- Implement a minimal yet modular GCN pipeline with PyTorch Geometric
- Generate synthetic bipartite graphs with controllable node sets
- Build a complete data flow: graph → features → labels → PyG format
- Train a 2-layer GCN on the synthetic dataset
- Visualize training performance through loss curves

## Experimental Setup

- GCN Model: 2-layer, ReLU activation, log-softmax output
- Loss: Negative Log Likelihood (NLL)
- Optimizer: Adam, LR=0.01
- Training Epochs: 200
- Dataset: Random bipartite graph generated per run

## Results Summary

Example training result (with fixed seed):

- Final Loss: ~0.7193
- Final Accuracy: 75%
- Loss curve saved at `figures/loss_curve.png`

## Key Insights

- Graphs generated from random bipartite patterns can still exhibit meaningful structure for GNN training.
- A properly configured GCN is able to converge and generalize on synthetic data.
- Loss stability can be improved by setting random seeds and tuning graph generation logic.

## Future Improvements

- Add accuracy curve (train & validation)
- Include dropout & layer normalization
- Integrate attention mechanism (GAT)
- Switch to real-world datasets (e.g. Cora, PubMed)
- Experiment with node attribute noise and class imbalance

## itation / References

- Kipf & Welling (2017), Semi-Supervised Classification with GCNs
- PyTorch Geometric Documentation
- NetworkX API Reference

---

*Author: Bruni (Liaoning University, 2025)*
