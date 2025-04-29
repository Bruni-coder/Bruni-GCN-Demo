# Bruni-GCN Demo: Node Classification on Custom Bipartite Graphs (with PyTorch Geometric)

This project demonstrates a clean, modular implementation of a Graph Convolutional Network (GCN) using [PyTorch Geometric (PyG)](https://pytorch-geometric.readthedocs.io/en/latest/) on **randomly generated bipartite graphs**. It serves as a lightweight yet complete research-style starter kit for graph-based node classification tasks.

---

##  Features

-  **Random bipartite graph generation** using `networkx`
-  **Manual node feature & label construction** for full control
-  **Graph conversion** from NetworkX to PyG `Data` format
-  **GCN implementation** using `GCNConv`
-  **One-click training pipeline** via `run_pipeline.py`
-  **Loss curve visualization** after training
-  **Project folder structure** ready for academic expansion
-  Minimal setup, easily portable

---

##  Project Structure

```bash
.
Bruni-GCN-demo/
├── src/                                      
│   ├── data_generator.py         
│   ├── Graph_conventrer.py       
│   ├── model_demo.py            
│   ├── run_pipeline.py          
│   └── plot_training_curves.py   
│
├── figures/                     
│   └── loss_curve.png
│
├── report/                       
│   ├── project_summary.md      
│   ├── improvement_ideas.md      
│   └── reference_list.md         
│
├── requirements.txt              
├── LICENSE                       
├── README.md                    
└── .gitignore                    


```

---
## Samples Output
✅ Running Bruni-GCN Demo
- Epoch 000, Loss: 1.1041
- Epoch 050, Loss: 0.9842
- Epoch 100, Loss: 0.8142
- Epoch 200, Loss: 0.7193
- Accuracy: 0.7500
---
## Run Training
```bash
python src/run_pipeline.py
```
- This will train a 2-layer GCN on a randomly generated bipartite graph.
- The loss and accuracy will be printed every 50 epochs.
---
## Training Loss Curve
After training, you can generate the training loss plot by running
```bash
python src/plot_training_curves.py
```
---
## License

This project is open-source under the MIT License.

---
See [Improvement Ideas](report/improvement_ideas.md) for potential future directions.
