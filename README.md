# ViT implimentation of BDH (Baby Dragon Hatchling) introduced by pathway

FILE STRUCTURE

BDH_ViT
 |
 |--- attention.py
 |



FILES
 
Implement O(N) BDH Linear Attention (attention.py)

- Replaced Softmax with L1 feature normalization for linear complexity.
- Implemented global Linear Attention (K^T * V) context aggregation.
- Added sparse expansion layer to induce scale-free 'hub' formation.

HOW TO