# ViT implimentation of BDH (Baby Dragon Hatchling) introduced by pathway

## > FILE STRUCTURE

BDH_ViT/
├── attention.py
└── backbone.py

## > FILES
 
### Implemented O(N) BDH Linear Attention (attention.py)

- Replaced Softmax with L1 feature normalization for linear complexity.
- Implemented global Linear Attention (K^T * V) context aggregation.
- Added sparse expansion layer with power law scaling to induce scale-free 'hub' formation.
- Added gated Persistant Memory structure for memory preservation.
- Added Task Conditional to determine what Memory stored is actually important

### Implemented Standard BDH ViT Block and Backbone
- Used both forward and backward Feature extraction with BDH Linear Attention Block
- Added Skip Connections so that no gradient vanishes and information is preserved across layers
- Patched up and linearized the image features for Attention to work

## > HOW TO