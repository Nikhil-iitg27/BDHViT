import torch
import torch.nn as nn
import torch.nn.functional as F

class BDHLinearAttention(nn.Module):
    
    def __init__(self, dim: int, expansion_factor=4, heads=8):
        super().__init__()
        self.dim = dim
        self.heads = heads
        self.sparse_dim = self.dim * expansion_factor
        self.head_dim = self.sparse_dim // heads
        
        # Q, K, V combined in one layer
        self.in_proj = nn.Linear(dim, self.sparse_dim * 3)
        self.out_proj = nn.Linear(self.sparse_dim, dim)
        
        self.norm = nn.LayerNorm(dim)
        
    def forward(self, x: torch.Tensor):
        """
            x (torch.Tensor): [Batch, SequenceLen, Dim]
        """
        B, N, D = x.shape
        H = self.heads
        
        qkv = self.in_proj(x)
        q, k, v = qkv.chunk(3, dim=-1)
        
        # BDH Like Sparse Activations (operations on "active" neurons)
        q = F.relu(q).view(B, N, H, -1).transpose(1, 2)
        k = F.relu(k).view(B, N, H, -1).transpose(1, 2)
        v = v.view(B, N, H, -1).transpose(1, 2)
        
        # Normalization replaces softmax
        k = k / (k.sum(dim=-1, keepdim=True) + 1e-6)
        
        # Compute the Context (Globally for ViT) via Synapses
        kT = k.transpose(-2, -1)
        state = torch.matmul(kT, v) # [B, H, Head_Dim, Head_Dim]
        
        y = torch.matmul(q, state) # [B, H, N, Head_Dim]
        y = y.transpose(1, 2).contiguous().view(B, N, self.sparse_dim)
        
        out = self.out_proj(y)
        return out
        
        
        