from .attention import BDHLinearAttention

import torch
import torch.nn as nn
import torch.nn.functional as F

class BDHBlock(nn.Module):
    """
    A Single Layer: Norm -> Attention -> Norm -> MLP
    """
    def __init__(self, dim: int, expansion_factor=4, dropout=0.1, layer_scale=1e-4):
        super().__init__()
        self.norm1 = nn.LayerNorm(dim)
        
        # Two BDH Attention Layers for Bi-Directional Context
        self.bdh_fwd = BDHLinearAttention(dim, expansion_factor)
        self.bdh_bwd = BDHLinearAttention(dim, expansion_factor)
        self.layer_scale_1 = nn.Parameter(layer_scale * torch.ones(dim), requires_grad=True)
        
        self.norm2 = nn.LayerNorm(dim)
        
        self.mlp = nn.Sequential(
            nn.Linear(dim, dim * 4),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(dim * 4, dim),
            nn.Dropout(dropout)
        )
        self.layer_scale_2 = nn.Parameter(layer_scale * torch.ones(dim), requires_grad=True)
        
    def forward(self, x: torch.Tensor):
        """
            x (torch.Tensor): [Batch, SequenceLen, Dim]
        """
        x_rev = torch.flip(x, dims=[1])
        
        residual = x
        x = self.norm1(x)
        
        fwd_features = self.bdh_fwd(x)
        bwd_features = torch.flip(self.bdh_bwd(x_rev), dims=[1])
        
        # Skip Connection 1
        x = residual + self.layer_scale_1 * (fwd_features + bwd_features)
        
        residual = x
        x = self.norm2(x)
        x = self.mlp(x)
        
        # Skip Connection 2
        x = residual + self.layer_scale_2 * x
        
        return x
    
    
class BDHVisionBackbone(nn.Module):
    """
    Encoder Only: Image -> Feature Sequence
    - Patch, Positional Embeddings thriugh Stack of BDHBlocks
    - Can attach any kind of head to this Vision Backbone
    """
    
    def __init__(self, img_dim = 256, patch_size = 4, dim = 128,
                 depth = 4, dropout=0.1, layer_scale=1e-4):
        super().__init__()
        
        self.patch_embed = nn.Conv2d(3, dim, kernel_size=patch_size, stride=patch_size)
        self.pos_embed = nn.Parameter(torch.zeros(1, (img_dim//patch_size)*(img_dim//patch_size), dim))
        
        self.blocks = nn.ModuleList([
            BDHBlock(dim, dropout=dropout, layer_scale=layer_scale) 
            for _ in range(depth)
        ])
        
        self.norm = nn.LayerNorm(dim)

    def forward(self, image: torch.Tensor):
        """
        image (torch.Tensor): [B, 3, H, W]
        Returns:
            x: [B, N, D] features
            (H_grid, W_grid): grid dimensions
        """
        # Patch Embeddings
        x = self.patch_embed(image)
        B, C, H_grid, W_grid = x.shape
        
        # [B, C, H, W] -> [B, N, C]
        x = x.flatten(2).transpose(1,2)
        
        # Positional Embeddings
        x = x + self.pos_embed[:, :x.size(1), :]
        
        for block in self.blocks:
            x = block(x)

        x = self.norm(x)
        
        return x, (H_grid, W_grid)
        
        