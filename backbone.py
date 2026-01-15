from .attention import BDHLinearAttention

import torch
import torch.nn as nn
import torch.nn.functional as F

class BDHBlock(nn.Module):
    """
    A Single Layer: Norm -> Attention -> Norm -> MLP
    """
    def __init__(self, dim: int, expansion_factor=4):
        super().__init__()
        self.norm1 = nn.LayerNorm(dim)
        
        # Two BDH Attention Layers for Bi-Directional Context
        self.bdh_fwd = BDHLinearAttention(dim, expansion_factor)
        self.bdh_bwd = BDHLinearAttention(dim, expansion_factor)
        
        self.norm2 = nn.LayerNorm(dim)
        
        self.mlp = nn.Sequential(
            nn.Linear(dim, dim * 4),
            nn.GELU(),
            nn.Linear(dim * 4, dim)
        )
        
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
        x = residual + (fwd_features + bwd_features)
        
        residual = x
        x = self.norm2(x)
        x = self.mlp(x)
        
        # Skip Connection 2
        x = residual + x
        
        return x
    
    
class BDHVisionBackbone(nn.Module):
    """
    Encoder Only: Image -> Feature Sequence
    - Can attach any kind of head to this Vision Backbone
    """
    
    def __init__(self, img_dim = 256, patch_size = 4, dim = 128, depth = 4):
        super().__init__()
        
        self.patch_embed = nn.Conv2d(3, dim, kernel_size=patch_size, stride=patch_size)
        
        self.blocks = nn.ModuleList([
            BDHBlock(dim) for _ in range
        ])
        
        self.norm = nn.LayerNorm(dim)

    def forward(self, image: torch.Tensor):
        x = self.patch_embed(image)
        B, C, H_grid, W_grid = x.shape
        
        # [B, C, H, W] -> [B, N, C]
        x = x.flatten(2).transpose(1,2)
        
        for block in self.blocks:
            x = block(x)
        
        x = self.norm(x)
        
        return x, (H_grid, W_grid)
        
        