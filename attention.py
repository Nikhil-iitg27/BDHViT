import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional

class BDHLinearAttention(nn.Module):
    """
    BDH Linear Attention:
    - Linear attention with power-law hub sparsity
    - Optional persistent memory with gated read/write
    - Task-conditioned memory support
    - Designed for ViT-style token sequences [B, N, D]
    """
    
    def __init__(
        self, 
        dim: int, 
        expansion_factor: int = 4, 
        heads: int = 8,
        hub_pow: float = 1.5,
        persist: float = 0.95,
        use_mem: bool = True,
        conditional: bool = False
    ):
        super().__init__()
        self.dim = dim
        self.heads = heads
        self.use_mem = use_mem
        self.conditional = conditional

        # Sparse expanded dimension
        self.sparse_dim = dim * expansion_factor
        self.head_dim = self.sparse_dim // heads
        
        # Power-law exponent for hub sparsity
        self.hub_pow = hub_pow
        
        # Persistent memory buffer (non-trainable)
        self.persist = persist
        self.register_buffer("memory", torch.zeros(heads, self.head_dim, self.head_dim))
        
        # Q, K, V combined projection
        self.in_proj = nn.Linear(dim, 3 * self.sparse_dim)
        self.out_proj = nn.Linear(self.sparse_dim, dim)
        
        # Gates
        self.read_gate = nn.Linear(dim, heads)
        self.write_gate = nn.Linear(dim, heads)
        self.residual_gate = nn.Linear(dim, 1)
        
        # Optional task-conditioning
        if conditional:
            self.task_gate = nn.Embedding(32, heads)
        
        # Pre-normalization
        self.norm = nn.LayerNorm(dim)

    def forward(self, x: torch.Tensor, *, update_mem: bool = True, read_mem: bool = True, task_id: Optional[torch.Tensor] = None):
        """
        Args:
            x: [B, N, D] token embeddings
            update_mem: allow updating persistent memory
            read_mem: allow reading from persistent memory
            task_id: optional task id for conditional memory gating
        Returns:
            out: [B, N, D] updated token embeddings
        """
        B, N, D = x.shape
        H = self.heads

        # Pre-LayerNorm for stability
        residual = x
        x = self.norm(x)

        # Project to QKV
        qkv = self.in_proj(x)  # [B, N, 3 * sparse_dim]
        q, k, v = qkv.chunk(3, dim=-1)

        # Reshape to [B, H, N, head_dim]
        q = F.relu(q).view(B, N, H, self.head_dim).transpose(1, 2)
        k = F.relu(k).view(B, N, H, self.head_dim).transpose(1, 2)
        v = v.view(B, N, H, self.head_dim).transpose(1, 2)

        # Power-law hub sparsity
        q = torch.pow(q + 1e-6, self.hub_pow)
        k = torch.pow(k + 1e-6, self.hub_pow)

        # L1-normalization over sequence dimension (replace softmax)
        k = k / (k.sum(dim=-2, keepdim=True) + 1e-6)

        # Global context: K^T * V
        kT = k.transpose(-2, -1)  # [B, H, head_dim, N]
        batch_state = torch.matmul(kT, v)  # [B, H, head_dim, head_dim]

        # Gated activations
        x_mean = x.mean(dim=1)  # [B, D]
        read = torch.sigmoid(self.read_gate(x_mean)).view(B, H, 1, 1)
        write = torch.sigmoid(self.write_gate(x_mean)).view(B, H, 1, 1)
        residue = torch.sigmoid(self.residual_gate(x)).expand_as(residual)

        # Task-conditioned gates
        if self.conditional and task_id is not None:
            task = torch.sigmoid(self.task_gate(task_id)).view(B, H, 1, 1)
            read = read * task
            write = write * task

        # Update persistent memory
        if self.use_mem and update_mem:
            state_mean = batch_state.mean(dim=0)  # [H, head_dim, head_dim]
            self.memory.mul_(self.persist).add_(
                (write.mean(dim=0) * state_mean) * (1 - self.persist)
            )

        # Read memory
        if self.use_mem and read_mem:
            mem = self.memory.unsqueeze(0) * read  # [B, H, head_dim, head_dim]
            y = torch.matmul(q, mem)  # [B, H, N, head_dim]
        else:
            # fallback: use batch-only state
            y = torch.matmul(q, batch_state)

        # Merge heads
        y = y.transpose(1, 2).contiguous().view(B, N, self.sparse_dim)

        # Output projection + gated residual
        out = residue * self.out_proj(y) + (1 - residue) * residual

        return out