import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint

class CombinedRotaryEmbedding(nn.Module):
    def __init__(self, n_state, n_head, num_rotations, base=10000, checkpointing=False):
        super().__init__()
        self.n_state = n_state  # Total embedding size
        self.n_head = n_head    # Number of attention heads
        self.h_dim = n_state // n_head  # Dimension per head
        self.num_rotations = num_rotations  # Number of Givens rotations
        self.base = base
        self.checkpointing = checkpointing
        
        # Parameters for Givens rotations
        self.thetas = nn.Parameter(torch.zeros(num_rotations))
        
        # Rotation matrix for rotation
        self.rotation_matrix = nn.Parameter(torch.eye(self.h_dim))
        
        # Inverse frequency for rotary embeddings
        inv_freq = 1.0 / (self.base ** (torch.arange(0, self.h_dim, 2).float() / self.h_dim))
        self.register_buffer('inv_freq', inv_freq)
    
    def givens_rotation_matrix(self, n_state, i, j, theta):
        G = torch.eye(n_state, device=theta.device)
        G[i, i] = math.cos(theta)
        G[i, j] = -math.sin(theta)
        G[j, i] = math.sin(theta)
        G[j, j] = math.cos(theta)
        return G
    
    def update_base(self, new_base):
        self.base = new_base
        inv_freq = 1.0 / (self.base ** (torch.arange(0, self.h_dim, 2).float() / self.h_dim))
        self.register_buffer('inv_freq', inv_freq)
    
    def reset_parameters(self):
        nn.init.orthogonal_(self.rotation_matrix)
        nn.init.zeros_(self.thetas)
    
    def forward(self, x):
        if self.checkpointing:
            return checkpoint(self._forward, x)
        else:
            return self._forward(x)
    
    def _forward(self, x):
        # Check input dimensions and reshape
        if x.dim() not in [3, 4]:
            raise ValueError(f"Expected input tensor to be 3D or 4D, but got {x.dim()}D")
        
        if x.dim() == 3:
            batch_size, seq_len, n_state = x.size()
            x = x.view(batch_size, seq_len, self.n_head, self.h_dim)
        else:
            batch_size, seq_len, n_head, h_dim = x.size()
            if n_head != self.n_head or h_dim != self.h_dim:
                raise ValueError(f"Expected n_head {self.n_head} and h_dim {self.h_dim}, but got n_head {n_head} and h_dim {h_dim}")
        
        # Flatten for rotation
        x = x.view(-1, self.h_dim)
        
        # Apply Givens rotations
        for k in range(self.num_rotations):
            i, j = k % self.h_dim, (k + 1) % self.h_dim
            theta = self.thetas[k]
            G = self.givens_rotation_matrix(self.h_dim, i, j, theta)
            x = torch.matmul(x, G)
        
        # Apply rotation matrix
        x = torch.matmul(x, self.rotation_matrix)
        
        # Reshape back to original dimensions
        x = x.view(batch_size, seq_len, self.n_head, self.h_dim)
        
        # Rotary embeddings
        sinusoid_inp = torch.einsum('i, j -> i j', torch.arange(seq_len, device=x.device), self.inv_freq)
        sin = sinusoid_inp.sin()[None, :, None, :]
        cos = sinusoid_inp.cos()[None, :, None, :]
        
        x1, x2 = x[..., ::2], x[..., 1::2]
        x = torch.cat([x1 * cos - x2 * sin, x1 * sin + x2 * cos], dim=-1)
        
        # Reshape back to [batch_size, seq_len, n_state]
        x = x.view(batch_size, seq_len, self.n_state)
        
        return x
