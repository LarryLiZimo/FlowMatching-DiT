import torch
import torch.nn as nn
from torch.utils.checkpoint import checkpoint
import math
from config import Config

class SinusoidalEmbedding(nn.Module):
    def __init__(self, dim):
        super().__init__()
        half_dim = dim // 2
        freqs = torch.exp(-math.log(10000) * torch.arange(half_dim) / (half_dim - 1))
        self.register_buffer('freqs', freqs)
        
    def forward(self, t: torch.Tensor) -> torch.Tensor:
        emb = t[:, None] * self.freqs[None, :]
        return torch.cat([emb.sin(), emb.cos()], dim=-1)

class AdaLN(nn.Module):
    def __init__(self, dim, emb_dim):
        super().__init__()
        self.norm = nn.LayerNorm(dim, elementwise_affine=False)
        self.modulation = nn.Sequential(
            nn.SiLU(),
            nn.Linear(emb_dim, dim * 2)
        )

    def forward(self, x: torch.Tensor, t_emb: torch.Tensor) -> torch.Tensor:
        scale, shift = self.modulation(t_emb).chunk(2, dim=-1)
        return self.norm(x) * (1 + scale.unsqueeze(1)) + shift.unsqueeze(1)

class DiTBlock(nn.Module):
    def __init__(self, dim, emb_dim, num_heads, dropout):
        super().__init__()
        self.norm1 = AdaLN(dim, emb_dim)
        self.attn = nn.MultiheadAttention(dim, num_heads, dropout=dropout, batch_first=True)
        self.attn_drop = nn.Dropout(dropout)
        self.norm2 = AdaLN(dim, emb_dim)
        self.mlp = nn.Sequential(
            nn.Linear(dim, dim * 4),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(dim * 4, dim),
            nn.Dropout(dropout),
        )

    def forward(self, x: torch.Tensor, t_emb: torch.Tensor) -> torch.Tensor:
        normed = self.norm1(x, t_emb)
        x = x + self.attn_drop(self.attn(normed, normed, normed)[0])
        x = x + self.mlp(self.norm2(x, t_emb))
        return x

class DiT(nn.Module):
    def __init__(self, cfg: Config):
        super().__init__()
        self.cfg = cfg
        self.t_embed = nn.Sequential(
            SinusoidalEmbedding(cfg.emb_hidden_dim),
            nn.Linear(cfg.emb_hidden_dim, cfg.emb_hidden_dim * 4),
            nn.SiLU(),
            nn.Linear(cfg.emb_hidden_dim * 4, cfg.emb_hidden_dim),
        )
        assert cfg.img_size % cfg.patch_size == 0, f'{cfg.img_size} / {cfg.patch_size}'
        num_patches = (cfg.img_size // cfg.patch_size) ** 2
        self.patch_embed = nn.Conv2d(cfg.in_channels, cfg.hidden_dim, cfg.patch_size, cfg.patch_size)
        self.pos_embed = nn.Parameter(torch.randn(1, num_patches, cfg.hidden_dim))
        self.blocks = nn.ModuleList([
            DiTBlock(cfg.hidden_dim, cfg.emb_hidden_dim, cfg.num_heads, cfg.dropout) for _ in range(cfg.num_layers)
        ])
        self.final_norm = AdaLN(cfg.hidden_dim, cfg.emb_hidden_dim)
        self.output_proj = nn.Linear(cfg.hidden_dim, cfg.in_channels * cfg.patch_size ** 2)

    def forward(self, x: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        B, C, H, W = x.shape
        t_emb = self.t_embed(t) # B, D
        x_emb = self.patch_embed(x) # B, D, sqrt(S), sqrt(S)
        x_emb = x_emb.flatten(2).transpose(1, 2) + self.pos_embed # B, S, D
        for block in self.blocks:
            x_emb = block(x_emb, t_emb)
            # x_emb = checkpoint(block, x_emb, t_emb, use_reentrant=False)
        H_p = W_p = self.cfg.img_size // self.cfg.patch_size
        x_emb = self.final_norm(x_emb, t_emb)
        v = self.output_proj(x_emb)                                     # B, S, C*P*P
        v = v.reshape(B, H_p, W_p, C, self.cfg.patch_size, self.cfg.patch_size)
        v = v.permute(0, 3, 1, 4, 2, 5).reshape(B, C, H, W)
        return v
