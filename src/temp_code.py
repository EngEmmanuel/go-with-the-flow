import torch
from torch import nn
from einops import rearrange, repeat

# ------------------------
# Existing building blocks
# ------------------------
class FeedForward(nn.Module):
    def __init__(self, dim, hidden_dim, dropout=0.):
        super().__init__()
        self.net = nn.Sequential(
            nn.LayerNorm(dim),
            nn.Linear(dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, dim),
            nn.Dropout(dropout)
        )
    def forward(self, x):
        return self.net(x)

class Attention(nn.Module):
    def __init__(self, dim, heads=8, dim_head=64, dropout=0.):
        super().__init__()
        inner_dim = dim_head * heads
        project_out = not (heads == 1 and dim_head == dim)

        self.heads = heads
        self.scale = dim_head ** -0.5

        self.norm = nn.LayerNorm(dim)
        self.attend = nn.Softmax(dim=-1)
        self.dropout = nn.Dropout(dropout)

        self.to_qkv = nn.Linear(dim, inner_dim * 3, bias=False)

        self.to_out = nn.Sequential(
            nn.Linear(inner_dim, dim),
            nn.Dropout(dropout)
        ) if project_out else nn.Identity()

    def forward(self, x):
        x = self.norm(x)
        qkv = self.to_qkv(x).chunk(3, dim=-1)
        q, k, v = map(lambda t: rearrange(t, 'b n (h d) -> b h n d', h=self.heads), qkv)

        dots = torch.matmul(q, k.transpose(-1, -2)) * self.scale
        attn = self.attend(dots)
        attn = self.dropout(attn)

        out = torch.matmul(attn, v)
        out = rearrange(out, 'b h n d -> b n (h d)')
        return self.to_out(out)

class Transformer(nn.Module):
    def __init__(self, dim, depth, heads, dim_head, mlp_dim, dropout=0.):
        super().__init__()
        self.layers = nn.ModuleList([])
        for _ in range(depth):
            self.layers.append(nn.ModuleList([
                Attention(dim, heads=heads, dim_head=dim_head, dropout=dropout),
                FeedForward(dim, mlp_dim, dropout=dropout)
            ]))
    def forward(self, x):
        for attn, ff in self.layers:
            x = attn(x) + x
            x = ff(x) + x
        return x

# ------------------------
# (2+1)D ViT for generation
# ------------------------
class ViT_2plus1D(nn.Module):
    def __init__(self, *, image_size, image_patch_size, frames, frame_patch_size,
                 dim, depth, heads, mlp_dim, channels=3, dim_head=64,
                 dropout=0., emb_dropout=0.):
        """
        This model outputs feature embeddings for each spatiotemporal token.
        No classification head — ready for use in diffusion / flow-matching.
        """
        super().__init__()
        image_height, image_width = image_size, image_size
        patch_height, patch_width = image_patch_size, image_patch_size

        assert image_height % patch_height == 0 and image_width % patch_width == 0
        assert frames % frame_patch_size == 0

        self.num_spatial_patches = (image_height // patch_height) * (image_width // patch_width)
        self.num_frame_patches = frames // frame_patch_size
        patch_dim = channels * patch_height * patch_width * frame_patch_size

        # Patch embedding
        self.to_patch_embedding = nn.Sequential(
            rearrange('b c (f pf) (h p1) (w p2) -> b f (h w) (p1 p2 pf c)',
                      p1=patch_height, p2=patch_width, pf=frame_patch_size),
            nn.LayerNorm(patch_dim),
            nn.Linear(patch_dim, dim),
            nn.LayerNorm(dim),
        )

        # Positional embeddings
        self.pos_emb_spatial = nn.Parameter(torch.randn(1, self.num_spatial_patches, dim))
        self.pos_emb_temporal = nn.Parameter(torch.randn(1, self.num_frame_patches, dim))

        self.dropout = nn.Dropout(emb_dropout)

        # Factorised transformers
        self.spatial_transformer = Transformer(dim, depth, heads, dim_head, mlp_dim, dropout)
        self.temporal_transformer = Transformer(dim, depth, heads, dim_head, mlp_dim, dropout)

    def forward(self, video):
        """
        video: shape (b, c, f, h, w)
        returns: tensor of shape (b, f, patches_per_frame, dim)
        """
        b, c, f, h, w = video.shape

        # Patch embedding: (b, frames, patches_per_frame, dim)
        x = self.to_patch_embedding(video)

        # Spatial attention (per frame)
        x = x + self.pos_emb_spatial
        x = rearrange(x, 'b f p d -> (b f) p d')
        x = self.spatial_transformer(x)
        x = rearrange(x, '(b f) p d -> b f p d', b=b)

        # Temporal attention (per patch position)
        x = rearrange(x, 'b f p d -> (b p) f d')
        x = x + self.pos_emb_temporal
        x = self.temporal_transformer(x)
        x = rearrange(x, '(b p) f d -> b f p d', b=b)

        return x  # This can go straight into your flow-matching/diffusion head
