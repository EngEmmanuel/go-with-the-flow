# This file should contains classes that are building blocks for the transformer model

import torch
import torch.nn.functional as F
from torch import nn, Tensor, tensor, is_tensor, cat, stack
from torch.nn import Module, ModuleList

import einx
from einops import rearrange, repeat, pack, unpack
from einops.layers.torch import Rearrange


from x_transformers import (
    Encoder,
    TransformerWrapper
)

from denoising_diffusion_pytorch import (
    GaussianDiffusion1D
)

def divisible_by(num, den):
    return (num % den) == 0


# random sinusoidal for times - used by deepmind a lot

class RandomSinusoidalPosEmb(Module):
    def __init__(self, dim):
        super().__init__()
        assert divisible_by(dim, 2)
        half_dim = dim // 2
        self.weights = nn.Parameter(torch.randn(half_dim), requires_grad = False)

    def forward(self, x):
        freqs = x * rearrange(self.weights, 'd -> 1 d') * 2 * torch.pi
        fouriered = torch.cat((freqs.sin(), freqs.cos()), dim = -1)
        return fouriered

# DiT wrapper

class DiffusionTransformerWrapper(Module):
    def __init__(
        self,
        dim_input,
        dim_time,
        transformer: Encoder
    ):
        super().__init__()

        self.transformer = transformer

        dim = transformer.dim

        self.proj_in = nn.Linear(dim_input, dim)

        self.to_time_cond = nn.Sequential(
            RandomSinusoidalPosEmb(dim),
            nn.Linear(dim, dim_time),
            nn.SiLU(),
        )

        self.proj_out = nn.Linear(dim, dim_input)