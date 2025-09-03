
# This file contains modified code from the HuggingFace Diffusers library.

import math
from dataclasses import dataclass
from typing import Any, Dict, Optional, Tuple, Union

import numpy as np
import torch
import torch._dynamo
import torch.nn as nn
import torch.nn.functional as F
import xformers
from diffusers.configuration_utils import ConfigMixin, register_to_config
from diffusers.loaders import UNet2DConditionLoadersMixin
from diffusers.models.attention import BasicTransformerBlock
from diffusers.models.attention_processor import (
    CROSS_ATTENTION_PROCESSORS,
    AttentionProcessor,
    AttnProcessor,
)
from diffusers.models.embeddings import PatchEmbed, TimestepEmbedding, Timesteps
from diffusers.models.modeling_outputs import Transformer2DModelOutput
from diffusers.models.modeling_utils import ModelMixin
from diffusers.models.unets.unet_3d_blocks import UNetMidBlockSpatioTemporal
from diffusers.models.unets.unet_3d_blocks import get_down_block as get_down_block_3d
from diffusers.models.unets.unet_3d_blocks import get_up_block as get_up_block_3d
from diffusers.utils import BaseOutput, is_torch_version
from einops import rearrange
from timm.layers.drop import DropPath
from timm.layers.mlp import Mlp
from torchvision.models import resnet18

approx_gelu = lambda: nn.GELU(approximate="tanh")


class SegDiTTransformer2DModel(ModelMixin, ConfigMixin):
    r"""
    A 2D Transformer model as introduced in DiT (https://arxiv.org/abs/2212.09748).
    Parameters:
        num_attention_heads (int, optional, defaults to 16): The number of heads to use for multi-head attention.
        attention_head_dim (int, optional, defaults to 72): The number of channels in each head.
        in_channels (int, defaults to 4): The number of channels in the input.
        out_channels (int, optional):
            The number of channels in the output. Specify this parameter if the output channel number differs from the
            input.
        num_layers (int, optional, defaults to 28): The number of layers of Transformer blocks to use.
        dropout (float, optional, defaults to 0.0): The dropout probability to use within the Transformer blocks.
        norm_num_groups (int, optional, defaults to 32):
            Number of groups for group normalization within Transformer blocks.
        attention_bias (bool, optional, defaults to True):
            Configure if the Transformer blocks' attention should contain a bias parameter.
        sample_size (int, defaults to 32):
            The width of the latent images. This parameter is fixed during training.
        patch_size (int, defaults to 2):
            Size of the patches the model processes, relevant for architectures working on non-sequential data.
        activation_fn (str, optional, defaults to "gelu-approximate"):
            Activation function to use in feed-forward networks within Transformer blocks.
        num_embeds_ada_norm (int, optional, defaults to 1000):
            Number of embeddings for AdaLayerNorm, fixed during training and affects the maximum denoising steps during
            inference.
        upcast_attention (bool, optional, defaults to False):
            If true, upcasts the attention mechanism dimensions for potentially improved performance.
        norm_type (str, optional, defaults to "ada_norm_zero"):
            Specifies the type of normalization used, can be 'ada_norm_zero'.
        norm_elementwise_affine (bool, optional, defaults to False):
            If true, enables element-wise affine parameters in the normalization layers.
        norm_eps (float, optional, defaults to 1e-5):
            A small constant added to the denominator in normalization layers to prevent division by zero.
    """

    _supports_gradient_checkpointing = True

    @register_to_config
    def __init__(
        self,
        num_attention_heads: int = 16,
        attention_head_dim: int = 72,
        in_channels: int = 4,
        out_channels: Optional[int] = None,
        num_layers: int = 28,
        dropout: float = 0.0,
        norm_num_groups: int = 32,
        attention_bias: bool = True,
        sample_size: int = 32,
        patch_size: int = 2,
        activation_fn: str = "gelu-approximate",
        num_embeds_ada_norm: Optional[int] = 1000,
        upcast_attention: bool = False,
        norm_type: str = "ada_norm_zero",
        norm_elementwise_affine: bool = False,
        norm_eps: float = 1e-5,
    ):
        super().__init__()

        # Validate inputs.
        if norm_type != "ada_norm_zero":
            raise NotImplementedError(
                f"Forward pass is not implemented when `patch_size` is not None and `norm_type` is '{norm_type}'."
            )
        elif norm_type == "ada_norm_zero" and num_embeds_ada_norm is None:
            raise ValueError(
                f"When using a `patch_size` and this `norm_type` ({norm_type}), `num_embeds_ada_norm` cannot be None."
            )

        # Set some common variables used across the board.
        self.attention_head_dim = attention_head_dim
        self.inner_dim = (
            self.config.num_attention_heads * self.config.attention_head_dim
        )
        self.out_channels = in_channels if out_channels is None else out_channels
        self.gradient_checkpointing = False

        # 2. Initialize the position embedding and transformer blocks.
        self.height = self.config.sample_size
        self.width = self.config.sample_size

        self.patch_size = self.config.patch_size
        self.pos_embed = PatchEmbed(
            height=self.config.sample_size,
            width=self.config.sample_size,
            patch_size=self.config.patch_size,
            in_channels=self.config.in_channels,
            embed_dim=self.inner_dim,
        )

        self.transformer_blocks = nn.ModuleList(
            [
                BasicTransformerBlock(
                    self.inner_dim,
                    self.config.num_attention_heads,
                    self.config.attention_head_dim,
                    dropout=self.config.dropout,
                    activation_fn=self.config.activation_fn,
                    num_embeds_ada_norm=self.config.num_embeds_ada_norm,
                    attention_bias=self.config.attention_bias,
                    upcast_attention=self.config.upcast_attention,
                    norm_type=norm_type,
                    norm_elementwise_affine=self.config.norm_elementwise_affine,
                    norm_eps=self.config.norm_eps,
                )
                for _ in range(self.config.num_layers)
            ]
        )

        # 3. Output blocks.
        self.norm_out = nn.LayerNorm(self.inner_dim, elementwise_affine=False, eps=1e-6)
        self.proj_out_1 = nn.Linear(self.inner_dim, 2 * self.inner_dim)
        self.proj_out_2 = nn.Linear(
            self.inner_dim,
            self.config.patch_size * self.config.patch_size * self.out_channels,
        )

    def _set_gradient_checkpointing(self, module, value=False):
        if hasattr(module, "gradient_checkpointing"):
            module.gradient_checkpointing = value

    def forward(
        self,
        hidden_states: torch.Tensor,
        timestep: Optional[torch.LongTensor] = None,
        class_labels: Optional[torch.LongTensor] = None,
        cross_attention_kwargs: Dict[str, Any] = None,
        segmentation: Optional[torch.LongTensor] = None,
        return_dict: bool = True,
    ):
        """
        The [`DiTTransformer2DModel`] forward method.
        Args:
            hidden_states (`torch.LongTensor` of shape `(batch size, num latent pixels)` if discrete, `torch.FloatTensor` of shape `(batch size, channel, height, width)` if continuous):
                Input `hidden_states`.
            timestep ( `torch.LongTensor`, *optional*):
                Used to indicate denoising step. Optional timestep to be applied as an embedding in `AdaLayerNorm`.
            class_labels ( `torch.LongTensor` of shape `(batch size, num classes)`, *optional*):
                Used to indicate class labels conditioning. Optional class labels to be applied as an embedding in
                `AdaLayerZeroNorm`.
            cross_attention_kwargs ( `Dict[str, Any]`, *optional*):
                A kwargs dictionary that if specified is passed along to the `AttentionProcessor` as defined under
                `self.processor` in
                [diffusers.models.attention_processor](https://github.com/huggingface/diffusers/blob/main/src/diffusers/models/attention_processor.py).
            return_dict (`bool`, *optional*, defaults to `True`):
                Whether or not to return a [`~models.unets.unet_2d_condition.UNet2DConditionOutput`] instead of a plain
                tuple.
        Returns:
            If `return_dict` is True, an [`~models.transformer_2d.Transformer2DModelOutput`] is returned, otherwise a
            `tuple` where the first element is the sample tensor.
        """

        # 0. If segmentation is provided, apply it to the input.
        if segmentation is not None:
            hidden_states = torch.cat([hidden_states, segmentation], dim=1)  # B C+1 H W

        # 1. Input
        height, width = (
            hidden_states.shape[-2] // self.patch_size,
            hidden_states.shape[-1] // self.patch_size,
        )
        hidden_states = self.pos_embed(hidden_states)

        # 2. Blocks
        for block in self.transformer_blocks:
            if torch.is_grad_enabled() and self.gradient_checkpointing:

                def create_custom_forward(module, return_dict=None):
                    def custom_forward(*inputs):
                        if return_dict is not None:
                            return module(*inputs, return_dict=return_dict)
                        else:
                            return module(*inputs)

                    return custom_forward

                ckpt_kwargs: Dict[str, Any] = (
                    {"use_reentrant": False} if is_torch_version(">=", "1.11.0") else {}
                )
                hidden_states = torch.utils.checkpoint.checkpoint(
                    create_custom_forward(block),
                    hidden_states,
                    None,
                    None,
                    None,
                    timestep,
                    cross_attention_kwargs,
                    class_labels,
                    **ckpt_kwargs,
                )
            else:
                hidden_states = block(
                    hidden_states,
                    attention_mask=None,
                    encoder_hidden_states=None,
                    encoder_attention_mask=None,
                    timestep=timestep,
                    cross_attention_kwargs=cross_attention_kwargs,
                    class_labels=class_labels,
                )

        # 3. Output
        conditioning = self.transformer_blocks[0].norm1.emb(
            timestep, class_labels, hidden_dtype=hidden_states.dtype
        )
        shift, scale = self.proj_out_1(F.silu(conditioning)).chunk(2, dim=1)
        hidden_states = (
            self.norm_out(hidden_states) * (1 + scale[:, None]) + shift[:, None]
        )
        hidden_states = self.proj_out_2(hidden_states)

        # unpatchify
        height = width = int(hidden_states.shape[1] ** 0.5)
        hidden_states = hidden_states.reshape(
            shape=(
                -1,
                height,
                width,
                self.patch_size,
                self.patch_size,
                self.out_channels,
            )
        )
        hidden_states = torch.einsum("nhwpqc->nchpwq", hidden_states)
        output = hidden_states.reshape(
            shape=(
                -1,
                self.out_channels,
                height * self.patch_size,
                width * self.patch_size,
            )
        )

        if not return_dict:
            return (output,)

        return Transformer2DModelOutput(sample=output)


def get_2d_sincos_pos_embed(
    embed_dim, grid_size, cls_token=False, extra_tokens=0, scale=1.0, base_size=None
):
    """
    grid_size: int of the grid height and width
    return:
    pos_embed: [grid_size*grid_size, embed_dim] or [1+grid_size*grid_size, embed_dim] (w/ or w/o cls_token)
    """
    if not isinstance(grid_size, tuple):
        grid_size = (grid_size, grid_size)

    grid_h = np.arange(grid_size[0], dtype=np.float32) / scale
    grid_w = np.arange(grid_size[1], dtype=np.float32) / scale
    if base_size is not None:
        grid_h *= base_size / grid_size[0]
        grid_w *= base_size / grid_size[1]
    grid = np.meshgrid(grid_w, grid_h)  # here w goes first
    grid = np.stack(grid, axis=0)

    grid = grid.reshape([2, 1, grid_size[1], grid_size[0]])
    pos_embed = get_2d_sincos_pos_embed_from_grid(embed_dim, grid)
    if cls_token and extra_tokens > 0:
        pos_embed = np.concatenate(
            [np.zeros([extra_tokens, embed_dim]), pos_embed], axis=0
        )
    return pos_embed


def get_2d_sincos_pos_embed_from_grid(embed_dim, grid):
    assert embed_dim % 2 == 0

    # use half of dimensions to encode grid_h
    emb_h = get_1d_sincos_pos_embed_from_grid(embed_dim // 2, grid[0])  # (H*W, D/2)
    emb_w = get_1d_sincos_pos_embed_from_grid(embed_dim // 2, grid[1])  # (H*W, D/2)

    emb = np.concatenate([emb_h, emb_w], axis=1)  # (H*W, D)
    return emb


def get_1d_sincos_pos_embed(embed_dim, length, scale=1.0):
    pos = np.arange(0, length)[..., None] / scale
    return get_1d_sincos_pos_embed_from_grid(embed_dim, pos)


def get_1d_sincos_pos_embed_from_grid(embed_dim, pos):
    """
    embed_dim: output dimension for each position
    pos: a list of positions to be encoded: size (M,)
    out: (M, D)
    """
    assert embed_dim % 2 == 0
    omega = np.arange(embed_dim // 2, dtype=np.float64)
    omega /= embed_dim / 2.0
    omega = 1.0 / 10000**omega  # (D/2,)

    pos = pos.reshape(-1)  # (M,)
    out = np.einsum("m,d->md", pos, omega)  # (M, D/2), outer product

    emb_sin = np.sin(out)  # (M, D/2)
    emb_cos = np.cos(out)  # (M, D/2)

    emb = np.concatenate([emb_sin, emb_cos], axis=1)  # (M, D)
    return emb


def t2i_modulate(x, shift, scale):
    return x * (1 + scale) + shift


class PatchEmbed3D(nn.Module):
    """Video to Patch Embedding.
    Args:
        patch_size (int): Patch token size. Default: (2,4,4).
        in_chans (int): Number of input video channels. Default: 3.
        embed_dim (int): Number of linear projection output channels. Default: 96.
        norm_layer (nn.Module, optional): Normalization layer. Default: None
    """

    def __init__(
        self,
        patch_size=(2, 4, 4),
        in_chans=3,
        embed_dim=96,
        norm_layer=None,
        flatten=True,
    ):
        super().__init__()
        self.patch_size = patch_size
        self.flatten = flatten

        self.in_chans = in_chans
        self.embed_dim = embed_dim

        self.proj = nn.Conv3d(
            in_chans, embed_dim, kernel_size=patch_size, stride=patch_size
        )
        if norm_layer is not None:
            self.norm = norm_layer(embed_dim)
        else:
            self.norm = None

    def forward(self, x):
        """Forward function."""
        # padding
        _, _, D, H, W = x.size()
        if W % self.patch_size[2] != 0:
            x = F.pad(x, (0, self.patch_size[2] - W % self.patch_size[2]))
        if H % self.patch_size[1] != 0:
            x = F.pad(x, (0, 0, 0, self.patch_size[1] - H % self.patch_size[1]))
        if D % self.patch_size[0] != 0:
            x = F.pad(x, (0, 0, 0, 0, 0, self.patch_size[0] - D % self.patch_size[0]))

        x = self.proj(x)  # (B C T H W)
        if self.norm is not None:
            D, Wh, Ww = x.size(2), x.size(3), x.size(4)
            x = x.flatten(2).transpose(1, 2)
            x = self.norm(x)
            x = x.transpose(1, 2).view(-1, self.embed_dim, D, Wh, Ww)
        if self.flatten:
            x = x.flatten(2).transpose(1, 2)  # BCTHW -> BNC
        return x


class Attention(nn.Module):
    def __init__(
        self,
        dim: int,
        num_heads: int = 8,
        qkv_bias: bool = False,
        qk_norm: bool = False,
        attn_drop: float = 0.0,
        proj_drop: float = 0.0,
        norm_layer: nn.Module = nn.LayerNorm,
        enable_flashattn: bool = False,
    ) -> None:
        super().__init__()
        assert dim % num_heads == 0, "dim should be divisible by num_heads"
        self.dim = dim
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.scale = self.head_dim**-0.5

        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.q_norm = norm_layer(self.head_dim) if qk_norm else nn.Identity()
        self.k_norm = norm_layer(self.head_dim) if qk_norm else nn.Identity()
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

        if enable_flashattn:
            print(
                "[WARNING] FlashAttention cannot be used. Set enable_flashattn to False."
            )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, N, C = x.shape
        qkv = self.qkv(x)
        qkv_shape = (B, N, 3, self.num_heads, self.head_dim)
        qkv_permute_shape = (2, 0, 3, 1, 4)
        qkv = qkv.view(qkv_shape).permute(qkv_permute_shape)
        q, k, v = qkv.unbind(0)
        q, k = self.q_norm(q), self.k_norm(k)

        dtype = q.dtype
        q = q * self.scale
        attn = q @ k.transpose(-2, -1)  # translate attn to float32
        attn = attn.to(torch.float32)
        attn = attn.softmax(dim=-1)
        attn = attn.to(dtype)  # cast back attn to original dtype
        attn = self.attn_drop(attn)
        x = attn @ v

        x_output_shape = (B, N, C)
        x = x.reshape(x_output_shape)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x


class MultiHeadCrossAttention(nn.Module):
    def __init__(self, d_model, num_heads, attn_drop=0.0, proj_drop=0.0):
        super(MultiHeadCrossAttention, self).__init__()
        assert d_model % num_heads == 0, "d_model must be divisible by num_heads"

        self.d_model = d_model
        self.num_heads = num_heads
        self.head_dim = d_model // num_heads

        self.q_linear = nn.Linear(d_model, d_model)
        self.kv_linear = nn.Linear(d_model, d_model * 2)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(d_model, d_model)
        self.proj_drop = nn.Dropout(proj_drop)

    @torch._dynamo.disable
    def forward(self, x, cond, mask=None):
        # query/value: img tokens; key: condition; mask: if padding tokens
        B, N, C = x.shape

        q = self.q_linear(x).view(1, -1, self.num_heads, self.head_dim)
        kv = self.kv_linear(cond).view(1, -1, 2, self.num_heads, self.head_dim)
        k, v = kv.unbind(2)

        attn_bias = None
        if mask is not None:
            attn_bias = xformers.ops.fmha.BlockDiagonalMask.from_seqlens([N] * B, mask)
        x = xformers.ops.memory_efficient_attention(
            q, k, v, p=self.attn_drop.p, attn_bias=attn_bias
        )

        x = x.view(B, -1, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x


class TimestepEmbedder(nn.Module):
    """
    Embeds scalar timesteps into vector representations.
    """

    def __init__(self, hidden_size, frequency_embedding_size=256):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(frequency_embedding_size, hidden_size, bias=True),
            nn.SiLU(),
            nn.Linear(hidden_size, hidden_size, bias=True),
        )
        self.frequency_embedding_size = frequency_embedding_size

    @staticmethod
    def timestep_embedding(t, dim, max_period=10000):
        """
        Create sinusoidal timestep embeddings.
        :param t: a 1-D Tensor of N indices, one per batch element.
                          These may be fractional.
        :param dim: the dimension of the output.
        :param max_period: controls the minimum frequency of the embeddings.
        :return: an (N, D) Tensor of positional embeddings.
        """
        # https://github.com/openai/glide-text2im/blob/main/glide_text2im/nn.py
        half = dim // 2
        freqs = torch.exp(
            -math.log(max_period)
            * torch.arange(start=0, end=half, dtype=torch.float32)
            / half
        )
        freqs = freqs.to(device=t.device)
        args = t[:, None].float() * freqs[None]
        embedding = torch.cat([torch.cos(args), torch.sin(args)], dim=-1)
        if dim % 2:
            embedding = torch.cat(
                [embedding, torch.zeros_like(embedding[:, :1])], dim=-1
            )
        return embedding

    def forward(self, t, dtype):
        t_freq = self.timestep_embedding(t, self.frequency_embedding_size)
        if t_freq.dtype != dtype:
            t_freq = t_freq.to(dtype)
        t_emb = self.mlp(t_freq)
        return t_emb


class CaptionEmbedder(nn.Module):
    """
    Embeds class labels into vector representations. Also handles label dropout for classifier-free guidance.
    """

    def __init__(
        self,
        in_channels,
        hidden_size,
        uncond_prob,
        act_layer=nn.GELU(approximate="tanh"),
        token_num=120,
    ):
        super().__init__()
        self.y_proj = Mlp(
            in_features=in_channels,
            hidden_features=hidden_size,
            out_features=hidden_size,
            act_layer=act_layer,
            drop=0,
        )
        self.register_buffer(
            "y_embedding",
            nn.Parameter(torch.randn(token_num, in_channels) / in_channels**0.5),
        )
        self.uncond_prob = uncond_prob

    def token_drop(self, caption, force_drop_ids=None):
        """
        Drops labels to enable classifier-free guidance.
        """
        if force_drop_ids is None:
            drop_ids = torch.rand(caption.shape[0]).cuda() < self.uncond_prob
        else:
            drop_ids = force_drop_ids == 1
        caption = torch.where(drop_ids[:, None, None, None], self.y_embedding, caption)
        return caption

    @torch._dynamo.disable
    def forward(self, caption, train, force_drop_ids=None):
        if train:
            assert caption.shape[2:] == self.y_embedding.shape
        use_dropout = self.uncond_prob > 0
        if (train and use_dropout) or (force_drop_ids is not None):
            caption = self.token_drop(caption, force_drop_ids)
        caption = self.y_proj(caption)
        return caption


class T2IFinalLayer(nn.Module):
    """
    The final layer of PixArt.
    """

    def __init__(self, hidden_size, num_patch, out_channels):
        super().__init__()
        self.norm_final = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
        self.linear = nn.Linear(hidden_size, num_patch * out_channels, bias=True)
        self.scale_shift_table = nn.Parameter(
            torch.randn(2, hidden_size) / hidden_size**0.5
        )
        self.out_channels = out_channels

    def forward(self, x, t):
        shift, scale = (self.scale_shift_table[None] + t[:, None]).chunk(2, dim=1)
        x = t2i_modulate(self.norm_final(x), shift, scale)
        x = self.linear(x)
        return x


class STDiTBlock(nn.Module):
    """
    STDiT: Spatio-Temporal Diffusion Transformer.
    Args:
        hidden_size (int): Hidden size of the model.
        num_heads (int): Number of attention heads.
        d_s (int): Spatial patch size.
        d_t (int): Temporal patch size.
        mlp_ratio (float): Ratio of hidden to mlp hidden size.
        drop_path (float): Drop path rate.
        enable_flashattn (bool): Enable FlashAttention.
    """

    def __init__(
        self,
        hidden_size,
        num_heads,
        d_s=None,
        d_t=None,
        mlp_ratio=4.0,
        drop_path=0.0,
        enable_flashattn=False,
        uncond=False,
    ):
        super().__init__()
        self.hidden_size = hidden_size
        self.enable_flashattn = enable_flashattn

        self.attn_cls = Attention
        self.mha_cls = MultiHeadCrossAttention

        self.norm1 = nn.LayerNorm(hidden_size, eps=1e-6, elementwise_affine=False)
        self.attn = self.attn_cls(
            hidden_size,
            num_heads=num_heads,
            qkv_bias=True,
            enable_flashattn=False,
        )
        if uncond:
            self.cross_attn = self.mha_cls(hidden_size, num_heads)
        self.norm2 = nn.LayerNorm(hidden_size, eps=1e-6, elementwise_affine=False)
        self.mlp = Mlp(
            in_features=hidden_size,
            hidden_features=int(hidden_size * mlp_ratio),
            act_layer=approx_gelu,
            drop=0,
        )
        self.drop_path = DropPath(drop_path) if drop_path > 0.0 else nn.Identity()
        self.scale_shift_table = nn.Parameter(
            torch.randn(6, hidden_size) / hidden_size**0.5
        )

        # temporal attention
        self.d_s = d_s
        self.d_t = d_t

        self.attn_temp = self.attn_cls(
            hidden_size,
            num_heads=num_heads,
            qkv_bias=True,
            enable_flashattn=self.enable_flashattn,
        )

    def forward(self, x, t, y=None, mask=None, tpe=None):
        """
        Args:
            x (torch.Tensor): noisy input tensor of shape [B, N, C]
            y (torch.Tensor): conditional input tensor of shape [B, N, C]
            t (torch.Tensor): input tensor; of shape [B, C]
            mask (torch.Tensor): input tensor; of shape [B, N]
            tpe (torch.Tensor): input tensor; of shape [B, C]
        """
        B, N, C = x.shape

        shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp = (
            self.scale_shift_table[None] + t.reshape(B, 6, -1)
        ).chunk(6, dim=1)
        x_m = t2i_modulate(self.norm1(x), shift_msa, scale_msa)

        # spatial branch
        x_s = rearrange(x_m, "B (T S) C -> (B T) S C", T=self.d_t, S=self.d_s)
        x_s = self.attn(x_s)
        x_s = rearrange(x_s, "(B T) S C -> B (T S) C", T=self.d_t, S=self.d_s)
        x = x + self.drop_path(gate_msa * x_s)

        # temporal branch
        x_t = rearrange(x, "B (T S) C -> (B S) T C", T=self.d_t, S=self.d_s)
        if tpe is not None:
            x_t = x_t + tpe
        x_t = self.attn_temp(x_t)
        x_t = rearrange(x_t, "(B S) T C -> B (T S) C", T=self.d_t, S=self.d_s)
        x = x + self.drop_path(gate_msa * x_t)

        # cross attn
        if y is not None:
            x = x + self.cross_attn(x, y, mask)

        # mlp
        x = x + self.drop_path(
            gate_mlp * self.mlp(t2i_modulate(self.norm2(x), shift_mlp, scale_mlp))
        )

        return x


# | Model | Layers N | Hidden size d | Heads | Gflops (I=32, p=4) |
# |-------|----------|---------------|-------|---------------------|
# | DiT-S | 12       | 384           | 6     | 1.4                 |
# | DiT-B | 12       | 768           | 12    | 5.6                 |
# | DiT-L | 24       | 1024          | 16    | 19.7                |
# | DiT-XL| 28       | 1152          | 16    | 29.1                |
class STDiT(nn.Module):
    def __init__(
        self,
        input_size=(1, 32, 32),  # T, H, W
        in_channels=4,
        out_channels=4,
        patch_size=(1, 2, 2),  # T, H, W
        hidden_size=1152,  #
        depth=28,  # Number of layers
        num_heads=16,
        mlp_ratio=4.0,
        class_dropout_prob=0.1,
        drop_path=0.0,
        no_temporal_pos_emb=False,
        caption_channels=4096,  # 0 to disable
        model_max_length=120,
        space_scale=1.0,
        time_scale=1.0,
        enable_flashattn=False,
    ):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.hidden_size = hidden_size
        self.patch_size = patch_size
        self.input_size = input_size
        num_patches = np.prod([input_size[i] // patch_size[i] for i in range(3)])
        self.num_patches = num_patches
        self.num_temporal = input_size[0] // patch_size[0]
        self.num_spatial = num_patches // self.num_temporal
        self.num_heads = num_heads
        self.no_temporal_pos_emb = no_temporal_pos_emb
        self.depth = depth
        self.mlp_ratio = mlp_ratio
        self.enable_flashattn = enable_flashattn
        self.space_scale = space_scale
        self.time_scale = time_scale

        if caption_channels == 0:
            print("Warning: caption_channels is 0, disabling text conditioning.")

        self.register_buffer("pos_embed", self.get_spatial_pos_embed())
        self.register_buffer("pos_embed_temporal", self.get_temporal_pos_embed())

        self.x_embedder = PatchEmbed3D(patch_size, in_channels, hidden_size)
        self.t_embedder = TimestepEmbedder(hidden_size)
        self.t_block = nn.Sequential(
            nn.SiLU(), nn.Linear(hidden_size, 6 * hidden_size, bias=True)
        )
        self.y_embedder = (
            CaptionEmbedder(
                in_channels=caption_channels,
                hidden_size=hidden_size,
                uncond_prob=class_dropout_prob,
                act_layer=approx_gelu,
                token_num=model_max_length,
            )
            if caption_channels > 0
            else None
        )

        drop_path = [x.item() for x in torch.linspace(0, drop_path, depth)]
        self.blocks = nn.ModuleList(
            [
                STDiTBlock(
                    self.hidden_size,
                    self.num_heads,
                    mlp_ratio=self.mlp_ratio,
                    drop_path=drop_path[i],
                    enable_flashattn=self.enable_flashattn,
                    d_t=self.num_temporal,
                    d_s=self.num_spatial,
                    uncond=(caption_channels > 0),
                )
                for i in range(self.depth)
            ]
        )
        self.final_layer = T2IFinalLayer(
            hidden_size, np.prod(self.patch_size), self.out_channels
        )

        # init model
        self.initialize_weights()
        self.initialize_temporal()

        # sequence parallel related configs
        self.sp_rank = None

    def forward(self, x, timestep, y=None, mask=None, cond_image=None):
        """
        Forward pass of STDiT.
        Args:
            x (torch.Tensor): latent representation of video; of shape [B, C, T, H, W]
            timestep (torch.Tensor): diffusion time steps; of shape [B]
            y (torch.Tensor): representation of prompts; of shape [B, 1, N_token, C]
            mask (torch.Tensor): mask for selecting prompt tokens; of shape [B, N_token]
        Returns:
            x (torch.Tensor): output latent representation; of shape [B, C, T, H, W]
        """

        # x = x.to(self.dtype)
        # timestep = timestep.to(self.dtype)
        # y = y.to(self.dtype)

        # embedding
        x = self.x_embedder(x)  # [B, N, C]
        # print(x.shape, self.num_temporal, self.num_spatial)
        x = rearrange(
            x, "B (T S) C -> B T S C", T=self.num_temporal, S=self.num_spatial
        )
        x = x + self.pos_embed
        x = rearrange(x, "B T S C -> B (T S) C")

        # shard over the sequence dim if sp is enabled
        # if self.enable_sequence_parallelism:
        #     x = split_forward_gather_backward(x, get_sequence_parallel_group(), dim=1, grad_scale="down")

        t = self.t_embedder(timestep, dtype=x.dtype)  # [B, C]
        t0 = self.t_block(t)  # [B, C]
        if self.y_embedder is not None and y is not None:
            y = self.y_embedder(y, self.training)  # [B, 1, N_token, C]

            if mask is not None:
                if mask.shape[0] != y.shape[0]:
                    mask = mask.repeat(y.shape[0] // mask.shape[0], 1)
                mask = mask.squeeze(1).squeeze(1)
                y = (
                    y.squeeze(1)
                    .masked_select(mask.unsqueeze(-1) != 0)
                    .view(1, -1, x.shape[-1])
                )
                y_lens = mask.sum(dim=1).tolist()
            else:
                y_lens = [y.shape[2]] * y.shape[0]  # N_token * B
                y = y.squeeze(1).view(1, -1, x.shape[-1])
        else:
            y = None
            y_lens = None

        # blocks
        for i, block in enumerate(self.blocks):
            if i == 0:
                tpe = self.pos_embed_temporal
            else:
                tpe = None
            x = block(x=x, t=t0, y=y, mask=y_lens, tpe=tpe)
        # x.shape: [B, N, C]

        # final process
        x = self.final_layer(x, t)  # [B, N, C=T_p * H_p * W_p * C_out]
        x = self.unpatchify(x)  # [B, C_out, T, H, W]

        return x

    def unpatchify(self, x):
        """
        Args:
            x (torch.Tensor): of shape [B, N, C]
        Return:
            x (torch.Tensor): of shape [B, C_out, T, H, W]
        """

        N_t, N_h, N_w = [self.input_size[i] // self.patch_size[i] for i in range(3)]
        T_p, H_p, W_p = self.patch_size
        x = rearrange(
            x,
            "B (N_t N_h N_w) (T_p H_p W_p C_out) -> B C_out (N_t T_p) (N_h H_p) (N_w W_p)",
            N_t=N_t,
            N_h=N_h,
            N_w=N_w,
            T_p=T_p,
            H_p=H_p,
            W_p=W_p,
            C_out=self.out_channels,
        )
        return x

    def unpatchify_old(self, x):
        c = self.out_channels
        t, h, w = [self.input_size[i] // self.patch_size[i] for i in range(3)]
        pt, ph, pw = self.patch_size

        x = x.reshape(shape=(x.shape[0], t, h, w, pt, ph, pw, c))
        x = rearrange(x, "n t h w r p q c -> n c t r h p w q")
        imgs = x.reshape(shape=(x.shape[0], c, t * pt, h * ph, w * pw))
        return imgs

    def get_spatial_pos_embed(self, grid_size=None):
        if grid_size is None:
            grid_size = self.input_size[1:]
        pos_embed = get_2d_sincos_pos_embed(
            self.hidden_size,
            (grid_size[0] // self.patch_size[1], grid_size[1] // self.patch_size[2]),
            scale=self.space_scale,
        )
        pos_embed = (
            torch.from_numpy(pos_embed).float().unsqueeze(0).requires_grad_(False)
        )
        return pos_embed

    def get_temporal_pos_embed(self):
        pos_embed = get_1d_sincos_pos_embed(
            self.hidden_size,
            self.input_size[0] // self.patch_size[0],
            scale=self.time_scale,
        )
        pos_embed = (
            torch.from_numpy(pos_embed).float().unsqueeze(0).requires_grad_(False)
        )
        return pos_embed

    def freeze_not_temporal(self):
        for n, p in self.named_parameters():
            if "attn_temp" not in n:
                p.requires_grad = False

    def freeze_text(self):
        for n, p in self.named_parameters():
            if "cross_attn" in n:
                p.requires_grad = False

    def initialize_temporal(self):
        for block in self.blocks:
            nn.init.constant_(block.attn_temp.proj.weight, 0)
            nn.init.constant_(block.attn_temp.proj.bias, 0)

    def initialize_weights(self):
        # Initialize transformer layers:
        def _basic_init(module):
            if isinstance(module, nn.Linear):
                torch.nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.constant_(module.bias, 0)

        self.apply(_basic_init)

        # Initialize patch_embed like nn.Linear (instead of nn.Conv2d):
        w = self.x_embedder.proj.weight.data
        nn.init.xavier_uniform_(w.view([w.shape[0], -1]))

        # Initialize timestep embedding MLP:
        nn.init.normal_(self.t_embedder.mlp[0].weight, std=0.02)
        nn.init.normal_(self.t_embedder.mlp[2].weight, std=0.02)
        nn.init.normal_(self.t_block[1].weight, std=0.02)

        # Initialize caption embedding MLP:
        if self.y_embedder is not None:
            nn.init.normal_(self.y_embedder.y_proj.fc1.weight, std=0.02)
            nn.init.normal_(self.y_embedder.y_proj.fc2.weight, std=0.02)

        # Zero-out adaLN modulation layers in PixArt blocks:
        for block in self.blocks:
            nn.init.constant_(block.cross_attn.proj.weight, 0)
            nn.init.constant_(block.cross_attn.proj.bias, 0)

        # Zero-out output layers:
        nn.init.constant_(self.final_layer.linear.weight, 0)
        nn.init.constant_(self.final_layer.linear.bias, 0)


@dataclass
class DiffuserSTDiTModelOutput(BaseOutput):
    """
    The output of [`DiffuserSTDiT`].
    Args:
        sample (`torch.FloatTensor` of shape `(batch_size, num_channels, num_frames, height, width)` or `(batch size, num_vector_embeds - 1, num_latent_pixels)` if [`Transformer2DModel`] is discrete):
            The hidden states output conditioned on the `encoder_hidden_states` input. If discrete, returns probability
            distributions for the unnoised latent pixels.
    """

    sample: torch.FloatTensor


class DiffuserSTDiT(ModelMixin, ConfigMixin):
    """
    STDiT: Spatio-Temporal Diffusion Transformer.
    Parameters:
        input_size (tuple): Input size of the video. Default: (1, 32, 32).
        in_channels (int): Number of input video channels. Default: 4.
        out_channels (int): Number of output video channels. Default: 4.
        patch_size (tuple): Patch token size. Default: (1, 2, 2).
        hidden_size (int): Hidden size of the model. Default: 1152.
        depth (int): Number of layers. Default: 28.
        num_heads (int): Number of attention heads. Default: 16.
        mlp_ratio (float): Ratio of hidden to mlp hidden size. Default: 4.0.
        class_dropout_prob (float): Probability of dropping class tokens. Default: 0.1.
        drop_path (float): Drop path rate. Default: 0.0.
        no_temporal_pos_emb (bool): Disable temporal positional embeddings. Default: False.
        caption_channels (int): Number of caption channels. Default: 4096.
        model_max_length (int): Maximum length of the model. Default: 120.
        space_scale (float): Spatial scale. Default: 1.0.
        time_scale (float): Temporal scale. Default: 1.0.
        enable_flashattn (bool): Enable FlashAttention. Default: False.
    """

    @register_to_config
    def __init__(
        self,
        input_size=(1, 32, 32),  # T, H, W
        in_channels=4,
        out_channels=4,
        patch_size=(1, 2, 2),  # T, H, W
        hidden_size=1152,  #
        depth=28,  # Number of layers
        num_heads=16,
        mlp_ratio=4.0,
        class_dropout_prob=0.1,
        drop_path=0.0,
        no_temporal_pos_emb=False,
        caption_channels=4096,  # 0 to disable
        model_max_length=120,
        space_scale=1.0,
        time_scale=1.0,
        enable_flashattn=False,
    ):

        super().__init__()

        self.model = STDiT(
            input_size=input_size,
            in_channels=in_channels,
            out_channels=out_channels,
            patch_size=patch_size,
            hidden_size=hidden_size,
            depth=depth,
            num_heads=num_heads,
            mlp_ratio=mlp_ratio,
            class_dropout_prob=class_dropout_prob,
            drop_path=drop_path,
            no_temporal_pos_emb=no_temporal_pos_emb,
            caption_channels=caption_channels,
            model_max_length=model_max_length,
            space_scale=space_scale,
            time_scale=time_scale,
            enable_flashattn=enable_flashattn,
        )

    def forward(
        self,
        x,
        timestep,
        encoder_hidden_states=None,
        cond_image=None,
        mask=None,
        return_dict=True,
        *args,
        **kwargs,
    ):
        """
        Args:
            x (torch.Tensor): latent representation of video; of shape [B, C, T, H, W]
            timestep (torch.Tensor): diffusion time steps; of shape [B]
            y (torch.Tensor): representation of prompts; of shape [B, 1, N_token, C]
            mask (torch.Tensor): mask for selecting prompt tokens; of shape [B, N_token]
            return_dict (bool): return a dictionary or not. Default: True.
        """
        if type(timestep) == int or timestep.ndim == 0:
            timestep = torch.ones(x.shape[0], device=x.device) * timestep

        encoder_hidden_states = (
            encoder_hidden_states.unsqueeze(1)
            if encoder_hidden_states is not None
            else None
        )

        if cond_image is not None:
            assert (
                x.shape == cond_image.shape
            ), "x and cond_image must have the same shape"
            x = torch.cat([x, cond_image], dim=1)  # B x 2C x T x H x W

        output = self.model(x, timestep, encoder_hidden_states, mask)
        if not return_dict:
            return (output,)

        return DiffuserSTDiTModelOutput(sample=output)


##############################
# Image-Conditionned ST UNet #
##############################


@torch._dynamo.disable
@dataclass
class UNetSTICOutput(BaseOutput):  # UNet-SpatioTemporal-ImageConditionned
    """
    The output of [`UNetSpatioTemporalConditionModel`].
    Args:
        sample (`torch.Tensor` of shape `(batch_size, num_frames, num_channels, height, width)`):
            The hidden states output conditioned on `encoder_hidden_states` input. Output of last layer of model.
    """

    sample: torch.Tensor = None


class UNetSTIC(ModelMixin, ConfigMixin, UNet2DConditionLoadersMixin):
    r"""
    A conditional Spatio-Temporal UNet model that takes a noisy video frames, conditional state, and a timestep and
    returns a sample shaped output.
    This model inherits from [`ModelMixin`]. Check the superclass documentation for it's generic methods implemented
    for all models (such as downloading or saving).
    Parameters:
        sample_size (`int` or `Tuple[int, int]`, *optional*, defaults to `None`):
            Height and width of input/output sample.
        in_channels (`int`, *optional*, defaults to 8): Number of channels in the input sample.
        out_channels (`int`, *optional*, defaults to 4): Number of channels in the output.
        down_block_types (`Tuple[str]`, *optional*, defaults to `("CrossAttnDownBlockSpatioTemporal", "CrossAttnDownBlockSpatioTemporal", "CrossAttnDownBlockSpatioTemporal", "DownBlockSpatioTemporal")`):
            The tuple of downsample blocks to use.
        up_block_types (`Tuple[str]`, *optional*, defaults to `("UpBlockSpatioTemporal", "CrossAttnUpBlockSpatioTemporal", "CrossAttnUpBlockSpatioTemporal", "CrossAttnUpBlockSpatioTemporal")`):
            The tuple of upsample blocks to use.
        block_out_channels (`Tuple[int]`, *optional*, defaults to `(320, 640, 1280, 1280)`):
            The tuple of output channels for each block.
        addition_time_embed_dim: (`int`, defaults to 256):
            Dimension to to encode the additional time ids.
        projection_class_embeddings_input_dim (`int`, defaults to 768):
            The dimension of the projection of encoded `added_time_ids`.
        layers_per_block (`int`, *optional*, defaults to 2): The number of layers per block.
        cross_attention_dim (`int` or `Tuple[int]`, *optional*, defaults to 1280):
            The dimension of the cross attention features.
        transformer_layers_per_block (`int`, `Tuple[int]`, or `Tuple[Tuple]` , *optional*, defaults to 1):
            The number of transformer blocks of type [`~models.attention.BasicTransformerBlock`]. Only relevant for
            [`~models.unets.unet_3d_blocks.CrossAttnDownBlockSpatioTemporal`],
            [`~models.unets.unet_3d_blocks.CrossAttnUpBlockSpatioTemporal`],
            [`~models.unets.unet_3d_blocks.UNetMidBlockSpatioTemporal`].
        num_attention_heads (`int`, `Tuple[int]`, defaults to `(5, 10, 10, 20)`):
            The number of attention heads.
        dropout (`float`, *optional*, defaults to 0.0): The dropout probability to use.
    """

    _supports_gradient_checkpointing = True

    @register_to_config
    def __init__(
        self,
        sample_size: Optional[int] = None,
        in_channels: int = 8,
        out_channels: int = 4,
        down_block_types: Tuple[str] = (
            "CrossAttnDownBlockSpatioTemporal",
            "CrossAttnDownBlockSpatioTemporal",
            "CrossAttnDownBlockSpatioTemporal",
            "DownBlockSpatioTemporal",
        ),
        up_block_types: Tuple[str] = (
            "UpBlockSpatioTemporal",
            "CrossAttnUpBlockSpatioTemporal",
            "CrossAttnUpBlockSpatioTemporal",
            "CrossAttnUpBlockSpatioTemporal",
        ),
        block_out_channels: Tuple[int] = (320, 640, 1280, 1280),
        addition_time_embed_dim: int = 256,
        projection_class_embeddings_input_dim: int = 768,
        layers_per_block: Union[int, Tuple[int]] = 2,
        cross_attention_dim: Union[int, Tuple[int]] = 1024,
        transformer_layers_per_block: Union[int, Tuple[int], Tuple[Tuple]] = 1,
        num_attention_heads: Union[int, Tuple[int]] = (5, 10, 20, 20),
        num_frames: int = 25,
    ):
        super().__init__()

        self.sample_size = sample_size

        # Check inputs
        if len(down_block_types) != len(up_block_types):
            raise ValueError(
                f"Must provide the same number of `down_block_types` as `up_block_types`. `down_block_types`: {down_block_types}. `up_block_types`: {up_block_types}."
            )

        if len(block_out_channels) != len(down_block_types):
            raise ValueError(
                f"Must provide the same number of `block_out_channels` as `down_block_types`. `block_out_channels`: {block_out_channels}. `down_block_types`: {down_block_types}."
            )

        if not isinstance(num_attention_heads, int) and len(num_attention_heads) != len(
            down_block_types
        ):
            raise ValueError(
                f"Must provide the same number of `num_attention_heads` as `down_block_types`. `num_attention_heads`: {num_attention_heads}. `down_block_types`: {down_block_types}."
            )

        if isinstance(cross_attention_dim, list) and len(cross_attention_dim) != len(
            down_block_types
        ):
            raise ValueError(
                f"Must provide the same number of `cross_attention_dim` as `down_block_types`. `cross_attention_dim`: {cross_attention_dim}. `down_block_types`: {down_block_types}."
            )

        if not isinstance(layers_per_block, int) and len(layers_per_block) != len(
            down_block_types
        ):
            raise ValueError(
                f"Must provide the same number of `layers_per_block` as `down_block_types`. `layers_per_block`: {layers_per_block}. `down_block_types`: {down_block_types}."
            )

        # input
        self.conv_in = nn.Conv2d(
            in_channels,
            block_out_channels[0],
            kernel_size=3,
            padding=1,
        )

        # time
        time_embed_dim = block_out_channels[0] * 4

        self.time_proj = Timesteps(block_out_channels[0], True, downscale_freq_shift=0)
        timestep_input_dim = block_out_channels[0]

        self.time_embedding = TimestepEmbedding(timestep_input_dim, time_embed_dim)

        # self.add_time_proj = Timesteps(
        #     addition_time_embed_dim, True, downscale_freq_shift=0
        # )
        # self.add_embedding = TimestepEmbedding(
        #     projection_class_embeddings_input_dim, time_embed_dim
        # )

        self.down_blocks = nn.ModuleList([])
        self.up_blocks = nn.ModuleList([])

        if isinstance(num_attention_heads, int):
            num_attention_heads = (num_attention_heads,) * len(down_block_types)

        if isinstance(cross_attention_dim, int):
            cross_attention_dim = (cross_attention_dim,) * len(down_block_types)

        if isinstance(layers_per_block, int):
            layers_per_block = [layers_per_block] * len(down_block_types)

        if isinstance(transformer_layers_per_block, int):
            transformer_layers_per_block = [transformer_layers_per_block] * len(
                down_block_types
            )

        blocks_time_embed_dim = time_embed_dim

        # down
        output_channel = block_out_channels[0]
        for i, down_block_type in enumerate(down_block_types):
            input_channel = output_channel
            output_channel = block_out_channels[i]
            is_final_block = i == len(block_out_channels) - 1

            down_block = get_down_block_3d(
                down_block_type,
                num_layers=layers_per_block[i],
                transformer_layers_per_block=transformer_layers_per_block[i],
                in_channels=input_channel,
                out_channels=output_channel,
                temb_channels=blocks_time_embed_dim,
                add_downsample=not is_final_block,
                resnet_eps=1e-5,
                cross_attention_dim=cross_attention_dim[i],
                num_attention_heads=num_attention_heads[i],
                resnet_act_fn="silu",
            )
            self.down_blocks.append(down_block)

        # mid
        self.mid_block = UNetMidBlockSpatioTemporal(
            block_out_channels[-1],
            temb_channels=blocks_time_embed_dim,
            transformer_layers_per_block=transformer_layers_per_block[-1],
            cross_attention_dim=cross_attention_dim[-1],
            num_attention_heads=num_attention_heads[-1],
        )

        # count how many layers upsample the images
        self.num_upsamplers = 0

        # up
        reversed_block_out_channels = list(reversed(block_out_channels))
        reversed_num_attention_heads = list(reversed(num_attention_heads))
        reversed_layers_per_block = list(reversed(layers_per_block))
        reversed_cross_attention_dim = list(reversed(cross_attention_dim))
        reversed_transformer_layers_per_block = list(
            reversed(transformer_layers_per_block)
        )

        output_channel = reversed_block_out_channels[0]
        for i, up_block_type in enumerate(up_block_types):
            is_final_block = i == len(block_out_channels) - 1

            prev_output_channel = output_channel
            output_channel = reversed_block_out_channels[i]
            input_channel = reversed_block_out_channels[
                min(i + 1, len(block_out_channels) - 1)
            ]

            # add upsample block for all BUT final layer
            if not is_final_block:
                add_upsample = True
                self.num_upsamplers += 1
            else:
                add_upsample = False

            up_block = get_up_block_3d(
                up_block_type,
                num_layers=reversed_layers_per_block[i] + 1,
                transformer_layers_per_block=reversed_transformer_layers_per_block[i],
                in_channels=input_channel,
                out_channels=output_channel,
                prev_output_channel=prev_output_channel,
                temb_channels=blocks_time_embed_dim,
                add_upsample=add_upsample,
                resnet_eps=1e-5,
                resolution_idx=i,
                cross_attention_dim=reversed_cross_attention_dim[i],
                num_attention_heads=reversed_num_attention_heads[i],
                resnet_act_fn="silu",
            )
            self.up_blocks.append(up_block)
            prev_output_channel = output_channel

        # out
        self.conv_norm_out = nn.GroupNorm(
            num_channels=block_out_channels[0], num_groups=32, eps=1e-5
        )
        self.conv_act = nn.SiLU()

        self.conv_out = nn.Conv2d(
            block_out_channels[0],
            out_channels,
            kernel_size=3,
            padding=1,
        )

        # self.set_default_attn_processor()

    @property
    def attn_processors(self) -> Dict[str, AttentionProcessor]:
        r"""
        Returns:
            `dict` of attention processors: A dictionary containing all attention processors used in the model with
            indexed by its weight name.
        """
        # set recursively
        processors = {}

        def fn_recursive_add_processors(
            name: str,
            module: torch.nn.Module,
            processors: Dict[str, AttentionProcessor],
        ):
            if hasattr(module, "get_processor"):
                processors[f"{name}.processor"] = module.get_processor()

            for sub_name, child in module.named_children():
                fn_recursive_add_processors(f"{name}.{sub_name}", child, processors)

            return processors

        for name, module in self.named_children():
            fn_recursive_add_processors(name, module, processors)

        return processors

    def set_attn_processor(self, processor):
        r"""
        Sets the attention processor to use to compute attention.
        Parameters:
            processor (`dict` of `AttentionProcessor` or only `AttentionProcessor`):
                The instantiated processor class or a dictionary of processor classes that will be set as the processor
                for **all** `Attention` layers.
                If `processor` is a dict, the key needs to define the path to the corresponding cross attention
                processor. This is strongly recommended when setting trainable attention processors.
        """
        count = len(self.attn_processors.keys())

        if isinstance(processor, dict) and len(processor) != count:
            raise ValueError(
                f"A dict of processors was passed, but the number of processors {len(processor)} does not match the"
                f" number of attention layers: {count}. Please make sure to pass {count} processor classes."
            )

        def fn_recursive_attn_processor(name: str, module: torch.nn.Module, processor):
            if hasattr(module, "set_processor"):
                if not isinstance(processor, dict):
                    module.set_processor(processor)
                else:
                    module.set_processor(processor.pop(f"{name}.processor"))

            for sub_name, child in module.named_children():
                fn_recursive_attn_processor(f"{name}.{sub_name}", child, processor)

        for name, module in self.named_children():
            fn_recursive_attn_processor(name, module, processor)

    def set_default_attn_processor(self):
        """
        Disables custom attention processors and sets the default attention implementation.
        """
        if all(
            proc.__class__ in CROSS_ATTENTION_PROCESSORS
            for proc in self.attn_processors.values()
        ):
            processor = AttnProcessor()
        else:
            raise ValueError(
                f"Cannot call `set_default_attn_processor` when attention processors are of type {next(iter(self.attn_processors.values()))}"
            )

        self.set_attn_processor(processor)

    def _set_gradient_checkpointing(self, module, value=False):
        if hasattr(module, "gradient_checkpointing"):
            module.gradient_checkpointing = value

    # Copied from diffusers.models.unets.unet_3d_condition.UNet3DConditionModel.enable_forward_chunking
    def enable_forward_chunking(
        self, chunk_size: Optional[int] = None, dim: int = 0
    ) -> None:
        """
        Sets the attention processor to use [feed forward
        chunking](https://huggingface.co/blog/reformer#2-chunked-feed-forward-layers).
        Parameters:
            chunk_size (`int`, *optional*):
                The chunk size of the feed-forward layers. If not specified, will run feed-forward layer individually
                over each tensor of dim=`dim`.
            dim (`int`, *optional*, defaults to `0`):
                The dimension over which the feed-forward computation should be chunked. Choose between dim=0 (batch)
                or dim=1 (sequence length).
        """
        if dim not in [0, 1]:
            raise ValueError(f"Make sure to set `dim` to either 0 or 1, not {dim}")

        # By default chunk size is 1
        chunk_size = chunk_size or 1

        def fn_recursive_feed_forward(
            module: torch.nn.Module, chunk_size: int, dim: int
        ):
            if hasattr(module, "set_chunk_feed_forward"):
                module.set_chunk_feed_forward(chunk_size=chunk_size, dim=dim)

            for child in module.children():
                fn_recursive_feed_forward(child, chunk_size, dim)

        for module in self.children():
            fn_recursive_feed_forward(module, chunk_size, dim)

    def forward(
        self,
        x: torch.Tensor,
        timestep: Union[torch.Tensor, float, int],
        encoder_hidden_states: torch.Tensor,
        cond_image=None,
        mask=None,
        # added_time_ids: torch.Tensor,
        return_dict: bool = True,
    ) -> Union[UNetSTICOutput, Tuple]:
        r"""
        The [`UNetSpatioTemporalConditionModel`] forward method.
        Args:
            sample (`torch.Tensor`):
                The noisy input tensor with the following shape `(batch, num_frames, channel, height, width)`.
            timestep (`torch.Tensor` or `float` or `int`): The number of timesteps to denoise an input.
            encoder_hidden_states (`torch.Tensor`):
                The encoder hidden states with shape `(batch, sequence_length, cross_attention_dim)`.
            added_time_ids: (`torch.Tensor`):
                The additional time ids with shape `(batch, num_additional_ids)`. These are encoded with sinusoidal
                embeddings and added to the time embeddings.
            return_dict (`bool`, *optional*, defaults to `True`):
                Whether or not to return a [`~models.unet_slatio_temporal.UNetSTICOutput`] instead
                of a plain tuple.
        Returns:
            [`~models.unet_slatio_temporal.UNetSTICOutput`] or `tuple`:
                If `return_dict` is True, an [`~models.unet_slatio_temporal.UNetSTICOutput`] is
                returned, otherwise a `tuple` is returned where the first element is the sample tensor.
        """

        sample = torch.cat([x, cond_image], dim=1)  # B C+1 T H W

        # pad to multiple of 2**n
        res_target = 2 ** (np.ceil(np.log2(sample.shape[-1])).astype(int))
        padding = (res_target - sample.shape[-1]) // 2
        sample = F.pad(
            sample, (padding, padding, padding, padding, 0, 0), mode="circular"
        )

        # reshape from B C T H W to B T C H W
        sample = sample.permute(0, 2, 1, 3, 4)

        # 1. time
        timesteps = timestep
        if not torch.is_tensor(timesteps):
            # TODO: this requires sync between CPU and GPU. So try to pass timesteps as tensors if you can
            # This would be a good case for the `match` statement (Python 3.10+)
            is_mps = sample.device.type == "mps"
            if isinstance(timestep, float):
                dtype = torch.float32 if is_mps else torch.float64
            else:
                dtype = torch.int32 if is_mps else torch.int64
            timesteps = torch.tensor([timesteps], dtype=dtype, device=sample.device)
        elif len(timesteps.shape) == 0:
            timesteps = timesteps[None].to(sample.device)

        # broadcast to batch dimension in a way that's compatible with ONNX/Core ML
        batch_size, num_frames = sample.shape[:2]
        timesteps = timesteps.expand(batch_size)

        t_emb = self.time_proj(timesteps)

        # `Timesteps` does not contain any weights and will always return f32 tensors
        # but time_embedding might actually be running in fp16. so we need to cast here.
        # there might be better ways to encapsulate this.
        t_emb = t_emb.to(dtype=sample.dtype)

        emb = self.time_embedding(t_emb)

        # time_embeds = self.add_time_proj(added_time_ids.flatten())
        # time_embeds = time_embeds.reshape((batch_size, -1))
        # time_embeds = time_embeds.to(emb.dtype)
        # aug_emb = self.add_embedding(time_embeds)
        # emb = emb + aug_emb

        # Flatten the batch and frames dimensions
        # sample: [batch, frames, channels, height, width] -> [batch * frames, channels, height, width]
        sample = sample.flatten(0, 1)
        # Repeat the embeddings num_video_frames times
        # emb: [batch, channels] -> [batch * frames, channels]
        emb = emb.repeat_interleave(num_frames, dim=0)
        # encoder_hidden_states: [batch, 1, channels] -> [batch * frames, 1, channels]
        encoder_hidden_states = encoder_hidden_states.repeat_interleave(
            num_frames, dim=0
        )

        # 2. pre-process
        sample = self.conv_in(sample)

        image_only_indicator = torch.zeros(
            batch_size, num_frames, dtype=sample.dtype, device=sample.device
        )

        down_block_res_samples = (sample,)
        for downsample_block in self.down_blocks:
            if (
                hasattr(downsample_block, "has_cross_attention")
                and downsample_block.has_cross_attention
            ):
                sample, res_samples = downsample_block(
                    hidden_states=sample,
                    temb=emb,
                    encoder_hidden_states=encoder_hidden_states,
                    image_only_indicator=image_only_indicator,
                )
            else:
                sample, res_samples = downsample_block(
                    hidden_states=sample,
                    temb=emb,
                    image_only_indicator=image_only_indicator,
                )

            down_block_res_samples += res_samples

        # 4. mid
        sample = self.mid_block(
            hidden_states=sample,
            temb=emb,
            encoder_hidden_states=encoder_hidden_states,
            image_only_indicator=image_only_indicator,
        )

        # 5. up
        for i, upsample_block in enumerate(self.up_blocks):
            res_samples = down_block_res_samples[-len(upsample_block.resnets) :]
            down_block_res_samples = down_block_res_samples[
                : -len(upsample_block.resnets)
            ]

            if (
                hasattr(upsample_block, "has_cross_attention")
                and upsample_block.has_cross_attention
            ):
                sample = upsample_block(
                    hidden_states=sample,
                    temb=emb,
                    res_hidden_states_tuple=res_samples,
                    encoder_hidden_states=encoder_hidden_states,
                    image_only_indicator=image_only_indicator,
                )
            else:
                sample = upsample_block(
                    hidden_states=sample,
                    temb=emb,
                    res_hidden_states_tuple=res_samples,
                    image_only_indicator=image_only_indicator,
                )

        # 6. post-process
        sample = self.conv_norm_out(sample)
        sample = self.conv_act(sample)
        sample = self.conv_out(sample)

        # 7. Reshape back to original shape
        sample = sample.reshape(batch_size, num_frames, *sample.shape[1:])

        if padding > 0:
            sample = sample[:, :, :, padding:-padding, padding:-padding]

        # reshape back to B C T H W
        sample = sample.permute(0, 2, 1, 3, 4)

        if not return_dict:
            return (sample,)

        return UNetSTICOutput(sample=sample)


class ContrastiveModel(nn.Module):
    def __init__(self, in_channels, out_channels, backbone=None, kl_loss_weight=0.0):
        super(ContrastiveModel, self).__init__()

        assert backbone is not None, "Backbone must be provided."
        self.backbone = backbone

        self.backbone = self.patch_backbone(self.backbone, in_channels, out_channels)

        self.fc_end = nn.Linear(out_channels, 1)

        self.kl_loss_weight = kl_loss_weight

    @classmethod
    def patch_backbone(cls, backbone, in_channels, out_channels):
        if "ResNet" in backbone.__class__.__name__:
            backbone.model.conv1 = nn.Conv2d(
                in_channels,
                64,
                kernel_size=(7, 7),
                stride=(2, 2),
                padding=(3, 3),
                bias=False,
            )
            backbone.model.fc = nn.Linear(
                in_features=512, out_features=out_channels, bias=True
            )
        else:
            raise Exception(
                "Invalid argument: "
                + backbone.__class__.__name__
                + "\nChoose ResNet! Other architectures are not yet implemented in this framework."
            )

        return backbone

    def forward_once(self, x):
        features = self.backbone(x)
        output = torch.sigmoid(features)
        return output, features

    def forward_constrastive(self, input1, input2):
        y1 = self.forward_once(input1)
        y2 = self.forward_once(input2)

        difference = torch.abs(y1 - y2)
        output = self.fc_end(difference)  # linear layer

        return output  # B x 1

    def forward_fused(self, input1, input2):
        inputs = torch.cat((input1, input2), dim=0)  # 2B x C x H x W
        outputs, features = self.forward_once(inputs)
        y1, y2 = torch.split(outputs, outputs.size(0) // 2, dim=0)
        difference = torch.abs(y1 - y2)
        output = self.fc_end(difference)

        # Compute KL divergence
        if self.kl_loss_weight > 0:
            mu = torch.mean(features, dim=0)
            var = torch.var(features, dim=0) + 1e-6  # Add epsilon to avoid log(0)
            kl_loss = 0.5 * torch.sum(mu.pow(2) + var - torch.log(var) - 1)
        else:
            kl_loss = torch.zeros((1,), device=output.device)
        return output, kl_loss

    def loss(self, output, target):
        return nn.functional.binary_cross_entropy_with_logits(output, target[:, None])

    def forward(self, input1, input2, target):
        y_hat, kl_loss = self.forward_fused(input1, input2)
        loss = self.loss(y_hat, target)
        total_loss = loss + self.kl_loss_weight * kl_loss
        return total_loss, loss, kl_loss


class ResNet18(ModelMixin, ConfigMixin):
    @register_to_config
    def __init__(self, weights=None, progress=False):
        super(ResNet18, self).__init__()
        self.model = resnet18(weights=weights, progress=progress)

    def forward(self, x):
        return self.model(x)

