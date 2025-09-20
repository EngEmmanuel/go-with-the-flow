"""
Quick utility to pass a small dummy video through the models defined in `store/models.py`.
This is intended as a shape / API exercise and minimal smoke test. It uses small tensors so it
should run on CPU. Many features (weights, attention processors, xformers) are NOT configured
— this is deliberate: the goal is to show how to call the models and what tensor shapes they
expect.

Usage:
    python run_video_through_models.py --model stdit
    python run_video_through_models.py --model diffuser
    python run_video_through_models.py --model unetstic

The script constructs a small random video tensor and prints shapes at each stage.
"""

import argparse
import torch
import torch.nn as nn

import my_src.models as sm


def make_random_video(batch=1, channels=3, frames=2, height=16, width=16, device="cpu"):
    """Create a random float32 video tensor.
    Returns tensor with shape: [B, C, T, H, W]
    dtype: torch.float32
    """
    return torch.randn(batch, channels, frames, height, width, dtype=torch.float32, device=device)


def run_stdit(video: torch.Tensor):
    """Run the plain STDiT model.

    Inputs:
      video: [B, C, T, H, W]
    STDiT.forward signature expects:
      x: [B, C, T, H, W]
      timestep: [B] (or scalar)

    Output:
      tensor with same shape [B, out_channels, T, H, W]
    """
    B, C, T, H, W = video.shape
    print(f"STDiT: input video shape {video.shape}")

    # Instantiate a tiny STDiT so it's cheap to run
    # enable a small caption/text conditioning to demonstrate cross-attention
    caption_channels = 32
    token_num = 8
    stdit = sm.STDiT(
        input_size=(T, H, W),
        in_channels=C,
        out_channels=C,
        patch_size=(1, 2, 2),
        hidden_size=64,
        depth=2,
        num_heads=4,
        caption_channels=caption_channels,
        model_max_length=token_num,
    )

    # small timestep vector (one per batch)
    timesteps = torch.zeros(B, dtype=torch.float32)

    # Dummy text prompt tensor `y` expected by STDiT: shape [B, 1, N_token, caption_channels]
    y = torch.randn(B, 1, token_num, caption_channels, dtype=video.dtype)

    out = stdit(video, timesteps, y=y, mask=None)
    print(f"STDiT output shape: {out.shape}")
    return out


def run_diffuser(video: torch.Tensor):
    """Run the Diffuser wrapper `DiffuserSTDiT` which internally calls STDiT.

    Input expected by DiffuserSTDiT.forward:
      x: [B, C, T, H, W]
      timestep: [B] or scalar
      encoder_hidden_states: optional [B, 1, cross_attention_dim]

    Output: DiffuserSTDiTModelOutput with `.sample` of shape [B, out_channels, T, H, W]
    """
    B, C, T, H, W = video.shape
    print(f"DiffuserSTDiT: input video shape {video.shape}")

    # Configure a small caption encoder for the Diffuser wrapper
    caption_channels = 32
    token_num = 8
    diffuser = sm.DiffuserSTDiT(
        input_size=(T, H, W),
        in_channels=C,
        out_channels=C,
        patch_size=(1, 2, 2),
        hidden_size=64,
        depth=2,
        num_heads=4,
        caption_channels=caption_channels,
        model_max_length=token_num,
    )

    timesteps = torch.zeros(B, dtype=torch.float32)
    # encoder_hidden_states for DiffuserSTDiT should have shape [B, N_token, caption_channels]
    encoder_hidden_states = torch.randn(B, token_num, caption_channels, dtype=video.dtype)

    out = diffuser(video, timesteps, encoder_hidden_states=encoder_hidden_states)
    sample = out.sample if hasattr(out, "sample") else out[0]
    print(f"DiffuserSTDiT output shape: {sample.shape}")
    return sample


def run_unetstic(video: torch.Tensor):
    """Run the Spatio-Temporal UNet `UNetSTIC`.

    UNetSTIC.forward expects:
      x: tensor with shape (batch, num_frames, channel, height, width)
      cond_image: tensor the same shape as `x` that is concatenated inside the forward pass
      encoder_hidden_states: [batch, sequence_length, cross_attention_dim]

    In this script we provide a dummy `cond_image` (zeros) and a small random
    `encoder_hidden_states` to satisfy the API.
    """
    # convert [B, C, T, H, W] -> [B, T, C, H, W]
    x_bcthw = video.permute(0, 2, 1, 3, 4).contiguous()
    B, T, C, H, W = x_bcthw.shape
    print(f"UNetSTIC: input (converted) shape {x_bcthw.shape} (B, T, C, H, W)")

    # cond_image must have same shape as `x` (B, C, T, H, W), the model concatenates them
    cond_image = video.clone()

    # UNetSTIC expects `in_channels` to match the concatenated channels (C + C here)
    concat_in_channels = C + C
    # Provide dummy encoder_hidden_states for cross-attention: shape [B, seq_len, cross_attention_dim]
    seq_len = 6
    cross_attention_dim = 1024
    encoder_hidden_states = torch.randn(B, seq_len, cross_attention_dim, dtype=video.dtype)

    unet = sm.UNetSTIC(
        sample_size=H,
        in_channels=concat_in_channels,
        out_channels=C,
        num_frames=T,
    )

    # UNetSTIC expects x as [batch, num_frames, channel, height, width], but `cond_image` is [B, C, T, H, W]
    # inside forward the implementation does `sample = torch.cat([x, cond_image], dim=1)` where x is B,C,T,H,W,
    # so we need to convert back to B,C,T,H,W for the concatenation call. The forward implementation performs
    # the concat at the top-level, so we'll call forward with the shapes it needs.

    # convert x back to B,C,T,H,W for the call
    x_bcthw_for_concat = x_bcthw.permute(0, 2, 1, 3, 4).contiguous()
    timesteps = 0
    out = unet(x_bcthw_for_concat, timesteps, encoder_hidden_states=encoder_hidden_states, cond_image=cond_image)
    sample = out.sample if hasattr(out, "sample") else out[0]
    print(f"UNetSTIC output shape: {sample.shape}")  # expected [B, C, T, H, W]
    return sample


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", choices=["stdit", "diffuser", "unetstic"], default="stdit")
    parser.add_argument("--device", default="cpu")
    args = parser.parse_args()

    device = args.device

    # tiny example video
    video = make_random_video(batch=1, channels=3, frames=2, height=16, width=16, device=device)

    if args.model == "stdit":
        run_stdit(video)
    elif args.model == "diffuser":
        run_diffuser(video)
    elif args.model == "unetstic":
        run_unetstic(video)


if __name__ == "__main__":
    main()
