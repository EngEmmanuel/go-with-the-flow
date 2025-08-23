# Model summary: main classes in `store/models.py`

This document summarizes the four primary model classes and their main init/forward arguments found in `store/models.py`.

## 1) STDiT

- File/class: `STDiT` (class)
- Purpose: Spatio-Temporal Diffusion Transformer for videos. Processes video latents with patch embeddings + transformer blocks that separate spatial and temporal attention.
- Main init args (common defaults shown):
  - `input_size=(T, H, W)` – tuple describing the frames, height and width (e.g. (1, 32, 32)).
  - `in_channels` (int) – input channels of video (e.g. 3 or 4).
  - `out_channels` (int) – output channels (e.g. 3 or 4).
  - `patch_size=(pt, ph, pw)` – patch tokenization size, default (1,2,2).
  - `hidden_size` (int) – transformer hidden dimension (e.g. 1152).
  - `depth` (int) – number of STDiT transformer blocks.
  - `num_heads` (int) – attention heads for the transformer.
  - `caption_channels` (int) – dimensionality of token embeddings for text conditioning (set 0 to disable).
  - `model_max_length` (int) – max number of text tokens.
- Forward signature:
  - `forward(x, timestep, y=None, mask=None, cond_image=None)`
  - `x`: Tensor `[B, C, T, H, W]` (video latents)
  - `timestep`: Tensor `[B]` or scalar
  - `y`: Optional text tokens tensor `[B, 1, N_token, caption_channels]`
  - Returns: Tensor `[B, out_channels, T, H, W]`

 - Intuitive meanings (common diffusion conventions):
   - `x` is the noisy latent at diffusion timestep `t` (i.e. the video latent after noise has been added). The model operates in latent space and attempts to predict either the added noise (epsilon) or a denoised latent, depending on training objective.
   - `timestep` selects the noise level/scale applied to `x`.
   - `y` / `mask` are conditioning tokens used in cross-attention; they usually represent semantic prompts (text or learned tokens) and are not spatial images.
   - `cond_image` (when used) supplies per-pixel conditional frames and is concatenated to `x` along channels; it provides spatial context (observed frames, masks, etc.).
   - The returned tensor is model-dependent: in a noise-prediction setup it is the predicted noise (epsilon) that should be subtracted from `x` to obtain a cleaner latent; in a denoising direct-prediction setup it is the predicted denoised latent. Check the model checkpoint/training script to know which objective was used.

## 2) DiffuserSTDiT

- File/class: `DiffuserSTDiT` (wrapper)
- Purpose: ModelWrapper (ModelMixin) that contains an `STDiT` instance and provides a Diffusers-style forward API.
- Main init args: same as STDiT but passed via `register_to_config`.
- Forward signature:
  - `forward(x, timestep, encoder_hidden_states=None, cond_image=None, mask=None, return_dict=True)`
  - `x`: `[B, C, T, H, W]`
  - `encoder_hidden_states`: `[B, 1, cross_attention_dim]` (or None)
  - Returns `DiffuserSTDiTModelOutput(sample=...)` where `sample` is `[B, out_channels, T, H, W]`.

 - Intuitive meanings:
   - `x` is the noisy video latent at the current diffusion timestep.
   - `encoder_hidden_states` contains token embeddings (text or frame-level features) which the model uses via cross-attention; these are not spatial maps unless explicitly encoded as such.
   - `cond_image` is spatial conditioning (per-frame images or masks) concatenated to `x` and provides observed pixel context.
   - The `sample` field in the returned `DiffuserSTDiTModelOutput` follows the training objective: it is typically the model's prediction (commonly the noise term) and must be interpreted according to whether the checkpoint was trained to predict noise, velocity, or the clean latent.

## 3) UNetSTIC

- File/class: `UNetSTIC` (ModelMixin)
- Purpose: Spatio-Temporal UNet with cross-attention and optional image-conditioned inputs. Internally flattens frames into the batch dimension for 2D UNet blocks.
- Main init args:
  - `sample_size`: image height/width
  - `in_channels`: input channel count (note: forward concatenates `cond_image` along channels before processing)
  - `out_channels`: output channels
  - `down_block_types`, `up_block_types`, `block_out_channels`, etc. – configuration for UNet blocks
  - `num_frames`: number of frames per video
- Forward signature:
  - `forward(x, timestep, encoder_hidden_states, cond_image=None, mask=None, return_dict=True)`
  - `x`: **expects** shape `(batch, num_frames, channel, height, width)`
  - `cond_image`: tensor with same shape as `x` but internal code concatenates along channel dim before processing. In implementation the call `sample = torch.cat([x, cond_image], dim=1)` expects both in shape `(B, C, T, H, W)` prior to that concat.
  - `encoder_hidden_states`: `[B, seq_len, cross_attention_dim]` used for cross-attention in several blocks.
  - Returns: `UNetSTICOutput.sample` of shape `[B, out_channels, T, H, W]`.

 - Intuitive meanings:
   - `x` is the noisy video latent (per-frame) at timestep `t` and is the primary input the UNet denoiser operates on.
   - `timestep` encodes the noise level applied to `x`.
   - `cond_image` carries spatial conditioning (observed frames, binary masks, or stacked context frames). It is concatenated channel-wise to `x` so the UNet receives both noisy input and observed context simultaneously.
   - `encoder_hidden_states` are global token embeddings (text or pooled frame features) used via cross-attention; the UNet treats them as external context rather than per-pixel signals.
   - The output `sample` is the model's prediction according to the training objective (commonly predicted noise epsilon). To obtain a denoised latent you either subtract the predicted noise from `x` (noise-prediction) or use the model output directly if it predicts clean latents.

## 4) SegDiTTransformer2DModel

- File/class: `SegDiTTransformer2DModel` (ModelMixin, ConfigMixin)
- Purpose: 2D DiT-style Transformer for images. Accepts optional `segmentation` channel to concatenate with input image and supports timestep & class conditioning via AdaLayerNorm.
- Main init args:
  - `num_attention_heads`, `attention_head_dim`, `in_channels`, `out_channels`, `num_layers`, `dropout`, `sample_size`, `patch_size`, `num_embeds_ada_norm`, etc.
- Forward signature:
  - `forward(hidden_states, timestep=None, class_labels=None, cross_attention_kwargs=None, segmentation=None, return_dict=True)`
  - `hidden_states`: for continuous input expected shape `(B, C, H, W)`.
  - Optional `segmentation` is concatenated to channels: `hidden_states = torch.cat([hidden_states, segmentation], dim=1)`
  - Returns: `Transformer2DModelOutput(sample=...)` where `sample` is an image-shaped tensor `(B, out_channels, H, W)`.

 - Intuitive meanings:
   - `hidden_states` is the input image (or image latent) and for diffusion-style training is typically a noisy image at timestep `t`.
   - `timestep` encodes the noise scale.
   - `class_labels` provide class-conditioning (classifier-free guidance setups may randomly drop them during training).
   - `segmentation` is a spatial conditioning map concatenated to the input channels (useful for segmentation-conditioned generation or inpainting masks).
   - `sample` is the model's prediction; in diffusion setups it is commonly the predicted noise that was added to `hidden_states` and must be used in the sampling update rule appropriate for the trained objective.

---

Notes & recommendations
- Shapes are precise: STDiT & Diffuser expect `[B, C, T, H, W]`. UNetSTIC uses `[B, T, C, H, W]` in its API but concatenates `cond_image` as `[B, C, T, H, W]` (so callers sometimes need to permute shapes accordingly).
- `caption_channels` and `encoder_hidden_states` control text conditioning: provide matching tensor shapes (`[B, 1, N_token, caption_channels]` for STDiT, `[B, seq_len, cross_attention_dim]` for UNetSTIC/Diffuser wrappers).
- If you want runnable examples using real videos, choose a loader (OpenCV / decord / torchvision.io) and decide whether to use CPU-only or GPU (handles dtype / mps quirks).

If you want, I can:
- Add a small helper to read MP4 and convert frames to the expected tensors.
- Add CLI options to select sizes, enable GPU and load checkpoints.
