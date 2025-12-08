import time
import torch
import pandas as pd

from tqdm import tqdm
from pathlib import Path
from lightning.pytorch.callbacks import Callback
from evaluation.ef_evaluation_schemes import UnscaleLatents


class SampleAndCheckpointCallback(Callback):
	'''
	Callback to sample latents from the model and save checkpoints at specified 
	intervals during training. Checkpoints saved in this callback contain only 
	model weights.
	'''
	def __init__(self, cfg, sample_dir: Path, sample_dl, checkpoint_dir: Path, debug=False, device='cuda'):
		super().__init__()
		self.cfg = cfg
		self.sample_dir = sample_dir
		self.sample_dl = sample_dl
		self.checkpoint_dir = checkpoint_dir
		self._last_sample_epoch = 0  
		self.debug = debug
		self.device = device
		# Make sure dirs exist
		self.sample_dir.mkdir(parents=True, exist_ok=True)
		self.checkpoint_dir.mkdir(parents=True, exist_ok=True)

	def on_validation_epoch_end(self, trainer, pl_module):
		self._sample_step(trainer, pl_module)
		
	def on_fit_end(self, trainer, pl_module):
		self._sample_step(trainer, pl_module, last=True)

	def _sample_step(self, trainer, pl_module, last=False):
		pl_module.model.to(self.device)

		if self.sample_dir is None or trainer.sanity_checking:
			return

		epoch = trainer.current_epoch
		is_sample_step = (
			epoch % self.cfg.sample.every_n_epochs == 0
			and epoch != self._last_sample_epoch
		)
		if last:
			is_sample_step = True
			print('Sampling last ckpt latents')

		if is_sample_step:
			if trainer.is_global_zero:
				out_name = 'last' if last else None
				sample_latents_from_model(
					model=pl_module.model,
					dl_list=[self.sample_dl],
					run_cfg=self.cfg,
					epoch=epoch,
					step=trainer.global_step,
					device=self.device,
					samples_dir=self.sample_dir,
					out_name=out_name
				)

				self._last_sample_epoch = epoch

			trainer.strategy.barrier()  # Ensure all processes sync here

			# Save checkpoint right after sampling
			if not last:
				ckpt_name = f"sample-epoch={epoch}-step={trainer.global_step}.ckpt"
				trainer.save_checkpoint(
					str(self.checkpoint_dir / ckpt_name),
					weights_only=True
				) # Must be called on all ranks



def apply_clamped_wrap_to_tensor(tensor: torch.Tensor, lower=0.15, upper=0.85) -> torch.Tensor:
	"""
	Applies the clamped wrap-around function to a tensor of shape (B, 1, N),
	assuming all values in the N dimension are identical for each B.
	"""
	# Get the original shape
	B, _, N = tensor.shape
	
	# Extract the unique values for each batch item. Shape will be (B,)
	# We only need one value from the N dimension, e.g., at index 0
	b_values = tensor[:, 0, 0]
	
	# Apply the wrap-around logic. Shape (B,)
	wrapped_values = (b_values + 0.5) % 1.0
	
	# Clamp the results. Shape (B,)
	clamped_values = torch.clamp(wrapped_values, min=lower, max=upper)
	
	# Reshape to (B, 1, 1) to prepare for broadcasting
	output_values = clamped_values.view(B, 1, 1)
	
	# Expand back to the original (B, 1, N) shape
	output_tensor = output_values.expand(B, 1, N)
	
	return output_tensor

def _test_rec_and_gen(batch) -> list:
	gen_batch = {
		'cond_image': batch['cond_image'],
		'encoder_hidden_states': apply_clamped_wrap_to_tensor(
			batch['encoder_hidden_states']
		)
	}

	batches = {}
	batches['rec'] = {'input': batch, 'ef': batch['encoder_hidden_states'][:, 0, 0].tolist()}
	batches['gen'] = {'input': gen_batch, 'ef': gen_batch['encoder_hidden_states'][:, 0, 0].tolist()}
	return batches



def sample_latents_from_model(model, dl_list, run_cfg, epoch, step, device, samples_dir, out_name=None, debug=False, kwargs={}):
	"""Evaluate model on list of DataLoaders and save latent videos and metadata.csv."""
	# timing start
	_t0 = time.perf_counter()

	model.eval()
	Cc, T, H, W = dl_list[0].dataset[0]['cond_image'].shape
	C = int(run_cfg.vae.resolution.split('f')[0])
	data_shape = (C, T, H, W)

	model_sample_kwargs = kwargs.get('model_sample_kwargs', run_cfg.sample.get('model_sample_kwargs', {}))

	out_name = out_name or f"sample-epoch={epoch}-step={step}"
	samples_dir = Path(samples_dir) / out_name
	samples_dir.mkdir(parents=True, exist_ok=True)

	metadata_df_rows = []
	for dl in tqdm(dl_list, total=len(dl_list), desc=f"Sampling latents: Epoch {epoch}"):
		unscale_latents = UnscaleLatents(run_cfg, dl.dataset)
		data_shape = tuple(dl.dataset[0]['x'].shape)
		nmf = dl.dataset.kwargs['n_missing_frames']
		nmf = f"{str(int(100*nmf))}p" if not isinstance(nmf, str) else nmf # 0.75 -> '75p', 'max' -> 'max'

		for batch in tqdm(dl, desc="Batches"):
			reference_batch, input_batch = batch
			batch_size = input_batch['cond_image'].shape[0]

			batches = _test_rec_and_gen(input_batch)
			for rec_or_gen, sub_batch in batches.items():
				sub_batch['input'] = {k: v.to(device) for k, v in sub_batch['input'].items()}

				sample_videos = model.sample(
					**sub_batch['input'],
					batch_size=batch_size,
					data_shape=data_shape,
					**model_sample_kwargs
				)  # (B, C, T, H, W)
				sample_videos = sample_videos.detach().cpu()
				sample_videos = unscale_latents(sample_videos)
				
				for j, (ef, video) in enumerate(zip(sub_batch['ef'], sample_videos)):
					ef  = round( int(100 * ef), 2)
					real_video_name = reference_batch['video_name'][j]
					video_name = f"{real_video_name}_ef{ef}_nmf{nmf}"
					
					obs_mask = reference_batch['observed_mask'][j].tolist()
					not_pad_mask = reference_batch['not_pad_mask'][j].tolist()
					# Save metadata
					metadata_df_rows.append({
						'video_name': video_name,
						'n_missing_frames': nmf,
						'EF': ef,
						'rec_or_gen': rec_or_gen,
						'original_real_video_name': real_video_name,
						'observed_mask': obs_mask,
						'not_pad_mask': not_pad_mask
					})

					torch.save(
						obj={'video': video},
						f=samples_dir / f"{video_name}.pt"
					)
			if debug:
				break
					

	df = pd.DataFrame(metadata_df_rows)
	df.to_csv(samples_dir / 'metadata.csv', index=False)

	# timing end + summary
	_elapsed = time.perf_counter() - _t0
	_saved = len(metadata_df_rows)
	throughput = (_saved / _elapsed) if _elapsed > 0 else float('nan')
	print(
		f"Sampling summary: epoch={epoch}, step={step}, dls={len(dl_list)}, "
		f"videos_saved={_saved}, time={_elapsed / 60:.2f}m, rate={throughput:.2f} vids/s, out='{samples_dir}'"
	)
      
	return samples_dir
