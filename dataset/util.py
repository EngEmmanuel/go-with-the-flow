import torch 


def make_sampling_collate(n: int):
    def _collate(batch):
        '''
        Returns a batch of size n by repeating the single example in the input batch.
        EF is varied but the cond_image is kept the same.
        '''
        assert len(batch) == 1, "Sampling DataLoader must use batch_size=1"
        sample = batch[0]

        video_name = sample['video_name']
        cond_image = sample['cond_image']  # (C, T, H, W)
        ehs = sample['encoder_hidden_states']  # (1, E)

        # Repeat data n times along batch dimension
        cond_rep = cond_image.unsqueeze(0).repeat(n, 1, 1, 1, 1)      # (n, C, T, H, W)

        ehs_dim = ehs.shape[-1]
        ef_orig = ehs.flatten()[0] # scalar EF value used to create the original EHS

        if n > 1:
            grid = torch.linspace(0.2, 0.8, steps=n - 1, device=ehs.device, dtype=ehs.dtype)
            ef_values = torch.cat([ef_orig.view(1), grid], dim=0)
        else:
            ef_values = ef_orig.view(1)

        ehs_rep = ef_values.view(n, 1).expand(n, ehs_dim)       # (n, E)

        reference_batch = {'cond_image': cond_image, 'ef_values': ef_values, 'video_name': video_name}
        repeated_batch = {'cond_image': cond_rep, 'encoder_hidden_states': ehs_rep}
        return reference_batch, repeated_batch
    return _collate