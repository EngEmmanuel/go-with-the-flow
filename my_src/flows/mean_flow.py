import torch
import torch.nn as nn
from torch.nn import Module
from torch.nn.attention import sdpa_kernel, SDPBackend
from torch import Tensor, tensor, stack, ones, zeros, zeros_like
from torchdiffeq import odeint
from einops import rearrange, repeat, reduce
from my_src.custom_loss import MaskedMeanFlowLoss
from torch.func import jvp
# Code adapted from https://github.com/lucidrains/rectified-flow-pytorch/blob/main/rectified_flow_pytorch/mean_flow.py

from random import random
from contextlib import nullcontext

def identity(t):
    return t

def exists(v):
    return v is not None

def default(v, d):
    return v if exists(v) else d

# tensor helpers
def append_dims(t, ndims):
    shape = t.shape
    return t.reshape(*shape, *((1,) * ndims))

class MeanFlow(Module):
    def __init__(
        self,
        model: Module,
        data_shape = None,
        use_adaptive_loss_weight = True,
        adaptive_loss_weight_p = 0.5, # 0.5 is approximately pseudo huber loss
        use_logit_normal_sampler = True,
        logit_normal_mean = -0.4,
        logit_normal_std = 1.,
        prob_default_flow_obj = 0.5,
        add_recon_loss = False,
        recon_loss_weight = 1.,
        noise_std_dev = 1.,
        eps = 1e-3
    ):
        super().__init__()
        self.model = model 
        self.data_shape = data_shape

        # weight loss related

        self.use_adaptive_loss_weight = use_adaptive_loss_weight
        self.adaptive_loss_weight_p = adaptive_loss_weight_p
        self.eps = eps

        # time sampler

        self.use_logit_normal_sampler = use_logit_normal_sampler
        self.logit_normal_mean = logit_normal_mean
        self.logit_normal_std = logit_normal_std

        # they do 25-50% normal flow matching obj

        assert 0. <= prob_default_flow_obj <= 1.
        self.prob_default_flow_obj = prob_default_flow_obj

        # recon loss

        self.add_recon_loss = add_recon_loss and recon_loss_weight > 0
        self.recon_loss_weight = recon_loss_weight

        self.noise_std_dev = noise_std_dev

        self.register_buffer('dummy', tensor(0), persistent = False)

        # loss function

        self.loss_fn = MaskedMeanFlowLoss(
            use_adaptive_loss_weight=use_adaptive_loss_weight,
            adaptive_loss_weight_p=adaptive_loss_weight_p,
            eps=eps,
            add_recon_loss=add_recon_loss,
            recon_loss_weight=recon_loss_weight
        )

    @property
    def device(self):
        return self.dummy.device

    def sample_times(self, batch):
        shape, device = (batch,), self.device

        if not self.use_logit_normal_sampler:
            return torch.rand(shape, device = device)

        mean = torch.full(shape, self.logit_normal_mean, device = device)
        std = torch.full(shape, self.logit_normal_std, device = device)
        return torch.normal(mean, std).sigmoid()
    

    @torch.no_grad()
    def slow_sample(
        self,
        encoder_hidden_states: Tensor,
        cond_image,
        steps = 16,
        batch_size = 1,
        noise = None,
        data_shape = None,
        **kwargs
    ):
        assert steps >= 1

        device = self.device

        if not exists(noise):
            noise = self.get_noise(batch_size, data_shape = data_shape)

        times = torch.linspace(1., 0., steps + 1, device = device)[:-1]
        delta = 1. / steps

        denoised = noise

        delta_time = zeros(batch_size, device = device)

        for time in times:
            time = time.expand(batch_size)
            pred_flow = self.model(
                x=denoised,
                timestep=time,
                encoder_hidden_states=encoder_hidden_states,
                cond_image=cond_image,
                cond_t=delta_time
            )

            denoised = denoised - delta * pred_flow

        return denoised


    def sample(
        self,
        encoder_hidden_states: Tensor,
        cond_image,
        batch_size = None,
        data_shape = None,
        requires_grad = False,
        noise = None,
        steps = 1
    ):
        data_shape = default(data_shape, self.data_shape)

        assert exists(data_shape), 'shape of the data must be passed in, or set at init or during training'

        batch_size = default(batch_size, 1)

        # maybe slow sample

        assert steps >= 1

        if steps > 1:
            return self.slow_sample(
                encoder_hidden_states = encoder_hidden_states,
                batch_size = batch_size,
                data_shape = data_shape,
                noise = noise,
                cond_image = cond_image,
                steps = steps
            )

        device = next(self.model.parameters()).device

        context = nullcontext if requires_grad else torch.no_grad

        # Algorithm 2

        if not exists(noise):
            noise = self.get_noise(batch_size, data_shape = data_shape)

        with context():
            pred = self.model(
                x=noise,
                timestep=ones(batch_size, device = device),
                encoder_hidden_states=encoder_hidden_states,
                cond_image=cond_image,
                cond_t=ones(batch_size, device = device),
            )
            denoised = noise - pred

        return denoised


    def get_noise(
        self,
        batch_size = 1,
        data_shape = None
    ):
        device = self.device

        data_shape = default(data_shape, self.data_shape)
        assert exists(data_shape), 'shape of the data must be passed in, or set at init or during training'

        noise = torch.randn((batch_size, *data_shape), device = device) * self.noise_std_dev
        return noise

    def forward(
        self,
        x,
        encoder_hidden_states: torch.Tensor,
        cond_image,
        noise: Tensor | None = None,
        return_loss_breakdown = True,
        loss_mask=None
    ):

        # shapes and variables

        shape, ndim = x.shape, x.ndim

        prob_time_end_start_same = self.prob_default_flow_obj

        self.data_shape = default(self.data_shape, shape[1:]) # store last data shape for inference
        batch, device = shape[0], x.device

        # flow logic

        times = self.sample_times(batch)

        # some set prob of the time, normal flow matching training (times == start integral times)

        normal_flow_match_obj = prob_time_end_start_same > 0. and random() < prob_time_end_start_same

        if normal_flow_match_obj:
            integral_start_times = times
        else:
            # section 4.3 logic for choosing r and t
            second_times = self.sample_times(batch)
            sorted_times = stack((times, second_times), dim = -1).sort(dim = -1)
            integral_start_times, times = sorted_times.values.unbind(dim = -1)

        # derive flows

        if not exists(noise):
            noise = torch.randn_like(x) * self.noise_std_dev

        flow = noise - x # flow is the velocity from data to noise, also what the model is trained to predict

        padded_times = append_dims(times, ndim - 1)
        noised = x.lerp(noise, padded_times) # noise the data with random amounts of noise (time) - from data -> noise from 0. to 1.

        # they condition the network on the delta time instead

        delta_times = times - integral_start_times
        padded_delta_times = append_dims(delta_times, ndim - 1)

        # model forward with maybe jvp
        # input order matters as positional args are used in jvp. model forward signature:
        # (x, timestep, encoder_hidden_states, cond_image, cond_t)
        # (input, d(input)/dt)
        pairs = [
            (noised,                flow),                                            # dz/dt = v(z, t)
            (times,                 ones(batch, device=device, dtype=times.dtype)),   # dt/dt = 1
            (encoder_hidden_states, zeros_like(encoder_hidden_states)),               # constant
            (cond_image,            zeros_like(cond_image)),                          # constant
            (delta_times,           ones(batch, device=device, dtype=times.dtype)),   # d(t-r)/dt = 1  → holds r fixed
        ]

        # unzip for jvp
        inputs, tangents = map(tuple, zip(*pairs))
        if normal_flow_match_obj:
            # Normal flow matching without jvp 25-50% of the time

            pred, rate_avg_vel_change = (
                self.model(*inputs),
                tensor(0., device = device)
            )
        else:
            # Algorithm 1
            with sdpa_kernel(SDPBackend.MATH):
                pred, rate_avg_vel_change = jvp(
                    self.model,
                    inputs,
                    tangents
                )

        # the new proposed target

        integral = padded_delta_times * rate_avg_vel_change.detach()

        target = flow - integral

        out = self.loss_fn(
            pred=pred, 
            flow=flow,
            integral=integral,
            loss_mask=loss_mask,
            noised_data=noised,
            data=x,
            padded_times=padded_times
        )

        if return_loss_breakdown:
            return out
        else:
            return out['loss']
        
