import math
import numpy as np
from tqdm import tqdm
import torch
import torch.nn as nn
import torch.nn.functional as F

from .model_utils import build_denoiser, build_img_encoder

from functools import partial


class ActDiff(nn.Module):
    def __init__(self, args):
        super().__init__()

        self.seed = args.seed
        self.latent_size = args.latent_size
        self.latent_channel = args.latent_channel

        self.denoiser_model = build_denoiser(args)
        self.img_encoder = build_img_encoder(args)

        self.step = 0

        if args.loss_type not in ["l1", "l2"]:
            raise ValueError("__init__() got unknown loss type")
        self.loss_type = args.loss_type
        # self.latent_chanel = args.latent_channel
        self.num_timesteps = args.timesteps

        self.betas = linear_beta_schedule(timesteps=self.num_timesteps)

        # define alphas
        self.alphas = 1. - self.betas
        self.alphas_cumprod = np.cumprod(self.alphas, axis=0)
        self.alphas_cumprod_prev = np.append(1.0, self.alphas_cumprod[:-1])
        self.alphas_cumprod_next = np.append(self.alphas_cumprod[1:], 0.0)

        # calculations for diffusion q(x_t | x_{t-1}) and others
        self.sqrt_recip_alphas = np.sqrt(1.0 / self.alphas)
        self.sqrt_alphas_cumprod = np.sqrt(self.alphas_cumprod)
        self.sqrt_one_minus_alphas_cumprod = np.sqrt(1.0 - self.alphas_cumprod)
        self.log_one_minus_alphas_cumprod = np.log(1.0 - self.alphas_cumprod)
        self.sqrt_recip_alphas_cumprod = np.sqrt(1.0 / self.alphas_cumprod)
        self.sqrt_recipm1_alphas_cumprod = np.sqrt(1.0 / self.alphas_cumprod - 1)

        # calculations for posterior q(x_{t-1} | x_t, x_0)
        self.posterior_variance = (self.betas * (1.0 - self.alphas_cumprod_prev) / (1.0 - self.alphas_cumprod))
        self.posterior_mean_coef1 = (
                self.betas * np.sqrt(self.alphas_cumprod_prev) / (1.0 - self.alphas_cumprod)
        )
        self.posterior_mean_coef2 = (
                (1.0 - self.alphas_cumprod_prev) * np.sqrt(self.alphas) / (1.0 - self.alphas_cumprod)
        )


    def forward(self, x, condition, mask, t, noise, train_mode):

        # text_emb = self.text_encoder(text)
        loss = self.p_losses(x_start=x, t=t, mask=mask, condition=condition, noise=noise, train_mode=train_mode)
        return loss

    def p_losses(self, x_start, t, mask, condition, noise=None, train_mode='noise'):
        if noise is None:
            noise = torch.randn_like(x_start)
        x_noisy = self.q_sample(x_start=x_start, t=t, noise=noise)

        if condition is not None:
            condi_embed = self.text_encoder(condition).to(x_start.device)
        else:
            condi_embed = None

        if train_mode == 'noise':
            predicted_noise = self.denoiser_model(x=x_noisy, t=t, attn_mask=mask, text_emb=condi_embed)
            if self.loss_type == 'l1':
                loss = F.l1_loss(noise, predicted_noise)
            elif self.loss_type == 'l2':
                loss = F.mse_loss(noise, predicted_noise)
            elif self.loss_type == "huber":
                loss = F.smooth_l1_loss(noise, predicted_noise)
            else:
                raise NotImplementedError()
        elif train_mode == 'x_start':
            pred_x_start = self.denoiser_model(x=x_noisy, t=t, attn_mask=mask, text_emb=condi_embed)
            if self.loss_type == 'l1':
                loss = F.l1_loss(x_start, pred_x_start)
            elif self.loss_type == 'l2':
                loss = F.mse_loss(x_start, pred_x_start)
            elif self.loss_type == "huber":
                loss = F.smooth_l1_loss(x_start, pred_x_start)
            else:
                raise NotImplementedError()
        else:
            raise NotImplementedError()

        return loss

    def q_sample(self, x_start, t, noise=None):
        if noise is None:
            noise = torch.randn_like(x_start)

        sqrt_alphas_cumprod_t = _extract_into_tensor(self.sqrt_alphas_cumprod, t, x_start.shape)
        sqrt_one_minus_alphas_cumprod_t = _extract_into_tensor(
            self.sqrt_one_minus_alphas_cumprod, t, x_start.shape
        )

        return sqrt_alphas_cumprod_t * x_start + sqrt_one_minus_alphas_cumprod_t * noise

    @torch.no_grad()
    def p_sample_x_start(self, x, condi_emb, mask, t, t_index):
        model_mean, pred_x_start = self.p_mean_variance(x, condi_emb, mask, t)
        if t_index == 0:
            return {'sample': model_mean, 'pred_x_start': pred_x_start}
        else:
            posterior_variance_t = _extract_into_tensor(self.posterior_variance, t, x.shape)
            noise = torch.randn_like(x)
            return {'sample': model_mean + torch.sqrt(posterior_variance_t) * noise, 'pred_x_start': pred_x_start}

    def p_sample_loop_x_start(self, condition, mask, shape):
        device = next(self.denoiser_model.parameters()).device

        b = shape[0]
        # start from pure noise (for each example in the batch)
        latent = torch.randn(shape, device=device)
        latents = []

        condi_emb = self.text_encoder(condition)

        for i in tqdm(reversed(range(0, self.num_timesteps)), desc='sampling loop time step', total=self.num_timesteps):
            out = self.p_sample_x_start(x=latent, condi_emb=condi_emb, mask=mask,
                                        t=torch.full((b,), i, device=device, dtype=torch.long), t_index=i)
            latent = out['sample']
            latents.append(latent.cpu().numpy())
        return latent, latents

    @torch.no_grad()
    def p_sample_noise(self, x, condi_emb, mask, t, t_index):

        noise = self.denoiser_model(x=x, t=t, attn_mask=mask, text_emb=condi_emb)
        pred_x_start = self.predict_x_start_from_noise(x_t=x, t=t, noise=noise)
        model_mean = self.q_posterior_mean_variance(x_start=pred_x_start, x_t=x, t=t)

        if t_index == 0:
            return {'sample': model_mean, 'pred_x_start': pred_x_start}
        else:
            posterior_variance_t = _extract_into_tensor(self.posterior_variance, t, x.shape)
            noise = torch.randn_like(x)
            return {'sample': model_mean + torch.sqrt(posterior_variance_t) * noise, 'pred_x_start': pred_x_start}

    def p_sample_loop_noise(self, condition, mask, shape):
        device = next(self.denoiser_model.parameters()).device
        b = shape[0]
        # start from pure noise (for each example in the batch)
        latent = torch.randn(shape, device=device)
        latents = []

        condi_emb = self.text_encoder(condition)

        for i in tqdm(reversed(range(0, self.num_timesteps)), desc='sampling loop time step', total=self.num_timesteps):
            out = self.p_sample_noise(x=latent, condi_emb=condi_emb,
                                      t=torch.full((b,), i, device=device, dtype=torch.long),
                                      mask=mask, t_index=i)
            img = out['sample']
            latents.append(img.cpu().numpy())

        return latent, latents

    @torch.no_grad()
    def q_posterior_mean_variance(self, x_start, x_t, t):
        # q(x_{t-1} | x_t, x_0)

        assert x_start.shape == x_t.shape
        posterior_mean = (
                _extract_into_tensor(self.posterior_mean_coef1, t, x_t.shape) * x_start
                + _extract_into_tensor(self.posterior_mean_coef2, t, x_t.shape) * x_t
        )

        return posterior_mean

    @torch.no_grad()
    def p_mean_variance(self, x, condi_embed, mask, t):
        model_output = self.denoiser_model(x=x, t=t, attn_mask=mask, text_emb=condi_embed)
        x_start = model_output
        mean = self.q_posterior_mean_variance(x_start, x, t)
        return mean, x_start

    @torch.no_grad()
    def denoise_result(self, x_start, condition, t, mask, train_mode, noise=None):
        x_noisy = self.q_sample(x_start=x_start, t=t, noise=noise)
        condi_emb = self.text_encoder(condition)
        model_result = self.denoiser_model(x=x_noisy, t=t, attn_mask=mask, text_emb=condi_emb)
        if train_mode == 'x_start':
            pred_x_start = model_result
        elif train_mode == 'noise':
            pred_x_start = self.predict_x_start_from_noise(x_t=x_noisy, t=t, noise=model_result)
        else:
            raise NotImplementedError()
        return pred_x_start

    def predict_x_start_from_noise(self, x_t, t, noise):
        return (_extract_into_tensor(self.sqrt_recip_alphas_cumprod, t, x_t.shape)*x_t
                - _extract_into_tensor(self.sqrt_recipm1_alphas_cumprod, t, x_t.shape)*noise)


def _extract_into_tensor(arr, timesteps, broadcast_shape):
    """
    Extract values from a 1-D numpy array for a batch of indices.

    :param arr: the 1-D numpy array.
    :param timesteps: a tensor of indices into the array to extract.
    :param broadcast_shape: a larger shape of K dimensions with the batch
                            dimension equal to the length of timesteps.
    :return: a tensor of shape [batch_size, 1, ...] where the shape has K dims.
    """
    res = torch.from_numpy(arr).to(device=timesteps.device)[timesteps].float()
    while len(res.shape) < len(broadcast_shape):
        res = res[..., None]
    return res.expand(broadcast_shape)


def linear_beta_schedule(timesteps):
    beta_start = 0.0001
    beta_end = 0.02
    return np.linspace(beta_start, beta_end, timesteps)


def cosine_beta_schedule(timesteps, s=0.008):
    """
    cosine schedule as proposed in https://arxiv.org/abs/2102.09672
    """
    return betas_for_alpha_bar(
        timesteps,
        lambda t: math.cos((t + s) / (1 + s) * math.pi / 2) ** 2,
    )


def betas_for_alpha_bar(num_diffusion_timesteps, alpha_bar, max_beta=0.999):
    """
    Create a beta schedule that discretizes the given alpha_t_bar function,
    which defines the cumulative product of (1-beta) over time from t = [0,1].

    :param num_diffusion_timesteps: the number of betas to produce.
    :param alpha_bar: a lambda that takes an argument t from 0 to 1 and
                      produces the cumulative product of (1-beta) up to that
                      part of the diffusion process.
    :param max_beta: the maximum beta to use; use values lower than 1 to
                     prevent singularities.
    """
    betas = []
    for i in range(num_diffusion_timesteps):
        t1 = i / num_diffusion_timesteps
        t2 = (i + 1) / num_diffusion_timesteps
        betas.append(min(1 - alpha_bar(t2) / alpha_bar(t1), max_beta))
    return np.array(betas)
