import math
import numpy as np
from tqdm import tqdm
import torch
import torch.nn as nn
import torch.nn.functional as F

from .model_utils import build_denoiser, build_cond_encoder



class ActDiff(nn.Module):
    def __init__(self, args):
        super().__init__()

        self.seed = args.seed
        self.latent_size = args.latent_size
        self.latent_channel = args.latent_channel

        self.denoiser_model = build_denoiser(args)
        '''
        if args.image_cond:
            self.image_encoder = build_cond_encoder(args)
        '''
        self.train_mode = args.train_mode
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

    def forward(self, x, t, noise, cond_act=None, cond_extra=None, label=None, mask=None):
        # cond_emb = self.text_encoder(text)
        loss = self.p_losses(x_start=x, t=t, cond_act=cond_act, label=label, noise=noise)
        return loss

    def p_losses(self, x_start, t, mask, cond_act=None, cond_extra=None, label=None, noise=None):
        if noise is None:
            noise = torch.randn_like(x_start)
        x_noisy = self.q_sample(x_start=x_start, t=t, noise=noise)


        if cond_extra is not None:
            extra_embed = self.cond_encoder(cond_extra).to(x_start.device)
        else:
            extra_embed = None

        if self.train_mode == 'noise':
            predicted_noise = self.denoiser_model(x=x_noisy, t=t, attn_mask=mask, cond_act=cond_act, label=label)
            if self.loss_type == 'l1':
                loss = F.l1_loss(noise, predicted_noise)
            elif self.loss_type == 'l2':
                loss = F.mse_loss(noise, predicted_noise)
            elif self.loss_type == "huber":
                loss = F.smooth_l1_loss(noise, predicted_noise)
            else:
                raise NotImplementedError()
        elif self.train_mode == 'x_start':
            pred_x_start = self.denoiser_model(x=x_noisy, t=t, attn_mask=mask, cond_act=cond_act, label=label)
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
    def ddpm_sample(self, x, cond_act, t, label, t_index, mask=None):
        p_mean_out = self.p_mean_variance(x=x, cond_act=cond_act, label=label, t=t)
        if t_index == 0:
            return {'sample': p_mean_out['mean'], 'pred_x_start': p_mean_out['x_start']}
        else:
            posterior_variance_t = _extract_into_tensor(self.posterior_variance, t, x.shape)
            noise = torch.randn_like(x)
            return {'sample': p_mean_out['mean'] + torch.sqrt(posterior_variance_t) * noise,
                    'pred_x_start': p_mean_out['x_start']}

    def ddpm_sample_loop(self, cond_act, shape, label, cond_extra=None, mask=None):
        device = next(self.denoiser_model.parameters()).device

        b = shape[0]
        # start from pure noise (for each example in the batch)
        latent = torch.randn(shape, device=device)
        latents = []

        if cond_extra is not None:
            extra_embed = self.cond_encoder(cond_extra)
        else:
            extra_embed = None

        for i in tqdm(reversed(range(0, self.num_timesteps)), desc='sampling loop time step', total=self.num_timesteps):
            out = self.ddpm_sample(x=latent, cond_act=cond_act, mask=mask, label=label,
                                   t=torch.full((b,), i, device=device, dtype=torch.long), t_index=i)
            latent = out['sample']
            latents.append(latent.cpu().numpy())
        return latent, latents

    def ddim_sample(self, x, cond_act, label, t, mask=None, eta=0.0):
        p_mean_out = self.p_mean_variance(x=x, cond_act=cond_act, label=label, t=t)
        '''
        if self.constraint is True:
            p_mean_out = self.condition_score(x=x, t=t, p_mean_var=p_mean_out, pose_cond=pose_cond,
                                              keyframe_mask=keyframe_mask, ref_motion=ref_motion, keyframe=keyframe)
        '''
        pred_x_start = p_mean_out['x_start']
        eps = self.predict_eps_from_x_start(x_t=x, t=t, x_start=pred_x_start)
        alpha_bar = _extract_into_tensor(self.alphas_cumprod, t, x.shape)
        alpha_bar_prev = _extract_into_tensor(self.alphas_cumprod_prev, t, x.shape)
        sigma = (
                eta
                * torch.sqrt((1 - alpha_bar_prev) / (1 - alpha_bar))
                * torch.sqrt(1 - alpha_bar / alpha_bar_prev)
        )
        # Equation 12.
        noise = torch.randn_like(x)
        model_mean = (
                pred_x_start * torch.sqrt(alpha_bar_prev)
                + torch.sqrt(1 - alpha_bar_prev - sigma ** 2) * eps
        )
        nonzero_mask = (
            (t != 0).float().view(-1, *([1] * (len(x.shape) - 1)))
        )  # no noise when t == 0

        return {'sample': model_mean + nonzero_mask * sigma * noise, 'pred_x_start': pred_x_start}

    def ddim_sample_loop(self, cond_act, shape, label, cond_extra=None, mask=None, eta=0.0):
        device = next(self.denoiser_model.parameters()).device

        b = shape[0]
        # start from pure noise (for each example in the batch)
        latent = torch.randn(shape, device=device)
        latents = []

        if cond_extra is not None:
            extra_embed = self.cond_encoder(cond_extra)
        else:
            extra_embed = None

        for i in tqdm(reversed(range(0, self.num_timesteps)), desc='sampling loop time step', total=self.num_timesteps):
            out = self.ddim_sample(x=latent, cond_act=cond_act, mask=mask, eta=eta, label=label,
                                   t=torch.full((b,), i, device=device, dtype=torch.long))
            latent = out['sample']
            latents.append(latent.cpu().numpy())
        return latent, latents

    def ddim_reverse_sample(self, x, t, cond_act, label, eta=0.0):
        assert eta == 0.0, "Reverse ODE only for deterministic path"
        p_mean_out = self.p_mean_variance(x=x, condtion=cond_act, t=t, label=label)

        # Usually our model outputs epsilon, but we re-derive it
        # in case we used x_start or x_prev prediction.
        eps = (
                      _extract_into_tensor(self.sqrt_recip_alphas_cumprod, t, x.shape) * x
                      - p_mean_out["pred_xstart"]
              ) / _extract_into_tensor(self.sqrt_recipm1_alphas_cumprod, t, x.shape)
        alpha_bar_next = _extract_into_tensor(self.alphas_cumprod_next, t, x.shape)

        # Equation 12. reversed
        mean_pred = (p_mean_out["pred_xstart"] * torch.sqrt(alpha_bar_next) + torch.sqrt(1 - alpha_bar_next) * eps)
        return {"sample": mean_pred, "pred_xstart": p_mean_out["pred_xstart"]}

    def ddim_reverse_loop(self, x, start, end, cond_input, label, eta=0.0):
        device = next(self.denoiser_model.parameters()).device
        self.motion_vae.set_input(x)
        zg = self.motion_vae.encode_global
        zl = self.motion_vae.encode_local
        latent = torch.cat((zl, zg), dim=1)
        x_shape = list(latent.shape)

        if cond_input is not None:
            cond_emb = self.cond_encoder(cond_input)
        else:
            cond_emb = None

        b = x_shape[0]

        for i in tqdm(range(start, end), desc='sampling loop time step', total=(end - start)):
            if i >= end - 1: continue
            out = self.ddim_reverse_sample(x=latent, condition=cond_emb, label=label,
                                           t=torch.full((b,), i, device=device, dtype=torch.long),  eta=eta)
            latent = out['sample']

        return latent

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
    def p_mean_variance(self, x, condtion, t, label=None, mask=None):
        model_output = self.denoiser_model(x=x, t=t, condtion=condtion, attn_mask=mask)
        if self.mean_type == 'x_start':
            x_start = model_output
            mean, variance = self.q_posterior_mean_variance(x_start, x, t)
        elif self.mean_type == 'noise':
            x_start = self.predict_x_start_from_noise(noise=model_output, x_t=x, t=t)
            mean, variance = self.q_posterior_mean_variance(x_start, x, t)
        elif self.mean_type == 'x_prev':
            x_start = self.predict_x_start_from_x_prev(x_prev=model_output, x_t=x, t=t)
            mean, variance = self.q_posterior_mean_variance(x_start, x, t)
        else:
            raise NotImplementedError()
        return {'mean': mean, 'variance': variance, 'x_start': x_start}

    @torch.no_grad()
    def denoise_result(self, x_start, condition, t, mask, noise=None):
        x_noisy = self.q_sample(x_start=x_start, t=t, noise=noise)
        condi_emb = self.text_encoder(condition)
        model_result = self.denoiser_model(x=x_noisy, t=t, attn_mask=mask, text_emb=condi_emb)
        if self.train_mode == 'x_start':
            pred_x_start = model_result
        elif self.train_mode == 'noise':
            pred_x_start = self.predict_x_start_from_noise(x_t=x_noisy, t=t, noise=model_result)
        else:
            raise NotImplementedError()
        return pred_x_start

    def condition_mean(self, p_mean_var, x, t, model_kwargs=None):
        """
        Compute the mean for the previous step, given a function cond_fn that
        computes the gradient of a conditional log probability with respect to
        x. In particular, cond_fn computes grad(log(p(y|x))), and we want to
        condition on y.

        This uses the conditioning strategy from Sohl-Dickstein et al. (2015).
        """
        # print(f'condition_mean_step{t}')
        x_start = p_mean_var['x_start']

        gradient = self.cond_fn()
        new_mean = (
                p_mean_var["mean"].float() + p_mean_var["variance"] * gradient.float()
        )
        return new_mean

    def condition_score(self, p_mean_var, x, t, condition, label, model_kwargs=None):
        """
        Compute what the p_mean_variance output would have been, should the
        model's score function be conditioned by cond_fn.

        See condition_mean() for details on cond_fn.

        Unlike condition_mean(), this instead uses the conditioning strategy
        from Song et al (2020).
        """
        x_start = p_mean_var['x_start']

        alpha_bar = _extract_into_tensor(self.alphas_cumprod, t, x.shape)

        eps = self.predict_eps_from_x_start(x, t, x_start)
        #  print(f'scaletimesteps{self._scale_timesteps(t)}')

        eps = eps - (1 - alpha_bar).sqrt() * self.cond_fn()

        out = p_mean_var.copy()
        out["x_start"] = self.predict_x_start_from_eps(x, t, eps)
        out["mean"], _ = self.q_posterior_mean_variance(
            x_start=out["x_start"], x_t=x, t=t
        )
        return out

    def predict_x_start_from_noise(self, x_t, t, noise):
        return (_extract_into_tensor(self.sqrt_recip_alphas_cumprod, t, x_t.shape) * x_t
                - _extract_into_tensor(self.sqrt_recipm1_alphas_cumprod, t, x_t.shape) * noise)

    def predict_x_start_from_x_prev(self, x_t, t, x_prev):
        assert x_t.shape == x_prev.shape
        return (  # (xprev - coef2*x_t) / coef1
                _extract_into_tensor(1.0 / self.posterior_mean_coef1, t, x_t.shape) * x_prev
                - _extract_into_tensor(
            self.posterior_mean_coef2 / self.posterior_mean_coef1, t, x_t.shape
        )
                * x_t
        )

    def predict_eps_from_x_start(self, x_t, t, x_start):
        return (
                _extract_into_tensor(self.sqrt_recip_alphas_cumprod, t, x_t.shape) * x_t
                - x_start
        ) / _extract_into_tensor(self.sqrt_recipm1_alphas_cumprod, t, x_t.shape)

    def predict_x_start_from_eps(self, x_t, t, eps):
        assert x_t.shape == eps.shape
        return (
                _extract_into_tensor(self.sqrt_recip_alphas_cumprod, t, x_t.shape) * x_t
                - _extract_into_tensor(self.sqrt_recipm1_alphas_cumprod, t, x_t.shape) * eps
        )


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
