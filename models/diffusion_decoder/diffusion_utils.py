import math
import torch
import numpy as np

def get_beta_schedule(beta_schedule, *, beta_start, beta_end,
                      num_diffusion_timesteps):

    print(f"The {beta_schedule} type of noise schedule to be used.")

    def sigmoid(x):
        return 1 / (np.exp(-x) + 1)
    if beta_schedule == "quad":
        betas = (np.linspace(
            beta_start**0.5,
            beta_end**0.5,
            num_diffusion_timesteps,
            dtype=np.float64,
        )**2)
    elif beta_schedule == "linear":
        betas = np.linspace(beta_start,
                            beta_end,
                            num_diffusion_timesteps,
                            dtype=np.float64)
    elif beta_schedule == "const":
        betas = beta_end * np.ones(num_diffusion_timesteps, dtype=np.float64)
    elif beta_schedule == "jsd":  # 1/T, 1/(T-1), 1/(T-2), ..., 1
        betas = 1.0 / np.linspace(num_diffusion_timesteps,
                                  1,
                                  num_diffusion_timesteps,
                                  dtype=np.float64)
    elif beta_schedule == "sigmoid":
        betas = np.linspace(-6, 6, num_diffusion_timesteps)
        betas = sigmoid(betas) * (beta_end - beta_start) + beta_start
    elif beta_schedule == 'cosine':
        step = num_diffusion_timesteps + 1
        s = 0.008
        x = np.linspace(0, step, step)
        alphas_cumprod = np.cos(((x / step) + s) / (1 + s) * np.pi * 0.5) ** 2
        alphas_cumprod = alphas_cumprod / alphas_cumprod[0]
        betas = 1 - (alphas_cumprod[1:] / alphas_cumprod[:-1])
        betas = np.clip(betas, a_min=0, a_max=0.999)
    else:
        raise NotImplementedError(beta_schedule)
    assert betas.shape == (num_diffusion_timesteps, )
    return betas


def noise_estimation_loss(model,
                          imgs: torch.Tensor,
                          x0: torch.Tensor,
                          t: torch.LongTensor,
                          e: torch.Tensor,
                          b: torch.Tensor,
                          keepdim=False):

    a = (1 - b).cumprod(dim=0).index_select(0, t).view(-1, 1, 1, 1)
    x = x0 * a.sqrt() + e * (1.0 - a).sqrt()

    data = {"img": imgs, "input": x}
    output = model(data, t.float())

    if keepdim:
        return (e - output).square().sum(dim=(1, 2, 3))
    else:
        return (e - output).square().sum(dim=(1, 2, 3)).mean(dim=0)

def to_torch(tensor):
    return torch.tensor(tensor, dtype=torch.float32)