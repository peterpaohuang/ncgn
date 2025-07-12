import torch
from diffusers.utils.torch_utils import randn_tensor

def diffusion_sampling(diffuser, model_output, t, prev_t, sample):

    if model_output.shape[1] == sample.shape[1] * 2 and diffuser.variance_type in ["learned", "learned_range"]:
        model_output, predicted_variance = torch.split(model_output, sample.shape[1], dim=1)
    else:
        predicted_variance = None

    # 1. compute alphas, betas
    alpha_prod_t = diffuser.alphas_cumprod[t]
    alpha_prod_t_prev = diffuser.alphas_cumprod[prev_t] if prev_t >= 0 else diffuser.one
    beta_prod_t = 1 - alpha_prod_t
    beta_prod_t_prev = 1 - alpha_prod_t_prev
    current_alpha_t = alpha_prod_t / alpha_prod_t_prev
    current_beta_t = 1 - current_alpha_t

    # 2. compute predicted original sample from predicted noise also called
    # "predicted x_0" of formula (15) from https://arxiv.org/pdf/2006.11239.pdf
    if diffuser.config.prediction_type == "epsilon":
        pred_original_sample = (sample - beta_prod_t ** (0.5) * model_output) / alpha_prod_t ** (0.5)
    elif diffuser.config.prediction_type == "sample":
        pred_original_sample = model_output
    elif diffuser.config.prediction_type == "v_prediction":
        pred_original_sample = (alpha_prod_t**0.5) * sample - (beta_prod_t**0.5) * model_output
    else:
        raise ValueError(
            f"prediction_type given as {diffuser.config.prediction_type} must be one of `epsilon`, `sample` or"
            " `v_prediction`  for the DDPMScheduler."
        )

    # 3. Clip or threshold "predicted x_0"
    if diffuser.config.thresholding:
        pred_original_sample = diffuser._threshold_sample(pred_original_sample)
    elif diffuser.config.clip_sample:
        pred_original_sample = pred_original_sample.clamp(
            -diffuser.config.clip_sample_range, diffuser.config.clip_sample_range
        )

    # 4. Compute coefficients for pred_original_sample x_0 and current sample x_t
    # See formula (7) from https://arxiv.org/pdf/2006.11239.pdf
    pred_original_sample_coeff = (alpha_prod_t_prev ** (0.5) * current_beta_t) / beta_prod_t
    current_sample_coeff = current_alpha_t ** (0.5) * beta_prod_t_prev / beta_prod_t

    # 5. Compute predicted previous sample µ_t
    # See formula (7) from https://arxiv.org/pdf/2006.11239.pdf
    pred_prev_sample = pred_original_sample_coeff * pred_original_sample + current_sample_coeff * sample

    # 6. Add noise
    variance = 0
    if t > 0:
        device = model_output.device
        variance_noise = randn_tensor(
            model_output.shape, generator=None, device=device, dtype=model_output.dtype
        )
        if diffuser.variance_type == "fixed_small_log":
            variance = diffuser._get_variance(t, predicted_variance=predicted_variance) * variance_noise
        elif diffuser.variance_type == "learned_range":
            variance = diffuser._get_variance(t, predicted_variance=predicted_variance)
            variance = torch.exp(0.5 * variance) * variance_noise
        else:
            variance = (diffuser._get_variance(t, predicted_variance=predicted_variance) ** 0.5) * variance_noise

    pred_prev_sample = pred_prev_sample + variance

    return pred_prev_sample