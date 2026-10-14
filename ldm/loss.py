import numpy as np
import torch

import projection_loss as pl


def mean_flat(x):
    """Take the mean over all non-batch dimensions."""
    return torch.mean(x, dim=list(range(1, len(x.size()))))


class SILoss:
    """Denoising and representation-alignment loss for AHPA and SRA2."""

    def __init__(
        self,
        prediction="v",
        path_type="linear",
        weighting="uniform",
        proj_coeff="0.5",
        loss_mode="ahpa",
    ):
        if loss_mode not in {"ahpa", "sra2"}:
            raise ValueError("loss_mode must be either 'ahpa' or 'sra2'")

        self.prediction = prediction
        self.weighting = weighting
        self.path_type = path_type
        self.loss_mode = loss_mode

        self.proj_coeff = float(proj_coeff)
        self.projection_loss = pl.make_projection_loss("cosine")

    def interpolant(self, t):
        if self.path_type == "linear":
            return 1 - t, t, -1, 1
        if self.path_type == "cosine":
            alpha_t = torch.cos(t * np.pi / 2)
            sigma_t = torch.sin(t * np.pi / 2)
            d_alpha_t = -np.pi / 2 * torch.sin(t * np.pi / 2)
            d_sigma_t = np.pi / 2 * torch.cos(t * np.pi / 2)
            return alpha_t, sigma_t, d_alpha_t, d_sigma_t
        raise NotImplementedError(f"Unsupported path type: {self.path_type}")

    def __call__(self, model, images, model_kwargs=None, zs=None, time_input=None):
        model_kwargs = model_kwargs or {}
        zs = zs or []

        if time_input is None:
            if self.weighting == "uniform":
                time_input = torch.rand((images.shape[0], 1, 1, 1))
            elif self.weighting == "lognormal":
                sigma = torch.randn((images.shape[0], 1, 1, 1)).exp()
                if self.path_type == "linear":
                    time_input = sigma / (1 + sigma)
                elif self.path_type == "cosine":
                    time_input = 2 / np.pi * torch.atan(sigma)
            else:
                raise ValueError(f"Unsupported weighting: {self.weighting}")

        time_input = time_input.to(device=images.device, dtype=images.dtype)
        noises = torch.randn_like(images)
        alpha_t, sigma_t, d_alpha_t, d_sigma_t = self.interpolant(time_input)
        model_input = alpha_t * images + sigma_t * noises

        if self.prediction != "v":
            raise NotImplementedError(f"Unsupported prediction type: {self.prediction}")
        model_target = d_alpha_t * images + d_sigma_t * noises

        model_output, zs_tilde, zs_tilde_original = model(
            model_input,
            time_input.flatten(),
            **model_kwargs,
        )
        denoising_loss = mean_flat((model_output - model_target) ** 2)

        if len(zs) != len(zs_tilde):
            raise ValueError(
                f"Target/projected feature count mismatch: {len(zs)} != {len(zs_tilde)}"
            )

        projection_value = torch.tensor(0.0, device=images.device, dtype=images.dtype)
        feature_weight = 1.0 / len(zs) if zs else 0.0
        for target, projected, original in zip(zs, zs_tilde, zs_tilde_original):
            projection_value = projection_value + feature_weight * self.projection_loss(
                target, projected, original
            )

        weighted_projection = self.proj_coeff * projection_value
        metrics = {
            "cosine": projection_value.detach().item(),
            "cosine_weighted": weighted_projection.detach().item(),
        }
        return denoising_loss, weighted_projection, metrics
