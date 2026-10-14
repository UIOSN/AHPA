import torch

from PIL import Image
import numpy as np


def center_crop_arr(image_arr, image_size):
    """
    Center cropping implementation from ADM.
    https://github.com/openai/guided-diffusion/blob/8fb3ad9197f16bbc40620447b2742e13458d2831/guided_diffusion/image_datasets.py#L126
    """
    pil_image = Image.fromarray(image_arr)
    while min(*pil_image.size) >= 2 * image_size:
        pil_image = pil_image.resize(
            tuple(x // 2 for x in pil_image.size), resample=Image.BOX
        )

    scale = image_size / min(*pil_image.size)
    pil_image = pil_image.resize(
        tuple(round(x * scale) for x in pil_image.size), resample=Image.BICUBIC
    )

    arr = np.array(pil_image)
    crop_y = (arr.shape[0] - image_size) // 2
    crop_x = (arr.shape[1] - image_size) // 2
    return arr[crop_y: crop_y + image_size, crop_x: crop_x + image_size]

#################################################################################
# spatial normalization
#################################################################################
ALL_SPNORM_METHODS = ["none", "zscore"]


def spatial_zscore(feat: torch.Tensor, alpha: float = 1.0, eps: float = 1e-6) -> torch.Tensor:
    """
    Z-score normalization along spatial dimension.

    Args:
        feat: (B, T, D) patch tokens
        alpha: scaling factor for mean subtraction (default 1.0)
        eps: small constant for numerical stability

    Returns:
        Normalized features (B, T, D)
    """
    mean = feat.mean(dim=1, keepdim=True)
    std = feat.std(dim=1, keepdim=True)
    return (feat - alpha * mean) / (std + eps)


class SpatialNormalization:
    """Apply the spatial normalization used by AHPA or the identity used by SRA2."""
    def __init__(self, method: str, *, eps: float = 1e-6):
        assert method in ALL_SPNORM_METHODS, f"Invalid method: {method}. Must be one of {ALL_SPNORM_METHODS}"
        self.method = method
        self.eps = eps

    def __call__(self, feat: torch.Tensor, **kwargs) -> torch.Tensor:
        if self.method == "none":
            return feat
        alpha = kwargs.get('zscore_alpha', 1.0)
        return spatial_zscore(feat, alpha=alpha, eps=self.eps)
