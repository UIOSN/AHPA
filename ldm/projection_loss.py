import torch.nn.functional as F


class CosineProjectionLoss:
    """Cosine alignment loss for token tensors shaped [B, T, D]."""

    def __call__(self, target, projected, _original=None):
        if target.ndim != 3 or projected.ndim != 3:
            raise ValueError(
                f"Expected [B, T, D] tensors, got {target.shape} and {projected.shape}"
            )
        if target.shape != projected.shape:
            raise ValueError(
                f"Feature shape mismatch: {target.shape} != {projected.shape}"
            )
        target = F.normalize(target, dim=-1)
        projected = F.normalize(projected, dim=-1)
        return -(target * projected).sum(dim=-1).mean()


def make_projection_loss(name, **_kwargs):
    if name != "cosine":
        raise ValueError("Only the cosine projection loss is supported")
    return CosineProjectionLoss()
