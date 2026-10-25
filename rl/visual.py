"""Phase pooling and fixed-projection visual latents for DA-CVQ."""
from collections.abc import Sequence
import torch


@torch.no_grad()
def project_visual_latent(
    phase_patch_states: Sequence[torch.Tensor],
    projection: torch.Tensor,
    *,
    phase_masks: Sequence[torch.Tensor],
) -> torch.Tensor:
    """Mean-pool, project, concatenate a/p/v features, and L2-normalize.

    Three patch tensors have shape [patches, input_dim]; binary masks select real
    patches. The caller supplies a fixed column-orthonormal projection of shape
    [input_dim, projected_dim], shared across phases and inputs. The result is
    [3 * projected_dim]. Features and projection are frozen numerical inputs.
    """
    if len(phase_patch_states) != 3 or len(phase_masks) != 3:
        raise ValueError("three phases in a/p/v order and their masks are required")
    if (projection.ndim != 2 or min(projection.shape) <= 0
            or not projection.is_floating_point() or not torch.isfinite(projection).all()):
        raise ValueError("projection must be a finite floating-point matrix")
    parts = []
    for states, mask in zip(phase_patch_states, phase_masks):
        if states.ndim != 2 or states.shape[1] != projection.shape[0]:
            raise ValueError("phase patch dimension must match projection input dimension")
        if states.device != projection.device or states.dtype != projection.dtype:
            raise ValueError("patch states and projection must share device and dtype")
        if mask.shape != states.shape[:1] or mask.device != states.device:
            raise ValueError("phase mask must select patches on the same device")
        if not torch.all((mask == 0) | (mask == 1)):
            raise ValueError("phase mask must be binary")
        selected = states[mask.bool()]
        if selected.shape[0] == 0 or not torch.isfinite(selected).all():
            raise ValueError("each phase needs finite retained patch states")
        parts.append(selected.mean(dim=0) @ projection)
    vector = torch.cat(parts)
    scale = vector.abs().max()
    if not torch.isfinite(vector).all() or scale <= 0:
        raise ValueError("projected visual latent must be finite and non-zero")
    vector = vector / scale
    return vector / torch.linalg.vector_norm(vector)
