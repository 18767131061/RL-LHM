"""Frozen Qwen3-VL patch features and fixed visual projection."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from pathlib import Path

import numpy as np
from PIL import Image
import torch

from rl.visual import project_visual_latent
from .huggingface import HuggingFaceBackend


def orthonormal_projection(input_dimension: int, projected_dimension: int, seed: int) -> torch.Tensor:
    """Construct a fixed column-orthonormal matrix using seeded Gaussian QR."""
    if (type(input_dimension) is not int or type(projected_dimension) is not int
            or not 0 < projected_dimension <= input_dimension):
        raise ValueError("Projection dimensions must satisfy 0 < projected <= input")
    if type(seed) is not int:
        raise ValueError("Projection seed must be an integer")
    generator = torch.Generator(device="cpu").manual_seed(seed)
    sample = torch.randn(input_dimension, projected_dimension, generator=generator, dtype=torch.float64)
    q, r = torch.linalg.qr(sample, mode="reduced")
    signs = torch.where(torch.diag(r) < 0, -1., 1.)
    return (q * signs).float()


def load_projection(path: str | Path) -> torch.Tensor:
    """Read a fixed numeric projection matrix from a NumPy file."""
    matrix = np.load(path, allow_pickle=False)
    if matrix.ndim != 2 or not np.issubdtype(matrix.dtype, np.floating):
        raise ValueError("The projection file must contain one floating-point matrix")
    return torch.from_numpy(matrix).float()


@torch.no_grad()
def extract_qwen3_visual_vectors(
    backend: HuggingFaceBackend,
    records: Sequence[Mapping],
    projection: torch.Tensor,
) -> list[torch.Tensor]:
    """Extract last-layer ViT patches, then phase-pool/project/concatenate.

    Qwen3-VL's final patch-merger input is the vision-encoder feature space.
    Image grids delimit real patches; all three phases use the same projection.
    The returned vectors are detached float32 tensors in record order.
    """
    if backend.model.config.model_type != "qwen3_vl":
        raise ValueError("Visual patch extraction supports the Qwen3-VL model type")
    dimension = backend.model.config.vision_config.hidden_size
    if (projection.ndim != 2 or projection.shape[0] != dimension
            or not 0 < projection.shape[1] <= dimension or not torch.isfinite(projection).all()):
        raise ValueError("Projection shape must match the Qwen3-VL vision hidden dimension")
    projection = projection.to(device=backend.model.device, dtype=torch.float32)
    gram = projection.T @ projection
    if not torch.allclose(gram, torch.eye(gram.shape[0], device=gram.device), atol=1e-5, rtol=1e-5):
        raise ValueError("Projection columns must be orthonormal")
    if not records:
        raise ValueError("Feature extraction needs at least one input group")
    images = []
    for record in records:
        for phase in ("a", "p", "v"):
            with Image.open(record["images"][phase]) as image:
                images.append(image.convert("RGB"))
    inputs = backend.processor.image_processor(images=images, return_tensors="pt")
    pixels = inputs["pixel_values"].to(device=backend.model.device, dtype=backend.model.dtype)
    grids = inputs["image_grid_thw"].to(backend.model.device)
    patch_counts = [int(grid.prod().item()) for grid in grids]
    if len(patch_counts) != 3 * len(records):
        raise ValueError("The processor must emit one patch grid per phase image")
    captured = []
    vision = backend.model.model.visual
    handle = vision.merger.register_forward_pre_hook(lambda module, args: captured.append(args[0]))
    training = backend.model.training
    backend.model.eval()
    try:
        vision(pixels, grid_thw=grids)
    finally:
        handle.remove()
        backend.model.train(training)
    if len(captured) != 1 or captured[0].shape != (sum(patch_counts), dimension):
        raise ValueError("Vision patch states must match image grids before the final merger")
    phases = torch.split(captured[0].float(), patch_counts)
    vectors = []
    for index in range(len(records)):
        states = phases[3 * index:3 * index + 3]
        vectors.append(project_visual_latent(
            states, projection,
            phase_masks=[torch.ones(state.shape[0], device=state.device, dtype=torch.bool)
                         for state in states],
        ))
    return vectors
