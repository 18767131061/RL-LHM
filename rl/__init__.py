"""Core advantages, policy objective, and visual curriculum algorithms."""
from .completion_likelihood import completion_likelihood
from .core import ShaeBaselineStore, compute_grpo_advantage, compute_shae_advantage, rollout_groups
from .curriculum import DACVQState, VGSSampler, vgs_sampling_probabilities, vgs_sampling_weights
from .policy_contract import grpo_token_loss
from .visual import project_visual_latent
from .workflow import update_curriculum

__all__ = [
    "completion_likelihood", "ShaeBaselineStore", "compute_grpo_advantage",
    "compute_shae_advantage", "rollout_groups", "DACVQState", "VGSSampler",
    "vgs_sampling_probabilities", "vgs_sampling_weights", "grpo_token_loss",
    "project_visual_latent", "update_curriculum",
]
