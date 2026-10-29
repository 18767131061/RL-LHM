"""Collectives and ZeRO-2 optimizer ownership for four local CUDA workers."""
from __future__ import annotations

import math
import os

import torch
import torch.distributed as dist


def initialize(world_size=4):
    if world_size != 4 or int(os.environ.get("LOCAL_WORLD_SIZE", "0")) != 4:
        raise ValueError("training requires four workers on one host")
    if int(os.environ.get("WORLD_SIZE", "1")) != world_size:
        raise ValueError(f"Launch with torchrun --standalone --nproc_per_node={world_size}")
    if not torch.cuda.is_available() or not torch.cuda.is_bf16_supported():
        raise ValueError("Training requires CUDA GPUs with BF16 support")
    local_rank = int(os.environ["LOCAL_RANK"])
    torch.cuda.set_device(local_rank)
    dist.init_process_group("nccl")
    return local_rank


class Runtime:
    """Global token normalization; local optimizer also supports CPU tests."""
    def __init__(self, model, optimizer=None, engine=None):
        self.model, self.optimizer, self.engine = model, optimizer, engine
        self.rank = dist.get_rank() if dist.is_initialized() else 0
        self.world_size = dist.get_world_size() if dist.is_initialized() else 1
        self.device = next(model.parameters()).device

    def gather(self, value):
        if self.world_size == 1:
            return [value]
        values = [None] * self.world_size
        dist.all_gather_object(values, value)
        return values

    def sum(self, value):
        tensor = torch.tensor(value, dtype=torch.float64, device=self.device)
        if self.world_size > 1:
            dist.all_reduce(tensor)
        return tensor.item()

    def slots(self, batch):
        """Equal forward/backward counts, with zero-weight padding on short tails."""
        count = math.ceil(len(batch) / self.world_size)
        return [(position, batch[position] if position < len(batch) else batch[0], position < len(batch))
                for position in (self.rank + i * self.world_size for i in range(count))]

    def begin(self, microsteps):
        if self.engine is None:
            self.optimizer.zero_grad(set_to_none=True)
        else:
            # CPU ZeRO accumulation uses this count in addition to the engine
            # boundary flag. Update it for the partial final optimizer batch.
            self.engine.set_train_batch_size(microsteps * self.world_size)
            self.engine.optimizer.gradient_accumulation_steps = microsteps

    def boundary(self, last):
        if self.engine is not None:
            self.engine.set_gradient_accumulation_boundary(last)

    def backward(self, loss):
        if loss.ndim != 0 or not loss.requires_grad or not torch.isfinite(loss):
            raise ValueError("loss must be a finite differentiable scalar")
        # DDP and ZeRO average gradients over ranks; the loss is already divided
        # by the global token count, so compensate exactly once here.
        loss = loss * self.world_size
        if self.engine is None:
            loss.backward()
        else:
            self.engine.backward(loss, scale_wrt_gas=False)

    def step(self, max_grad_norm):
        if self.engine is not None:
            self.engine.step()
            norm = float(self.engine.get_global_grad_norm())
        else:
            norm = float(torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_grad_norm,
                                                       error_if_nonfinite=True))
            self.optimizer.step()
        if not math.isfinite(norm) or norm < 0:
            raise FloatingPointError("non-finite global gradient norm")
        return norm


def zero2_runtime(backend, settings, max_grad_norm):
    import deepspeed
    from deepspeed.ops.adam import DeepSpeedCPUAdam

    if settings["amsgrad"]:
        raise ValueError("DeepSpeedCPUAdam does not support AMSGrad")
    optimizer = DeepSpeedCPUAdam(
        backend.model.parameters(), lr=settings["learning_rate"], betas=tuple(settings["betas"]),
        eps=settings["epsilon"], weight_decay=settings["weight_decay"], adamw_mode=True,
        fp32_optimizer_states=True,
    )
    config = {
        "train_micro_batch_size_per_gpu": 1, "gradient_accumulation_steps": 2,
        "bf16": {"enabled": True}, "gradient_clipping": max_grad_norm,
        "zero_optimization": {
            "stage": 2, "offload_optimizer": {"device": "cpu", "pin_memory": True},
            "contiguous_gradients": True, "overlap_comm": False,
            "reduce_scatter": True, "reduce_bucket_size": 20000000,
            "allgather_bucket_size": 20000000,
        },
        "data_types": {"grad_accum_dtype": "fp32"},
        "steps_per_print": 1000000, "wall_clock_breakdown": False,
    }
    engine, _, _, _ = deepspeed.initialize(model=backend.model, optimizer=optimizer,
                                          config=config, dist_init_required=False)
    backend.forward_model = engine
    return Runtime(backend.model, engine=engine)
