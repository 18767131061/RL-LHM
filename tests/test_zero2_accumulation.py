"""Check installed ZeRO CPU-offload accumulation using rank-averaged gradients."""
import ast
from importlib.util import find_spec
from pathlib import Path
from types import MethodType, SimpleNamespace

import pytest
import torch


def offload_methods(source):
    """Load actual dependency methods without importing its accelerator extensions."""
    tree = ast.parse(source.read_text())
    cls = next(node for node in tree.body if isinstance(node, ast.ClassDef)
               and node.name == "DeepSpeedZeroOptimizer")
    names = {
        "get_param_gradient_attribute", "clear_grad_attribute",
        "async_accumulate_grad_in_cpu_via_gpu", "copy_grads_in_partition",
        "async_inplace_copy_grad_to_fp32_buffer_from_gpu",
        "set_norm_for_param_grad_in_gpu", "zero_grad", "_clear_param_grad_only",
        "_release_ipg_buffers", "independent_gradient_partition_epilogue",
    }
    namespace = {"torch": torch, "see_memory_usage": lambda *args, **kwargs: None}
    for node in cls.body:
        if isinstance(node, ast.FunctionDef) and node.name in names:
            exec(compile(ast.Module(body=[node], type_ignores=[]), str(source), "exec"), namespace)
    return {name: namespace[name] for name in names if name in namespace}


@pytest.fixture(scope="module")
def methods():
    spec = find_spec("deepspeed")
    if spec is None:
        pytest.skip("install requirements-training.txt for the ZeRO regression tests")
    return offload_methods(Path(spec.origin).parent / "runtime/zero/stage_1_and_2.py")


class OffloadPartition:
    def __init__(self, methods, dtype, width, offset=0, length=None):
        self.offset = offset
        self.length = width - offset if length is None else length
        self.parameter = torch.nn.Parameter(torch.zeros(width, dtype=dtype))
        self.master = torch.nn.Parameter(torch.zeros(self.length))
        self.master.grad = torch.zeros_like(self.master)
        self.adam = torch.optim.AdamW([self.master], lr=0.01, weight_decay=0.01)
        self.state = SimpleNamespace(
            cpu_offload=True, cpu_offload_pin_memory=False, device="cpu",
            fp16_master_weights_and_gradients=False, low_precision_master_weights_and_grads=False,
            master_weights_and_grads_dtype=torch.float32, use_grad_accum_attribute=False,
            accumulated_grads_in_cpu={}, norm_for_param_grads={},
            temp_grad_buffer_for_gpu_offload=torch.zeros(width, dtype=dtype),
            single_partition_of_fp32_groups=[self.master],
            grad_position={0: [0, offset, 0, self.length]},
            contiguous_gradients=True, overlap_comm=False, ipg_buckets={},
            bit16_groups=[[self.parameter]], params_already_reduced=[False],
        )
        for name, method in methods.items():
            setattr(self.state, name, MethodType(method, self.state))
        self.state.get_param_id = lambda parameter: 0
        self.state.report_ipg_memory_usage = lambda *args, **kwargs: None
        self.state.update_offload_overflow_tracker_for_param_grad = lambda parameter: None
        self.state.reduce_ipg_grads = lambda: self.state.copy_grads_in_partition(self.parameter)

    def accumulate(self, rows):
        self.state.gradient_accumulation_steps = len(rows)
        for index, row in enumerate(rows):
            self.state.micro_step_id = index
            self.state.is_gradient_accumulation_boundary = index == len(rows) - 1
            # Autograd produces each microbatch gradient before reduction/offload.
            (self.parameter * row.to(self.parameter.dtype)).sum().backward()
            self.state.independent_gradient_partition_epilogue()
            assert self.parameter.grad is None
        return self.master.grad.detach().clone()


CASES = [
    [[1.], [2.], [4.]],
    [[1.], [-4.], [1.]],
    [[1.]] * 32,
    [[1., 0.], [0., 10.], [-1., 0.]],
    [[1.], [2.]],
    [[-3.]],
]


@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
@pytest.mark.parametrize("values", CASES)
def test_offload_matches_unsplit_gradient_and_adam(methods, dtype, values):
    rows = torch.tensor(values)
    offload = OffloadPartition(methods, dtype, rows.shape[1])
    reference = torch.nn.Parameter(torch.zeros(rows.shape[1]))
    optimizer = torch.optim.AdamW([reference], lr=0.01, weight_decay=0.01)
    (rows * reference).sum().backward()
    torch.testing.assert_close(offload.accumulate(rows), reference.grad, rtol=0, atol=0)
    torch.testing.assert_close(offload.state.norm_for_param_grads[0], reference.grad.double().norm())
    torch.nn.utils.clip_grad_norm_([offload.master], 1.0)
    torch.nn.utils.clip_grad_norm_([reference], 1.0)
    offload.adam.step()
    optimizer.step()
    torch.testing.assert_close(offload.master, reference, rtol=0, atol=0)


@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
def test_partition_tails_and_restart_match_unsplit_updates(methods, dtype, tmp_path):
    # A parameter crossing a rank partition; nonzero source offset and short slice.
    windows = [torch.tensor([[1., -2., 3., -4., 5., -6.]]) * factor
               for factor in (1., -1., 2., -2.)]
    windows = [windows[0].repeat(32, 1), torch.cat(windows[:3]), windows[3],
               torch.cat(windows[:2]), windows[2].repeat(32, 1)]
    reference = torch.nn.Parameter(torch.zeros(3))
    optimizer = torch.optim.AdamW([reference], lr=0.01, weight_decay=0.01)
    offload = OffloadPartition(methods, dtype, 6, offset=2, length=3)
    resumed = None
    for index, rows in enumerate(windows):
        optimizer.zero_grad(set_to_none=True)
        (rows[:, 2:5] * reference).sum().backward()
        for current in (offload,) if resumed is None else (offload, resumed):
            torch.testing.assert_close(current.accumulate(rows), reference.grad, rtol=0, atol=0)
            torch.nn.utils.clip_grad_norm_([current.master], 1.0)
            current.adam.step()
        torch.nn.utils.clip_grad_norm_([reference], 1.0)
        optimizer.step()
        for current in (offload,) if resumed is None else (offload, resumed):
            torch.testing.assert_close(current.master, reference, rtol=0, atol=0)
            for key in ("step", "exp_avg", "exp_avg_sq"):
                torch.testing.assert_close(current.adam.state[current.master][key],
                                           optimizer.state[reference][key], rtol=0, atol=0)
        if index == 0:
            checkpoint = tmp_path / "optimizer.pt"
            torch.save({"master": offload.master.detach(), "optimizer": offload.adam.state_dict()}, checkpoint)
            saved = torch.load(checkpoint, weights_only=True)
            resumed = OffloadPartition(methods, dtype, 6, offset=2, length=3)
            with torch.no_grad():
                resumed.master.copy_(saved["master"])
            resumed.adam.load_state_dict(saved["optimizer"])
