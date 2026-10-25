"""CPU checks of exact upstream loss functions and patched worker method.

The actual model, DataProto container, distributed communication and FSDP wrapper
are substitutes. The extracted update_policy body, loss, KL and Torch autograd
execute as written on the CPU.
"""
from __future__ import annotations
import ast
import copy
import importlib.util
import os
import sys
import types
from pathlib import Path
import pytest
import torch
from omegaconf import OmegaConf
from rl.policy_contract import POLICY_OVERRIDES, verify_overrides, verify_core, global_token_batch_info
from rl.patch_verl_token_mean import patch_source, require_token_mean_patch

@pytest.fixture
def upstream():
    directory = os.environ.get("RL_LHM_VERL_REFERENCE_DIR")
    if not directory: pytest.skip("Set RL_LHM_VERL_REFERENCE_DIR to pinned core_algos.py/dp_actor.py")
    root = Path(directory)
    verify_core(root / "core_algos.py")
    return root


def namespace(upstream):
    tree = ast.parse((upstream / "core_algos.py").read_text())
    functions = [n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name in
                 {"agg_loss", "compute_policy_loss_vanilla", "kl_penalty", "kl_penalty_forward"}]
    for node in functions: node.decorator_list = []
    tensor_helpers = types.SimpleNamespace(masked_sum=lambda x,m: (x*m).sum(), masked_mean=lambda x,m: (x*m).sum()/m.sum().clamp_min(1))
    ns = {"torch":torch,"verl_F":tensor_helpers,"AlgoConfig":type("AlgoConfig",(),{})}
    module = ast.Module(body=[ast.ImportFrom(module="__future__",names=[ast.alias(name="annotations")],level=0),*functions],type_ignores=[])
    exec(compile(ast.fix_missing_locations(module),"pinned_core.py","exec"),ns)
    return ns


def config():
    return OmegaConf.create({"clip_ratio":.2,"clip_ratio_low":.2,"clip_ratio_high":.2,"clip_ratio_c":3.,
                            "global_batch_info":{},"ppo_mini_batch_size":2,"ppo_micro_batch_size_per_gpu":1,
                            "use_dynamic_bsz":False,"ppo_epochs":1,"use_kl_loss":True,"kl_loss_coef":.01,
                            "kl_loss_type":"low_var_kl","policy_loss":{"loss_mode":"vanilla"},
                            "loss_agg_mode":"token-mean","calculate_entropy":False,"entropy_coeff":0,
                            "use_rollout_log_probs":True})


def test_config_explicit_and_cannot_override_silently():
    args = [f"{k}={v}" for k,v in POLICY_OVERRIDES.items()]
    assert verify_overrides(args)["actor_rollout_ref.actor.loss_agg_mode"] == "token-mean"
    with pytest.raises(RuntimeError): verify_overrides(args[:-1])
    with pytest.raises(RuntimeError): verify_overrides(args+["actor_rollout_ref.actor.loss_agg_mode=seq-mean-token-mean"])


def test_patch_exact_idempotent_and_runtime_check(upstream,tmp_path):
    original = (upstream / "dp_actor.py").read_text()
    updated, existed = patch_source(original)
    assert existed is False
    assert patch_source(updated) == (updated,True)
    target=tmp_path/"dp_actor.py";target.write_text(original)
    with pytest.raises(RuntimeError,match="Missing"):require_token_mean_patch(target)
    target.write_text(updated)
    assert require_token_mean_patch(target)["patch_version"] == 1
    with pytest.raises(RuntimeError):patch_source(original.replace('loss_scale_factor = 1 / self.gradient_accumulation','loss_scale_factor = 2 / self.gradient_accumulation'))


def test_token_ratios_dual_clip_mask_and_kl(upstream):
    ns=namespace(upstream);cfg=config()
    lp=torch.tensor([[2.,-2.,50.],[-50.,0.,1.]],requires_grad=True)
    old=torch.zeros_like(lp);adv=torch.tensor([[1.,-1.,2.],[-2.,1.,-1.]])
    mask=torch.tensor([[1.,1.,0.],[1.,0.,1.]])
    loss,_=ns["compute_policy_loss_vanilla"](old,lp,adv,mask,config=cfg)
    ratio=torch.exp(lp.clamp(-20,20))
    surrogate=torch.minimum(ratio*adv,ratio.clamp(.8,1.2)*adv)
    surrogate=torch.where(adv<0,torch.maximum(surrogate,3*adv),surrogate)
    expected=-(surrogate*mask).sum()/mask.sum()
    torch.testing.assert_close(loss,expected)
    assert ns["kl_penalty"](lp,old,"low_var_kl").max() <= 10
    loss.backward();assert lp.grad[0,2] == 0 and lp.grad[1,1] == 0


class Data:
    def __init__(self,batch):
        self.batch=batch;self.non_tensor_batch={};self.meta_info={"temperature":1.}
    def select(self,**kwargs):return self
    def split(self,n):
        size=next(iter(self.batch.values())).shape[0]
        return [Data({key:value[i:i+n] for key,value in self.batch.items()}) for i in range(0,size,n)]
    def to(self,device):return self


def run_worker(upstream,patched):
    ns=namespace(upstream)
    src=(upstream/"dp_actor.py").read_text()
    if patched:src=patch_source(src)[0]
    fn=next(n for n in ast.walk(ast.parse(src)) if isinstance(n,ast.FunctionDef) and n.name=="update_policy")
    fn.decorator_list=[]
    ns.update({"get_device_id":lambda:"cpu", "get_policy_loss_fn":lambda name:ns["compute_policy_loss_vanilla"],
               "append_to_dict":lambda target,items:None})
    module=ast.Module(body=[ast.ImportFrom(module="__future__",names=[ast.alias(name="annotations")],level=0),fn],type_ignores=[])
    exec(compile(ast.fix_missing_locations(module),"pinned_worker.py","exec"),ns)
    parameter=torch.nn.Parameter(torch.tensor(.1))
    cfg=config();mask=torch.tensor([[1.,0.,0.],[1.,1.,1.]])
    advantages=torch.tensor([[1.,0.,0.],[3.,3.,3.]])
    batch={name:torch.zeros_like(mask) for name in ["responses","input_ids","attention_mask","position_ids","old_log_probs","ref_log_prob"]}
    batch.update({"response_mask":mask,"advantages":advantages})
    optimizer=torch.optim.SGD([parameter],lr=.05)
    actor=types.SimpleNamespace(actor_module=torch.nn.Identity(),config=cfg,actor_optimizer=optimizer,
                                use_prefix_grouper=False,ulysses_sequence_parallel_size=1,scaler=None)
    actor._forward_micro_batch=lambda inputs,**kwargs:{"log_probs":parameter.expand_as(inputs["response_mask"])}
    gradient=[]
    def step():
        gradient.append(parameter.grad.detach().clone());optimizer.step();return torch.tensor(1.)
    actor._optimizer_step=step
    # The patched worker imports the genuine project helper by module name.
    from rl import policy_contract
    sys.modules["policy_contract"]=policy_contract
    ns["update_policy"](actor,Data(batch))
    return gradient[0]


def test_actual_worker_microbatch_gradient_equals_global_token_mean(upstream):
    patched=run_worker(upstream,True);original=run_worker(upstream,False)
    expected=-2.5*torch.exp(torch.tensor(.1))+.01*(1-torch.exp(torch.tensor(-.1)))
    torch.testing.assert_close(patched,expected)
    assert not torch.isclose(original,expected)  # original equals sequence/microbatch mean


def test_dp_denominator_compensates_gradient_average(monkeypatch):
    monkeypatch.setattr(torch.distributed,"is_available",lambda:True)
    monkeypatch.setattr(torch.distributed,"is_initialized",lambda:True)
    monkeypatch.setattr(torch.distributed,"get_world_size",lambda:2)
    monkeypatch.setattr(torch.distributed,"all_reduce",lambda count,op:count.add_(3))
    assert global_token_batch_info(torch.tensor([[1.,0.,0.]]),"cpu") == {"dp_size":2,"batch_num_tokens":4}
    # Local numerator 1 and remote numerator 9: DP mean of scaled sums = 10/4.
    assert ((1*2/4)+(9*2/4))/2 == 2.5


def test_actual_launcher_policy_guard_executes_before_carrier(upstream,tmp_path,monkeypatch):
    from training import verl_lhm_launcher as launcher
    from rl import policy_contract, patch_verl_token_mean
    monkeypatch.setitem(sys.modules,"policy_contract",policy_contract)
    monkeypatch.setitem(sys.modules,"patch_verl_token_mean",patch_verl_token_mean)
    target=tmp_path/"dp_actor.py"
    target.write_text((upstream/"dp_actor.py").read_text())
    monkeypatch.setattr(launcher,"_import_local_extensions",lambda:None)
    monkeypatch.setattr(sys,"argv",["launcher","algorithm.adv_estimator=grpo",*[f"{k}={v}" for k,v in POLICY_OVERRIDES.items()]])
    original_find=launcher.importlib.util.find_spec
    monkeypatch.setattr(launcher.importlib.util,"find_spec",lambda name: (
        types.SimpleNamespace(origin=str(target)) if name=="verl.workers.actor.dp_actor" else
        types.SimpleNamespace(origin=str(upstream/"core_algos.py")) if name=="verl.trainer.ppo.core_algos" else original_find(name)))
    audit=tmp_path/"audit.json"
    monkeypatch.setenv("RL_LHM_POLICY_AUDIT_PATH",str(audit))
    calls=[]
    monkeypatch.setattr(launcher.runpy,"run_module",lambda *a,**k:calls.append(a))
    with pytest.raises(RuntimeError,match="Missing"):
        launcher.main()
    assert not calls
    target.write_text(patch_source(target.read_text())[0])
    launcher.main()
    assert calls==[("verl.trainer.main_ppo",)]
    import json
    result=json.loads(audit.read_text())
    assert result["status"]=="verified_current_source_and_explicit_overrides"
    assert result["overrides"]["actor_rollout_ref.actor.loss_agg_mode"]=="token-mean"
    assert result["dependency_versions"]["torch"]
