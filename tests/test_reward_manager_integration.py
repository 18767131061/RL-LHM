"""Real reward-manager/callback execution; VeRL base/container and tokenizer are substitutes."""
import asyncio
import importlib.util
import json
import math
import sys
import types
from pathlib import Path
import pytest
import torch
from rewards.verl_liver_reward import compute_score


@pytest.fixture
def manager_module(monkeypatch):
    names=["verl","verl.experimental","verl.experimental.reward_loop","verl.experimental.reward_loop.reward_manager",
           "verl.experimental.reward_loop.reward_manager.base","verl.utils","verl.utils.reward_score"]
    for name in names:
        module=types.ModuleType(name);module.__path__=[];monkeypatch.setitem(sys.modules,name,module)
    class Base:
        def __init__(self,config,tokenizer,compute_score):
            self.tokenizer=tokenizer;self.loop=asyncio.get_running_loop()
    sys.modules[names[4]].RewardManagerBase=Base
    sys.modules[names[-1]].default_compute_score=lambda **kwargs:0.
    path=Path(__file__).resolve().parents[1]/"rewards/verl_completion_likelihood_reward_manager.py"
    spec=importlib.util.spec_from_file_location("synthetic_reward_manager",path)
    module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
    return module


def test_real_manager_passes_masked_completion_nll_into_real_reward(manager_module,monkeypatch):
    monkeypatch.setenv("RL_LHM_REWARD_MODE","rars")
    raw=json.dumps({"detections":[{"phase":p,"label":"ICC","bbox_2d":[0,0,10,10]} for p in ["a","p","v"]]})
    truth={"gt_dets":[{"phase":p,"label":"HCC","bbox":[0,0,10,10]} for p in ["a","p","v"]]}
    decoded=[]
    tokenizer=types.SimpleNamespace(decode=lambda ids,**kwargs:(decoded.append(ids.tolist()) or raw))
    item=types.SimpleNamespace(batch={"responses":torch.tensor([[11,12,0]]),
                                      "attention_mask":torch.tensor([[1,1,1,1,0]]),
                                      "rollout_log_probs":torch.tensor([[-.1,-.3,float("nan")]])},
                               non_tensor_batch={"data_source":"synthetic","reward_model":{"ground_truth":truth},
                                                 "extra_info":{"data_mode":"private","class_counts":{"HCC":10,"ICC":10}}})
    async def execute():
        manager=manager_module.CompletionLikelihoodRewardManager({},tokenizer,compute_score)
        return await manager.run_single([item])
    result=asyncio.run(execute())
    assert decoded==[[11,12]]
    assert result["reward_extra_info"]["completion_nll"]==pytest.approx(.2)
    assert result["reward_extra_info"]["seq_confidence"]==pytest.approx(math.exp(-.2))
    assert result["reward_extra_info"]["parse_ok"]==1
    assert -6<=result["reward_score"]<=3
    assert result["reward_score"]<0


@pytest.mark.parametrize("fault",["length","mask","nonfinite"])
def test_reward_manager_rejects_invalid_generated_span(manager_module,fault):
    data={"responses":torch.tensor([[11,12,0]]),"attention_mask":torch.tensor([[1,1,1,1,0]]),
          "rollout_log_probs":torch.tensor([[-.1,-.3,-99.]])}
    if fault=="length":data["rollout_log_probs"]=torch.tensor([[-.1]])
    elif fault=="mask":data["attention_mask"][0,2]=2
    else:data["rollout_log_probs"][0,0]=float("nan")
    with pytest.raises(RuntimeError):
        ids,lp=manager_module.extract_generated_tokens_and_log_probs(data)
        manager_module.completion_likelihood(lp,len(ids))
