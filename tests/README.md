# Tests

The required test modules use hand-checkable toy fixtures only. Run them through
`../scripts/run_required_tests.sh` to create the audited test log.

`test_verl_sample_uid.py` tests separate rollout-group and persistent-input
identity, repeated-input EMA updates, sampler state updates, and bridge validation.
Set `RL_LHM_VERL_RAY_TRAINER` to the supported reference `ray_trainer.py` to run its
source-integration cases as well; these require PyTorch but no patient data,
model weights, GPU job, or running Ray service.

`test_visual_feature_cache.py` checks projection/pooling, target isolation,
cache conflicts and recovery. Its real small vision-model forward requires
`transformers==4.57.1` and safetensors; otherwise that one case is explicitly
skipped. All other feature contract tests use Torch with synthetic inputs.
