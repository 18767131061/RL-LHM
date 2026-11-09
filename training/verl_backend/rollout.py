"""Real VeRL/vLLM generation; defer RARS until raw actor logprobs exist."""
from __future__ import annotations

import time
from verl.experimental.agent_loop.agent_loop import AgentLoopBase, AgentLoopManager, AgentLoopOutput


class DeferredRewardManager(AgentLoopManager):
    def __init__(self, config, llm_client, teacher_client=None, reward_loop_worker_handles=None):
        # Disable the carrier's early streaming reward call. No fabricated
        # score is returned: the driver computes RARS after actor inference.
        super().__init__(config, llm_client, teacher_client, reward_loop_worker_handles=None)


class RawSingleTurnLoop(AgentLoopBase):
    """Preserve generated token IDs, without a second assistant-template merge."""

    async def run(self, sampling_params, priority=0, **kwargs):
        messages = list(kwargs["raw_prompt"])
        mm = await self.process_multi_modal_info(messages)
        self._assert_mm_supported(bool(mm))
        mm_inputs = {}
        prompt = await self.ct_build_initial_tokens(messages, images=mm.get("images"),
                                                    videos=mm.get("videos"), audios=mm.get("audios"),
                                                    mm_inputs_out=mm_inputs)
        generation = self.config.rl_lhm.generation
        if len(prompt) > self.rollout_config.prompt_length:
            raise ValueError("expanded three-image prompt exceeds max_prompt_length")
        parameters = dict(sampling_params)
        parameters.update(seed=int(kwargs["extra_info"]["rl_lhm_seed"]),
                          max_tokens=int(generation.max_new_tokens),
                          stop_token_ids=list(generation.eos_token_ids))
        if generation.suppress_token_ids:
            # vLLM's allowed_token_ids is a sampling-only vocabulary mask.
            # Raw actor/reference scoring retains the full model vocabulary.
            from transformers import AutoConfig
            hf = AutoConfig.from_pretrained(self.config.actor_rollout_ref.model.path, local_files_only=True)
            text_config = hf.get_text_config()
            blocked = set(generation.suppress_token_ids)
            parameters["allowed_token_ids"] = [i for i in range(text_config.vocab_size) if i not in blocked]
        start = time.monotonic()
        output = await self.server_manager.generate(
            request_id=f"{kwargs['extra_info']['rid']}:{kwargs['extra_info']['member']}",
            prompt_ids=prompt, sampling_params=parameters, image_data=mm.get("images"),
            video_data=mm.get("videos"), audio_data=mm.get("audios"),
            mm_processor_kwargs=self._get_mm_processor_kwargs(mm.get("audios")),
            mm_processor_output=self.build_sglang_video_payload(mm.get("videos"), mm_inputs), priority=int(priority),
        )
        tokens = list(output.token_ids)
        if not tokens:
            raise RuntimeError("VeRL rollout returned no generated tokens")
        if len(tokens) > generation.max_new_tokens:
            raise RuntimeError("VeRL rollout exceeded the configured response cap")
        eos_positions = [i for i, token in enumerate(tokens) if token in generation.eos_token_ids]
        if eos_positions:
            tokens = tokens[:eos_positions[0] + 1]
        elif len(tokens) < generation.max_new_tokens:
            raise RuntimeError("rollout stopped before the response cap without retaining its EOS")
        return AgentLoopOutput(
            prompt_ids=prompt, response_ids=tokens, response_mask=[1] * len(tokens), num_turns=2,
            multi_modal_data=mm, mm_processor_kwargs=self._get_mm_processor_kwargs(mm.get("audios")),
            metrics={"generate_sequences": time.monotonic() - start},
            extra_fields={"turn_scores": [], "tool_rewards": []},
        )
