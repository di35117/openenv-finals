"""
GRPO Training Script — EnterpriseOps Gym
Follows the standard OpenEnv + TRL pattern.

Everything runs on the H100 (or equivalent GPU):
  - vLLM (colocate mode) handles agent inference during GRPO
  - OpenEnv server runs on port 8000 (talks to simulated APIs and LLM Judge)
  - External judge (Claude/GPT via API) handles Tier 5 Candidates and scoring

Setup (2 terminals):

  # Terminal 1: OpenEnv server 
  uv run server --port 8000

  # Terminal 2: GRPO training (full 80GB for agent)
  python train.py --vllm-mode colocate
"""

from __future__ import annotations

import argparse
import logging
import os
from datetime import datetime
from pathlib import Path

# Help PyTorch reuse fragmented GPU memory (critical for TRL+vLLM colocate)
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
os.environ.setdefault("TRL_EXPERIMENTAL_SILENCE", "1")

from datasets import Dataset
from transformers import AutoTokenizer
from peft import LoraConfig
from trl import GRPOConfig, GRPOTrainer
from trl.experimental.openenv import generate_rollout_completions

from models import EnterpriseOpsAction
from client import EnterpriseOpsEnv

# ---- TRL / vLLM Compatibility Patch (Directly ported from Kube SRE Gym) ----
_orig_vllm_gen = None

def _patch_vllm_generate(trainer):
    global _orig_vllm_gen
    if _orig_vllm_gen is not None or not hasattr(trainer, 'vllm_generation'):
        return
    _orig_vllm_gen = trainer.vllm_generation.generate

    def _wrapped_generate(**kwargs):
        result = _orig_vllm_gen(**kwargs)
        prompt_ids, completion_ids, logprobs, *rest = result
        if logprobs and logprobs[0] and isinstance(logprobs[0][0], float):
            logprobs = [[[lp] for lp in seq] for seq in logprobs]
        return (prompt_ids, completion_ids, logprobs, *rest)

    trainer.vllm_generation.generate = _wrapped_generate

def patch_trl_vllm_compat():
    _orig_train = GRPOTrainer.train
    def _patched_train(self, *args, **kwargs):
        _patch_vllm_generate(self)
        return _orig_train(self, *args, **kwargs)
    GRPOTrainer.train = _patched_train

if __name__ == "__main__":
    patch_trl_vllm_compat()

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)


# ============================================================
# System Prompt (The Agent's Persona & Action Space)
# ============================================================

SYSTEM_PROMPT = """You are an AI HR Operations Coordinator. Your goal is to negotiate a job offer with a candidate, close the deal, and successfully execute their onboarding while navigating corporate policies and IT systems.

Output ONE command per turn. No explanations, no markdown, no prefixes.

PHASE 1: NEGOTIATION
You must discover the candidate's hidden requirements. Use these tools:
- diagnose:situation <your thoughts> (use this to record your strategy)
- make_offer <base> <equity> <signing>
- apply_deadline_pressure
- request_competing_offer_details
- escalate_to_budget_owner (unlocks 10% more budget, but costs time)
- close_deal (transitions to Phase 2)

PHASE 2: ONBOARDING OPERATIONS
Once the deal is closed, you must provision the employee using MCP connectors.
- policy:lookup <POLICY_KEY> (Always check GDPR_DATA_RETENTION_DAYS in case of schema drift)
- email:send <recipient> <subject>
- calendar:schedule <meeting_title>
- drive:share <document_name> <user_email>
- ticket:list
- ticket:close <ticket_id>

CRITICAL RULES:
1. If corporate policy changes mid-episode, you MUST do a policy:lookup before closing tickets or you will fail compliance.
2. An offer that is too high will cause a delayed Finance rejection (-0.4 reward).
3. An offer that is too low will cause the candidate to quit at day 87 (-0.6 reward).
4. You must clear all open IT tickets, share docs, and send a welcome email to win.
"""

# ============================================================
# Helpers
# ============================================================

def format_observation(obs) -> str:
    text = f"""{obs.command_output}

CURRENT SYSTEM STATUS:
{obs.system_status_summary}"""
    if obs.hint:
        text += f"\n\nHINT: {obs.hint}"
    text += f"\n\nStep {obs.steps_taken}/{obs.max_steps}. What is your next command?"
    return text

def format_history(history: list[dict]) -> str:
    if not history:
        return ""
    lines = ["PREVIOUS COMMANDS AND RESULTS:"]
    for entry in history:
        cmd = entry["command"]
        output = entry["output"]
        if len(output) > 300:
            output = output[:300] + "... (truncated)"
        lines.append(f"$ {cmd}")
        lines.append(f"  Output: {output}")
    return "\n".join(lines)

def parse_commands(text: str) -> list[str]:
    """Extract tools from agent output."""
    commands = []
    seen = set()
    for line in text.strip().split("\n"):
        line = line.strip()
        # Catch our new HR/IT specific commands
        valid_prefixes = (
            "make_offer", "apply_deadline_pressure", "request_competing_offer_details",
            "escalate_to_budget_owner", "close_deal", "policy:lookup", "email:send",
            "calendar:schedule", "drive:share", "ticket:", "escalate:manager", "diagnose:"
        )
        if line.startswith(valid_prefixes):
            if line not in seen:
                commands.append(line)
                seen.add(line)
        if len(commands) >= 2:
            break
    return commands

def apply_chat_template(tokenizer, messages):
    try:
        return tokenizer.apply_chat_template(messages, add_generation_prompt=True, tokenize=False, enable_thinking=False)
    except TypeError:
        return tokenizer.apply_chat_template(messages, add_generation_prompt=True, tokenize=False)


# ============================================================
# Rollout Logic
# ============================================================

def rollout_once(trainer, env: EnterpriseOpsEnv, tokenizer, system_prompt: str, max_turns: int) -> dict:
    result = env.reset()
    observation = result.observation

    prompt_ids, completion_ids, logprobs = [], [], []
    step_rewards, negotiation_rewards, onboarding_rewards = [], [], []
    conversation_history = []
    
    MAX_TOTAL_TOKENS = 4096

    for _turn in range(max_turns):
        if result.done or len(completion_ids) >= MAX_TOTAL_TOKENS:
            break

        history_text = format_history(conversation_history)
        obs_text = format_observation(observation)
        user_prompt = f"{history_text}\n\n---\n\nCURRENT OBSERVATION:\n{obs_text}" if history_text else obs_text

        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ]
        prompt_text = apply_chat_template(tokenizer, messages)

        rollout_outputs = generate_rollout_completions(trainer, [prompt_text])[0]
        prompt_ids.extend(rollout_outputs["prompt_ids"])
        completion_ids.extend(rollout_outputs["completion_ids"])
        logprobs.extend(rollout_outputs["logprobs"])

        completion_text = rollout_outputs.get("text") or tokenizer.decode(rollout_outputs["completion_ids"], skip_special_tokens=True)
        commands = parse_commands(completion_text)

        if not commands:
            step_rewards.append(-0.5)
            conversation_history.append({"command": "INVALID", "output": "error: expected valid tool command."})
            continue

        for cmd in commands:
            try:
                result = env.step(EnterpriseOpsAction(command=cmd))
                reward = float(result.reward or 0.0)
                step_rewards.append(reward)
                observation = result.observation

                conversation_history.append({"command": cmd, "output": observation.command_output[:500]})

                # Map Phase 1 to 'diagnosis' and Phase 2 to 'fix' for standard plotting
                if "offer" in cmd or "pressure" in cmd or "escalate" in cmd:
                    negotiation_rewards.append(reward)
                elif ":" in cmd: # MCP tools
                    onboarding_rewards.append(reward)

                if result.done:
                    break
            except Exception as e:
                logger.warning(f"Step error: {e}")
                break

    total_reward = sum(step_rewards) if step_rewards else -1.0
    
    # Save transcript
    try:
        import json
        with open("agent_transcripts.jsonl", "a") as f:
            f.write(json.dumps({
                "total_reward": total_reward,
                "num_steps": len(conversation_history),
                "conversation": conversation_history,
            }) + "\n")
    except Exception:
        pass

    return {
        "prompt_ids": prompt_ids,
        "completion_ids": completion_ids,
        "logprobs": logprobs,
        "total_reward": total_reward,
        "diagnosis_reward": negotiation_rewards[-1] if negotiation_rewards else 0.0,
        "fix_reward": onboarding_rewards[-1] if onboarding_rewards else 0.0,
    }


# ============================================================
# Reward Functions (TRL format)
# ============================================================

def reward_total(completions, **kwargs): return [float(r) for r in kwargs.get("total_reward", [])] if kwargs else [0.0]*len(completions)
def reward_diagnosis(completions, **kwargs): return [float(r) for r in kwargs.get("diagnosis_reward", [])] if kwargs else [0.0]*len(completions)
def reward_fix(completions, **kwargs): return [float(r) for r in kwargs.get("fix_reward", [])] if kwargs else [0.0]*len(completions)


# ============================================================
# Main Loop
# ============================================================

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-id", default="Qwen/Qwen3-0.6B")
    parser.add_argument("--env-url", default="http://localhost:8000")
    parser.add_argument("--dataset-size", type=int, default=50)
    parser.add_argument("--max-turns", type=int, default=20)
    parser.add_argument("--num-generations", type=int, default=8)
    parser.add_argument("--vllm-mode", default="colocate")
    args = parser.parse_args()

    tokenizer = AutoTokenizer.from_pretrained(args.model_id)
    if tokenizer.pad_token is None: tokenizer.pad_token = tokenizer.eos_token

    env = EnterpriseOpsEnv(base_url=args.env_url)
    dataset = Dataset.from_dict({"prompt": ["Process this candidate and onboarding."] * args.dataset_size})
    
    output_dir = Path("outputs") / f"enterpriseops-grpo-{datetime.now().strftime('%Y%m%d_%H%M')}"
    output_dir.mkdir(parents=True, exist_ok=True)

    grpo_config = GRPOConfig(
        use_vllm=True, vllm_mode=args.vllm_mode, output_dir=str(output_dir),
        learning_rate=2e-6, lr_scheduler_type="cosine", max_grad_norm=1.0,
        gradient_accumulation_steps=8, per_device_train_batch_size=1,
        generation_batch_size=args.num_generations, num_generations=args.num_generations,
        max_completion_length=512, temperature=1.0, gradient_checkpointing=True,
    )

    import csv
    reward_log_path = output_dir / "reward_log.csv"
    with open(reward_log_path, "w", newline="") as f:
        csv.writer(f).writerow(["episode", "total_reward", "diagnosis_reward", "fix_reward", "timestamp"])

    def rollout_func(prompts, trainer):
        results = {"prompt_ids": [], "completion_ids": [], "logprobs": [], "total_reward": [], "diagnosis_reward": [], "fix_reward": []}
        for prompt in prompts:
            ep = rollout_once(trainer, env, tokenizer, SYSTEM_PROMPT, args.max_turns)
            for k in results: results[k].append(ep[k])
            with open(reward_log_path, "a", newline="") as f:
                csv.writer(f).writerow([len(results["total_reward"]), ep["total_reward"], ep["diagnosis_reward"], ep["fix_reward"], datetime.now().isoformat()])
        return results

    peft_config = LoraConfig(r=16, lora_alpha=32, lora_dropout=0.05, bias="none", task_type="CAUSAL_LM", target_modules=["q_proj", "k_proj", "v_proj", "o_proj"])
    
    trainer = GRPOTrainer(
        model=args.model_id, processing_class=tokenizer,
        reward_funcs=[reward_total, reward_diagnosis, reward_fix],
        train_dataset=dataset, args=grpo_config, rollout_func=rollout_func, peft_config=peft_config
    )

    logger.info("Starting GRPO Training for EnterpriseOps Gym...")
    try:
        trainer.train()
    finally:
        env.close()
        trainer.save_model(str(output_dir))

if __name__ == "__main__":
    main()