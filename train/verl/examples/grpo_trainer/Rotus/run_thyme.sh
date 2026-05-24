#!/bin/bash

export JUDGE_SERVER="${JUDGE_SERVER:-http://localhost:18901/v1}"
export THYME_CACHE_DIR="${THYME_CACHE_DIR:-/workspace/verl/tmp_cache}"
export CUDA_VISIBLE_DEVICES="0,1,2"

# sandbox.py dependencies
pip install timeout-decorator autopep8 weave -q

set -x

# Configuration
MODEL_PATH="${MODEL_PATH:-/workspace/models/Thyme/Thyme-RL}"
TRAIN_DATA="${TRAIN_DATA:-/workspace/data/Dataset/combined_filtered/thyme/combined_train_10000.json}"
VAL_DATA="${VAL_DATA:-/workspace/data/Dataset/combined_filtered/thyme/combined_val_200.json}"
PROJECT_NAME="${PROJECT_NAME:-thyme}"
EXPERIMENT_NAME="${EXPERIMENT_NAME:-thyme_run}"
TOOL_CONFIG_PATH="${TOOL_CONFIG_PATH:-$(pwd)/examples/grpo_trainer/config/tool_config/thyme_sandbox_tool_config.yaml}"
REWARD_FN_PATH="${REWARD_FN_PATH:-verl/utils/reward_score/rotus.py}"

# GPU config
N_GPUS="${N_GPUS:-3}"
NNODES="${NNODES:-1}"

mkdir -p /workspace/ray_tmp

ulimit -n 65535

python3 -m verl.trainer.main_ppo \
    --config-path="$(pwd)/examples/grpo_trainer/config" \
    --config-name='multiturn_grpo' \
    algorithm.adv_estimator=grpo \
    data.train_files="${TRAIN_DATA}" \
    data.val_files="${VAL_DATA}" \
    data.train_batch_size=24 \
    data.max_prompt_length=16384 \
    data.max_response_length=4096 \
    data.filter_overlong_prompts=True \
    data.truncation='error' \
    data.return_raw_chat=True \
    data.return_multi_modal_inputs=False \
    data.image_key=images \
    actor_rollout_ref.hybrid_engine=True \
    actor_rollout_ref.model.path="${MODEL_PATH}" \
    actor_rollout_ref.actor.optim.lr=5e-7\
    actor_rollout_ref.model.use_remove_padding=True \
    actor_rollout_ref.actor.ppo_mini_batch_size=6 \
    actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=2 \
    actor_rollout_ref.actor.use_kl_loss=True \
    actor_rollout_ref.actor.kl_loss_coef=0.001 \
    actor_rollout_ref.actor.kl_loss_type=low_var_kl \
    actor_rollout_ref.actor.entropy_coeff=0 \
    actor_rollout_ref.model.enable_gradient_checkpointing=True \
    actor_rollout_ref.actor.fsdp_config.param_offload=False \
    actor_rollout_ref.actor.fsdp_config.optimizer_offload=False \
    actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu=4 \
    actor_rollout_ref.rollout.tensor_model_parallel_size=1 \
    actor_rollout_ref.rollout.name=vllm \
    actor_rollout_ref.rollout.gpu_memory_utilization=0.6 \
    actor_rollout_ref.rollout.n=8 \
    actor_rollout_ref.rollout.agent.default_agent_loop=tool_agent \
    actor_rollout_ref.rollout.multi_turn.enable=True \
    actor_rollout_ref.rollout.multi_turn.max_assistant_turns=5 \
    actor_rollout_ref.rollout.multi_turn.tool_config_path="${TOOL_CONFIG_PATH}" \
    actor_rollout_ref.rollout.multi_turn.format="code_block" \
    +actor_rollout_ref.rollout.multi_turn.sandbox_role="inline" \
    +actor_rollout_ref.rollout.multi_turn.stop_words='["</answer>", "</code>"]' \
    actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu=4 \
    actor_rollout_ref.ref.fsdp_config.param_offload=True \
    algorithm.use_kl_in_reward=False \
    reward_model.use_reward_loop=True \
    reward_model.reward_manager=naive_async \
    +reward_model.reward_kwargs.num_workers=32 \
    custom_reward_function.path="${REWARD_FN_PATH}" \
    custom_reward_function.name="compute_score" \
    +custom_reward_function.reward_kwargs.tool_tag="code" \
    +custom_reward_function.reward_kwargs.max_assistant_turns=5 \
    +custom_reward_function.reward_kwargs.optimal_tool_calls=2 \
    +custom_reward_function.reward_kwargs.penalty_lambda=0.1 \
    trainer.critic_warmup=0 \
    trainer.logger='["console","wandb"]' \
    trainer.project_name="${PROJECT_NAME}" \
    trainer.experiment_name="${EXPERIMENT_NAME}" \
    trainer.n_gpus_per_node="${N_GPUS}" \
    trainer.nnodes="${NNODES}" \
    trainer.save_freq=10 \
    trainer.test_freq=10 \
    trainer.val_before_train=False \
    trainer.total_epochs=3 \
    +ray_kwargs.ray_init.runtime_env.env_vars.JUDGE_SERVER="${JUDGE_SERVER}" \
    +ray_kwargs.ray_init.runtime_env.env_vars.THYME_CACHE_DIR="${THYME_CACHE_DIR}" \
    +ray_kwargs.ray_init.runtime_env.env_vars.VERL_LOGGING_LEVEL="INFO" "$@" \
    +ray_kwargs.ray_init._temp_dir="/workspace/ray_tmp" "$@"
