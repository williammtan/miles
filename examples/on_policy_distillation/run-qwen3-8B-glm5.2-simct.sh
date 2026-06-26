#!/bin/bash
#
# SimCT cross-tokenizer on-policy distillation: GLM5.2 teacher -> Qwen3-8B student.
#
# Sibling of run-qwen3.6-35B-A3B-glm5.2-cross-tokenizer.sh, but uses the SimCT
# method (arXiv 2605.07711) via miles.rollout.simct_opd and a dense Qwen3-8B
# student. This is the STUDENT TRAINER launcher; the GLM5.2 teacher runs as a
# SEPARATE SGLang server (own 8xH200 node). Point RM_URL at it. See
# examples/on_policy_distillation/k8s/cross-tokenizer-opd.yaml for the 2-node topology.
#
# Required env:
#   STUDENT_MODEL          HF id or local dir of Qwen3-8B (e.g. /workspace/Qwen3-8B)
#   TEACHER_TOKENIZER      HF id or local dir of the GLM5.2 tokenizer (e.g. /workspace/models/glm5.2-fp8)
#   RM_URL                 GLM5.2 SGLang server generate endpoint (e.g. http://glm52-teacher:30000/generate)
# Optional env:
#   STUDENT_TORCH_DIST     torch_dist checkpoint dir for the student (built here if missing)
#   DATA_PATH              prompt jsonl (default /workspace/dapo-math-17k/dapo-math-17k.jsonl)
#   OPD_KL_COEF            distillation strength (default 1.0)
#   NUM_ROLLOUT            rollout steps (default 1000; set small to smoke-test)
#   CANDIDATE_K            SimCT top-k per side (default 20)
#   MAX_CONT_LEN           SimCT max continuation tokens (default 4)
#   MEGATRON_PATH          Megatron-LM path (default /root/Megatron-LM)
#
# usage: bash examples/on_policy_distillation/run-qwen3-8B-glm5.2-simct.sh

set -ex

pkill -9 sglang || true
sleep 3
ray stop --force || true
pkill -9 ray || true
pkill -9 python || true
sleep 3

export PYTHONBUFFERED=16

STUDENT_MODEL="${STUDENT_MODEL:?set STUDENT_MODEL to the Qwen3-8B HF id or local path}"
TEACHER_TOKENIZER="${TEACHER_TOKENIZER:?set TEACHER_TOKENIZER to the GLM5.2 tokenizer HF id or local path}"
RM_URL="${RM_URL:?set RM_URL to the GLM5.2 SGLang server, e.g. http://glm52-teacher:30000/generate}"
STUDENT_TORCH_DIST="${STUDENT_TORCH_DIST:-${STUDENT_MODEL%/}_torch_dist}"
DATA_PATH="${DATA_PATH:-/workspace/dapo-math-17k/dapo-math-17k.jsonl}"
OPD_KL_COEF="${OPD_KL_COEF:-1.0}"
NUM_ROLLOUT="${NUM_ROLLOUT:-1000}"
CANDIDATE_K="${CANDIDATE_K:-20}"
MAX_CONT_LEN="${MAX_CONT_LEN:-4}"
MEGATRON_PATH="${MEGATRON_PATH:-/root/Megatron-LM}"

NVLINK_COUNT=$(nvidia-smi topo -m 2>/dev/null | grep -o 'NV[0-9][0-9]*' | wc -l)
if [ "$NVLINK_COUNT" -gt 0 ]; then HAS_NVLINK=1; else HAS_NVLINK=0; fi

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" &>/dev/null && pwd)"
REPO_ROOT="$(cd -- "${SCRIPT_DIR}/../.." &>/dev/null && pwd)"
source "${REPO_ROOT}/scripts/models/qwen3-8B.sh"

# Build the student torch_dist if missing (overlaps with teacher bring-up).
if [ ! -d "${STUDENT_TORCH_DIST}" ]; then
    echo "Building student torch_dist at ${STUDENT_TORCH_DIST} ..."
    PYTHONPATH="${MEGATRON_PATH}" python3 tools/convert_hf_to_torch_dist.py \
        "${MODEL_ARGS[@]}" \
        --hf-checkpoint "${STUDENT_MODEL}" \
        --save "${STUDENT_TORCH_DIST}"
fi

TEACHER_HEALTH="${RM_URL%/generate}/health_generate"
echo "Waiting for the GLM5.2 teacher server at ${TEACHER_HEALTH} ..."
until curl -sf "${TEACHER_HEALTH}" >/dev/null; do
    echo "  teacher not ready yet; retrying in 10s..."
    sleep 10
done
echo "GLM5.2 teacher server is up."

CKPT_ARGS=(
   --hf-checkpoint "${STUDENT_MODEL}"
   --ref-load "${STUDENT_TORCH_DIST}"
   --load "${SAVE_DIR:-/weka/checkpoints/qwen3-8B-glm5.2-simct}"
   --save "${SAVE_DIR:-/weka/checkpoints/qwen3-8B-glm5.2-simct}"
   --save-interval 50
)

ROLLOUT_ARGS=(
   --prompt-data "${DATA_PATH}"
   --input-key prompt
   --label-key label
   --apply-chat-template
   --rollout-shuffle
   --num-rollout "${NUM_ROLLOUT}"
   --rollout-batch-size 16
   --n-samples-per-prompt 8
   --rollout-max-response-len 8192
   --rollout-temperature 1.0

   --global-batch-size 128
   --balance-data
)

EVAL_ARGS=(
   # --eval-interval 50
   # --eval-prompt-data aime /workspace/miles-smoke/aime-2024/aime-2024.jsonl
   # --n-samples-per-eval-prompt 16
   # --eval-max-response-len 16384
)

# Dense Qwen3-8B: tensor-parallel 2, no expert parallelism.
PERF_ARGS=(
   --tensor-model-parallel-size 2
   --sequence-parallel
   --pipeline-model-parallel-size 1
   --context-parallel-size 1

   --recompute-granularity full
   --recompute-method uniform
   --recompute-num-layers 1

   --use-dynamic-batch-size
   --max-tokens-per-gpu 16384
)

# SimCT cross-tokenizer OPD: pure distillation (task reward 0; the learning signal
# is the per-token reverse-KL over the common text-unit supervision space).
GRPO_ARGS=(
   --advantage-estimator grpo
   --use-opd
   --opd-type sglang
   --opd-kl-coef "${OPD_KL_COEF}"
   --opd-log-prob-top-k 0
   --opd-teacher-tokenizer "${TEACHER_TOKENIZER}"
   --opd-ct-method simct
   --opd-ct-candidate-k "${CANDIDATE_K}"
   --opd-ct-max-continuation-len "${MAX_CONT_LEN}"
   --use-kl-loss
   --kl-loss-coef 0.00
   --kl-loss-type low_var_kl
   --entropy-coef 0.00
)

RM_ARGS=(
   --custom-rm-path miles.rollout.simct_opd.reward_func
   --custom-reward-post-process-path miles.rollout.simct_opd.post_process_rewards
   --rm-url "${RM_URL}"
)

OPTIMIZER_ARGS=(
   --optimizer adam
   --lr 1e-6
   --lr-decay-style constant
   --weight-decay 0.1
   --adam-beta1 0.9
   --adam-beta2 0.98
)

WANDB_ARGS=()
if [ -n "${WANDB_API_KEY:-}" ]; then
   WANDB_ARGS=(
      --use-wandb
      --wandb-project "${WANDB_PROJECT:-osmosis-testing}"
      --wandb-group "${WANDB_GROUP:-qwen3-8B-glm5.2-simct}"
   )
fi

SGLANG_ARGS=(
   --rollout-num-gpus-per-engine 2
   --sglang-mem-fraction-static 0.5
)

MISC_ARGS=(
   --attention-dropout 0.0
   --hidden-dropout 0.0
   --accumulate-allreduce-grads-in-fp32
   --attention-softmax-in-fp32
   --attention-backend flash
)

export MASTER_ADDR=${MASTER_ADDR:-"127.0.0.1"}
ray start --head --node-ip-address ${MASTER_ADDR} --num-gpus 8 --disable-usage-stats --dashboard-host=0.0.0.0 --dashboard-port=8265

RUNTIME_ENV_JSON="{
  \"env_vars\": {
    \"PYTHONPATH\": \"${MEGATRON_PATH}/\",
    \"CUDA_DEVICE_MAX_CONNECTIONS\": \"1\",
    \"NCCL_NVLS_ENABLE\": \"${HAS_NVLINK}\"
  }
}"

ray job submit --address="http://127.0.0.1:8265" \
   --runtime-env-json="${RUNTIME_ENV_JSON}" \
   -- python3 train.py \
   --actor-num-nodes 1 \
   --actor-num-gpus-per-node 8 \
   --colocate \
   "${MODEL_ARGS[@]}" \
   "${CKPT_ARGS[@]}" \
   "${ROLLOUT_ARGS[@]}" \
   "${OPTIMIZER_ARGS[@]}" \
   "${GRPO_ARGS[@]}" \
   "${RM_ARGS[@]}" \
   "${WANDB_ARGS[@]}" \
   "${PERF_ARGS[@]}" \
   "${EVAL_ARGS[@]}" \
   "${SGLANG_ARGS[@]}" \
   "${MISC_ARGS[@]}"

pkill -9 sglang || true
sleep 3
ray stop --force || true
pkill -9 ray || true
pkill -9 python || true
