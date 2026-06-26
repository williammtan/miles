#!/bin/bash
#
# POC: SimCT cross-tokenizer on-policy distillation, COLOCATED on a SINGLE node.
#   teacher  ~32B Llama-family model (different tokenizer than Qwen)
#   student  Qwen3-8B
#
# Unlike the GLM5.2 (744B) setup, a ~32B teacher and an 8B student co-fit on ONE
# 8xH200 node, so this script launches BOTH on the same node: the teacher as a
# local SGLang server on a GPU slice, and the student trainer+rollout (colocated)
# on the remaining GPUs. It is a wiring/throughput proof-of-concept for the SimCT
# reward path (miles.rollout.simct_opd), not a tuned training recipe.
#
# Required env:
#   TEACHER_MODEL      HF id or local dir of the ~32B teacher. Llama has no exact 32B
#                      checkpoint, so set this to your chosen Llama-family 32B (any
#                      non-Qwen tokenizer exercises the cross-tokenizer path).
# Optional env:
#   TEACHER_TOKENIZER  teacher tokenizer (default: TEACHER_MODEL)
#   STUDENT_MODEL      Qwen3-8B HF id or dir (default: /workspace/Qwen3-8B)
#   TEACHER_GPUS       GPU ids for the teacher SGLang server (default: "6,7" -> tp2)
#   STUDENT_GPUS       GPU ids for the student (default: "0,1,2,3,4,5" -> 6 GPUs)
#   DATA_PATH          prompt jsonl (default: /workspace/dapo-math-17k/dapo-math-17k.jsonl)
#   NUM_ROLLOUT        rollout steps (default: 3 -- POC smoke)
#   CANDIDATE_K        SimCT top-k per side (default: 20)
#   MAX_CONT_LEN       SimCT max continuation tokens (default: 4)
#   OPD_KL_COEF        distillation strength (default: 1.0)
#   MEGATRON_PATH      Megatron-LM path (default: /root/Megatron-LM)
#
# usage: TEACHER_MODEL=<32b-llama> bash examples/on_policy_distillation/run-llama32b-qwen3-8B-simct-poc.sh

set -ex

pkill -9 sglang || true
sleep 3
ray stop --force || true
pkill -9 ray || true
pkill -9 python || true
sleep 3

export PYTHONBUFFERED=16

TEACHER_MODEL="${TEACHER_MODEL:?set TEACHER_MODEL to a ~32B Llama-family teacher (HF id or local path)}"
TEACHER_TOKENIZER="${TEACHER_TOKENIZER:-${TEACHER_MODEL}}"
STUDENT_MODEL="${STUDENT_MODEL:-/workspace/Qwen3-8B}"
STUDENT_TORCH_DIST="${STUDENT_TORCH_DIST:-${STUDENT_MODEL%/}_torch_dist}"
TEACHER_GPUS="${TEACHER_GPUS:-6,7}"
STUDENT_GPUS="${STUDENT_GPUS:-0,1,2,3,4,5}"
DATA_PATH="${DATA_PATH:-/workspace/dapo-math-17k/dapo-math-17k.jsonl}"
NUM_ROLLOUT="${NUM_ROLLOUT:-3}"
CANDIDATE_K="${CANDIDATE_K:-20}"
MAX_CONT_LEN="${MAX_CONT_LEN:-4}"
OPD_KL_COEF="${OPD_KL_COEF:-1.0}"
MEGATRON_PATH="${MEGATRON_PATH:-/root/Megatron-LM}"
TEACHER_PORT="${TEACHER_PORT:-30000}"
RM_URL="http://127.0.0.1:${TEACHER_PORT}/generate"

NUM_STUDENT_GPUS=$(awk -F, '{print NF}' <<<"${STUDENT_GPUS}")
NUM_TEACHER_GPUS=$(awk -F, '{print NF}' <<<"${TEACHER_GPUS}")

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" &>/dev/null && pwd)"
REPO_ROOT="$(cd -- "${SCRIPT_DIR}/../.." &>/dev/null && pwd)"
source "${REPO_ROOT}/scripts/models/qwen3-8B.sh"

# --- 1. Launch the teacher SGLang server on its GPU slice (background) ---------
LOG_FILE="/tmp/sglang_teacher_$(head /dev/urandom | tr -dc A-Za-z0-9 | head -c 6).log"
CUDA_VISIBLE_DEVICES="${TEACHER_GPUS}" python3 -m sglang.launch_server \
    --model-path "${TEACHER_MODEL}" \
    --tp "${NUM_TEACHER_GPUS}" \
    --host 0.0.0.0 \
    --port "${TEACHER_PORT}" \
    --mem-fraction-static 0.85 \
    --chunked-prefill-size 8192 \
    --trust-remote-code \
    > "${LOG_FILE}" 2>&1 &

echo "Starting teacher (${TEACHER_MODEL}) on GPUs ${TEACHER_GPUS}..."
until curl -sf "http://127.0.0.1:${TEACHER_PORT}/health_generate" >/dev/null; do
    echo "  teacher not ready; tail of log:"; tail -n 5 "${LOG_FILE}" || true
    sleep 10
done
echo "Teacher server is up at ${RM_URL}."

# --- 2. Build the student torch_dist if missing -------------------------------
if [ ! -d "${STUDENT_TORCH_DIST}" ]; then
    echo "Building student torch_dist at ${STUDENT_TORCH_DIST} ..."
    PYTHONPATH="${MEGATRON_PATH}" python3 tools/convert_hf_to_torch_dist.py \
        "${MODEL_ARGS[@]}" \
        --hf-checkpoint "${STUDENT_MODEL}" \
        --save "${STUDENT_TORCH_DIST}"
fi

CKPT_ARGS=(
   --hf-checkpoint "${STUDENT_MODEL}"
   --ref-load "${STUDENT_TORCH_DIST}"
   --load "${SAVE_DIR:-/weka/checkpoints/qwen3-8B-llama32b-simct-poc}"
   --save "${SAVE_DIR:-/weka/checkpoints/qwen3-8B-llama32b-simct-poc}"
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

# --- 3. Launch the student trainer+rollout (colocated) on its GPU slice --------
export CUDA_VISIBLE_DEVICES="${STUDENT_GPUS}"
export MASTER_ADDR=${MASTER_ADDR:-"127.0.0.1"}
ray start --head --node-ip-address "${MASTER_ADDR}" --num-gpus "${NUM_STUDENT_GPUS}" \
    --disable-usage-stats --dashboard-host=0.0.0.0 --dashboard-port=8265

RUNTIME_ENV_JSON="{
  \"env_vars\": {
    \"PYTHONPATH\": \"${MEGATRON_PATH}/\",
    \"CUDA_DEVICE_MAX_CONNECTIONS\": \"1\",
    \"CUDA_VISIBLE_DEVICES\": \"${STUDENT_GPUS}\"
  }
}"

ray job submit --address="http://127.0.0.1:8265" \
   --runtime-env-json="${RUNTIME_ENV_JSON}" \
   -- python3 train.py \
   --actor-num-nodes 1 \
   --actor-num-gpus-per-node "${NUM_STUDENT_GPUS}" \
   --colocate \
   "${MODEL_ARGS[@]}" \
   "${CKPT_ARGS[@]}" \
   "${ROLLOUT_ARGS[@]}" \
   "${OPTIMIZER_ARGS[@]}" \
   "${GRPO_ARGS[@]}" \
   "${RM_ARGS[@]}" \
   "${PERF_ARGS[@]}" \
   "${SGLANG_ARGS[@]}" \
   "${MISC_ARGS[@]}"

# clean up
pkill -9 sglang || true
sleep 3
ray stop --force || true
pkill -9 ray || true
pkill -9 python || true
