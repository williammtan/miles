# PTD-PO for Handshake

Implementation source: PTD-PO paper, https://arxiv.org/pdf/2606.07000, equations
8–17 and appendix D.3. Official reference code was audited at
https://github.com/XszNeverSleep/PTD-PO commit
`9953ea2b5c68aa6454b2fd693d5f5ff0fe34ed97`.

## Scope and objective

The student generates from the original image/question. Every valid incorrect
training response can receive an independent tutoring term. GRPO rewards and
group normalization remain intact. Correct responses, judge failures, rejected
hints, and evaluation receive no tutoring. The coefficient defaults to zero.

At each response position, compute the **current** student's Top-K, then request
the frozen teacher's Top-K and exact probabilities at those student IDs together
in **one teacher forward pass**. Build the union using only that response. The remaining
probability mass is an explicit bucket. The true JSD is differentiated through
student probabilities; teacher values are detached. TP ranks normalize over
the complete vocabulary and communicate sparse support. Vocabulary workspaces
are chunked and recomputed for backward.

The tutoring reduction is the sum over selected response tokens divided by the
selected token count over the entire optimizer step, including DP/microbatches.
It is independent of GRPO's denominator and is never multiplied by the
group-relative correctness advantage.

## Deliberate source differences

The pinned official `verl/trainer/core_algos.py` has two relevant discrepancies:

1. `_topk_match_and_gather` approximates missing teacher values by a uniform tail
   using a hardcoded vocabulary of 150000. `compute_topk_kl` uses student-only
   support. The paper specifies the union and residual mass; this implementation
   uses exact cross-scores and the configured vocabulary (248320 for Handshake).
2. The upstream `jsd_kl` branch detaches the student weights but differentiates
   the mixture, canceling the intended student gradient away from clamp bounds.
   `tests/ptd/reproduce_upstream_jsd.py` isolates the issue on the same support
   and probabilities. We follow the paper's mathematical JSD instead. This is
   not a bitwise reproduction of the released training script.

The user requires **online GPT-6 Astra hints conditioned on this student's
actual failed rollout**, image, question and reference answer. This differs from
the paper's offline problem-level hints. The callback can cache exact requests
for retry/resume, but cannot reuse a generic hint across different responses.
It returns only the compact hint to Qwen, or None when no usable hint is found.

## Integration

Set `MILES_USE_LEGACY_ROLLOUT_V1=1`. Supported initial configuration is Megatron,
TP>=1, CP=1, THD, per-token policy loss, frozen base teacher with a LoRA student.
Run `tools/patch_sglang_ptd.py` before SGLang startup; this adds per-position
scoring to the existing private runtime without replacing its package. The
patch preserves originals and rejects unknown/partial patch states.

Required arguments:

```
--ptd-coef 0.05 --ptd-top-k 100 --ptd-vocab-size 248320
--ptd-hint-function-path integrations.handshake.ptd_hint.generate_hint
```

The reward integration must mark `sample.metadata['grade_valid']` as True only
for real verdicts, including missing final answers. The teacher uses the exact
generation media payload saved in `ptd_media_payload`, prepends a teacher-only
Qwen system prefix, and preserves all original prompt and response IDs. Returned
response token IDs are checked before accepting teacher distributions. Explicit
`lora_path=None` requests the base, with no student adapter.

Use synchronous `train.py` initially. A separately routed frozen teacher URL is
supported. The shared rollout engine path has not been validated for concurrent
asynchronous rollout aborts; do not treat the synchronous recipe as that claim.

## Verification

```
python -m pytest --confcutdir=tests/ptd tests/ptd/test_ptd.py
PYTHONPATH=. torchrun --standalone --nproc-per-node=2 tests/ptd/check_tp.py --backend gloo
PYTHONPATH=. torchrun --standalone --nproc-per-node=2 tests/ptd/check_tp.py --backend nccl
PYTHONPATH=. python tests/ptd/reproduce_upstream_jsd.py --upstream /path/to/PTD-PO
PYTHONPATH=. python tests/ptd/check_multimodal_score.py --url http://engine:port
```

Seventy CPU checks cover all-wrong nonzero gradients, correct-response zero
gradients, lambda=0 GRPO equivalence, dense reference values/gradients, masks,
first-correct/later-wrong batches, grade validity, unusable hints, stale targets,
EOS/truncation alignment, and global normalization. The implementation agent
reported a two-H200 NCCL pass; the coordinator additionally ran two-rank Gloo
with loss error 1.11e-16 and gradient errors below 4.17e-17.

The independent review found a CP=1 LoRA-provider bug in the baseline: the
provider did not copy `calculate_per_token_loss`, so Megatron averaged per-
microbatch token means. This branch forwards it before finalization and asserts
the final setting when PTD is enabled. A two-process test with unequal
microbatches and a rank with zero selected tokens verifies the actual PTD
normalizers/reduction against dense reference gradients. This changes GRPO
weighting relative to the historical baseline and must be matched in a future
controlled comparison.

The live multimodal score check and independent review are required before the
full training run. Live checks found a 0.37457 log-probability difference between
an initial and later cached teacher request. Its cause is not established. This
motivated joint training-time scoring: neither teacher probabilities nor teacher
Top-K support are reused across forward passes. The online rollout callback
stores only the hint and exact context; empty legacy score fields preserve the
batch codec. Regression tests prove that stale scores/support cannot influence
the loss and that new union IDs and scores are broadcast correctly across TP.

The live joint check verifies overlapping values from both fields of the same
HTTP response, including EOS. Cross-request cold/warm drift is diagnostic only;
each computed JSD uses a coherent q from one response. Completed live and
training results are recorded with the recipe; unit tests alone do not establish
serving correctness.

A longer real response exposed a 9.10937 log-probability cold/warm difference.
Each teacher request now uses a fresh `cache_salt`, isolating its KV and Mamba
prefix state. Repeating that 3,653-token response with K=100 then gave exactly
zero cross-request difference and zero same-forward disagreement. Retries also
get a fresh salt while preserving tokens, media and support. This avoids reusing
a prefix cached by a timed-out attempt. Within-request chunk continuation and
image-feature caching remain available.
