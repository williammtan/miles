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

After an online hint is accepted, the rollout phase asks the frozen teacher for
its Top-K distribution once over the exact sampled continuation. These fixed
targets travel with the rollout. During optimization, the current student's
Top-K is computed locally. Teacher probabilities for student IDs absent from
the stored teacher Top-K use a uniform share of the teacher tail, and all mass
outside the student Top-K becomes one residual bucket. The true JSD is
differentiated through student probabilities; teacher values are detached. TP
ranks normalize over the complete vocabulary and communicate sparse support.
Vocabulary workspaces are chunked and recomputed for backward. No HTTP request
occurs inside forward or backward.

The tutoring reduction is the sum over selected response tokens divided by the
selected token count over the entire optimizer step, including DP/microbatches.
It is independent of GRPO's denominator and is never multiplied by the
group-relative correctness advantage.

## Deliberate source differences

The pinned official `verl/trainer/core_algos.py` has two relevant discrepancies:

1. `_topk_match_and_gather` approximates missing teacher values by a uniform tail
   using a hardcoded vocabulary of 150000. This implementation uses the same
   released-code approximation with the configured vocabulary (248320 for
   Handshake), FP32 tail arithmetic, chunked matching, and a residual bucket.
   The paper's equations instead use the union with exact cross-probabilities;
   this is an explicit throughput-oriented objective approximation.
2. The upstream `jsd_kl` branch detaches the student weights but differentiates
   the mixture, canceling the intended student gradient away from clamp bounds.
   `tests/ptd/reproduce_upstream_jsd.py` isolates the issue on the same support
   and probabilities. We follow the paper's mathematical JSD instead. This is
   not a bitwise reproduction of the released training script.

The user requires **online hints conditioned on this student's
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
Its Qwen preprocessing extension also preserves exact, already-expanded image
prompt and response IDs. It supports image-only PTD contexts and rejects image
span/count mismatches. Text-only scoring does not use this preprocessing path.

Required arguments:

```
--ptd-coef 0.05 --ptd-top-k 100 --ptd-vocab-size 248320
--ptd-score-mode precomputed_teacher_topk_tail_v1 --ptd-score-concurrency 16
--ptd-hint-function-path integrations.handshake.ptd_hint.generate_hint
--use-miles-router
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

The pinned SGLang Rust router (0.3.2) reserializes `/generate` and drops both
`token_ids_logprob_positions` and `cache_salt`. A CPU echo-worker probe reproduced
this after training failed with missing cross-probabilities despite direct-engine
checks passing. PTD therefore requires the existing Miles byte-preserving proxy
when sharing the rollout router; startup rejects the incompatible default. An
explicit `--ptd-teacher-url` must reach a patched engine directly or use a proxy
that preserves both fields. Live checks must cover the actual production proxy.
MilesRouter also honors `--router-disable-health-check`, so that flag does not
silently leave Python-router worker quarantine active during long prefills.
Automatic worker replacement requires a `/remove_worker` route that MilesRouter
does not currently implement; this recipe uses a fixed synchronous worker fleet.

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
full training run. Teacher Top-K is computed once for every accepted exact
rollout/hint pair and is immutable during the update, matching the official
implementation's execution order. Empty per-position sparse rows still activate
the patched exact-ID multimodal path. Old traces containing empty score tensors
are rejected instead of silently disabling PTD. The scoring mode and concurrency
are included in checkpoint identity, so a checkpoint from the earlier joint-RPC
objective cannot be resumed as this experiment.

A longer real response exposed a 9.10937 log-probability cold/warm difference.
Each logical teacher request uses a fresh `cache_salt`, and retries get another
salt, so the stored teacher forward cannot reuse hybrid prefix state from an
earlier request. Within-request chunk continuation and image-feature caching
remain available.

A later smoke batch exposed a separate Qwen preprocessing issue: supplying
`input_ids` still decoded and retokenized them when an image was present. The
actual sampled pair `[469, 26000]` became `[14944, 334]` at response offsets 6–7,
with unchanged total length, and strict response-ID validation stopped training.
The additive Qwen patch now restores all original IDs before model input
construction. It validates expanded image spans against processed image sizes,
copies media items before updating their offsets, and recomputes padded IDs and
mRoPE from the exact sequence. Existing cached image features remain reusable.
The patch applies only to PTD per-position scoring; ordinary generation follows
the existing path. Regression tests execute the patched deployed Qwen method
for equal-length and unequal-length retokenization, moved image spans, and
invalid media alignment. Numeric mismatch diagnostics retain strict validation
without printing response text.
