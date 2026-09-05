"""Add sparse prefill scoring and exact image token preservation to Miles SGLang.

This patches existing files in place with exact anchors; it never installs or
replaces a package. Original files are retained as .ptd-po-backup. Reapplication
is a no-op. Run before starting SGLang engines on every rollout node.
"""

import argparse
import importlib.util
from pathlib import Path


MARKER = "# MILES_PTD_SPARSE_V1"
MM_MARKER = "# MILES_PTD_EXACT_MM_TOKENS_V1"
SPARSE_SOURCES = ("managers/io_struct.py", "managers/tokenizer_manager.py", "managers/schedule_batch.py",
                  "layers/logprob_processor.py")
MM_SOURCE = "multimodal/processors/qwen_vl.py"

_MM_HELPER = '''    def _ptd_preserve_image_input(self, original_ids, input_ids, mm_items):
        """Restore exact expanded PTD tokens before padding and mRoPE construction."""
        # MILES_PTD_EXACT_MM_TOKENS_V1
        if not isinstance(original_ids, list) or any(type(token) is not int for token in original_ids):
            raise ValueError("PTD image scoring requires the original integer input_ids")
        if not mm_items or any(not item.is_image() for item in mm_items):
            raise ValueError("PTD exact multimodal scoring currently supports images only")
        spans = []
        for position, token in enumerate(original_ids):
            if token != self.mm_tokens.image_token_id:
                continue
            if spans and spans[-1][1] == position - 1:
                spans[-1] = (spans[-1][0], position)
            else:
                spans.append((position, position))
        if len(spans) != len(mm_items):
            raise ValueError("PTD expanded image token spans do not match the processed images")
        restored_items = []
        for span, item in zip(spans, mm_items, strict=True):
            if (item.offsets is None or len(item.offsets) != 1
                    or span[1] - span[0] != item.offsets[0][1] - item.offsets[0][0]):
                raise ValueError("PTD expanded image token count differs from the image processor")
            # Feature tensors may be cached; never mutate a cached item's offsets.
            restored = copy.copy(item)
            restored.offsets = [span]
            restored_items.append(restored)
        return input_ids.new_tensor(original_ids), restored_items

'''


def _replace(text, old, new):
    count = text.count(old)
    if count != 1:
        raise ValueError(f"Unsupported SGLang version: expected one patch anchor, found {count}: {old[:100]!r}")
    return text.replace(old, new, 1)


def patch_sparse_sources(sources):
    if all(MARKER in source for source in sources.values()):
        return sources
    if any(MARKER in source for source in sources.values()):
        raise ValueError("Partial PTD patch detected; restore .ptd-po-backup files before retrying")
    result = dict(sources)
    name = "managers/io_struct.py"
    result[name] = _replace(result[name],
        "    token_ids_logprob: Optional[Union[List[List[int]], List[int]]] = None",
        "    " + MARKER + "\n"
        "    token_ids_logprob_positions: Optional[List[List[int]]] = None\n"
        "    token_ids_logprob: Optional[Union[List[List[int]], List[int]]] = None")

    name = "managers/tokenizer_manager.py"
    result[name] = _replace(result[name],
        "    def _validate_token_ids_logprob(self, obj: GenerateReqInput) -> None:\n",
        "    def _validate_token_ids_logprob(self, obj: GenerateReqInput) -> None:\n"
        "        " + MARKER + "\n"
        "        positions = obj.token_ids_logprob_positions\n"
        "        if positions is not None:\n"
        "            if obj.token_ids_logprob is not None:\n"
        "                raise ValueError('Use only one token_ids_logprob mode')\n"
        "            if not obj.input_ids or not isinstance(obj.input_ids[0], int):\n"
        "                raise ValueError('Per-position scoring requires a single input_ids request')\n"
        "            if obj.sampling_params.get('max_new_tokens') != 0 or not obj.return_logprob:\n"
        "                raise ValueError('Per-position scoring requires max_new_tokens=0 and return_logprob')\n"
        "            if obj.logprob_start_len < 0 or len(positions) != len(obj.input_ids) - obj.logprob_start_len:\n"
        "                raise ValueError('Per-position rows must align with logprob_start_len, including placeholder')\n"
        "            if positions[0]:\n"
        "                raise ValueError('The initial placeholder must have an empty ID list')\n"
        "            for row in positions:\n"
        "                if not isinstance(row, list) or any(type(i) is not int or i < 0 or i >= self.model_config.vocab_size for i in row):\n"
        "                    raise ValueError('Invalid per-position vocabulary IDs')\n"
        "            obj.token_ids_logprob = {'ptd_positions': positions}\n"
        "            return\n")

    name = "managers/schedule_batch.py"
    result[name] = _replace(result[name],
        "            self.token_ids_logprobs = [r.logprob.token_ids_logprob for r in reqs]\n",
        "            self.token_ids_logprobs = [r.logprob.token_ids_logprob for r in reqs]\n"
        "            " + MARKER + "\n"
        "            for i, req in enumerate(reqs):\n"
        "                spec = self.token_ids_logprobs[i]\n"
        "                if isinstance(spec, dict) and 'ptd_positions' in spec:\n"
        "                    # A logit at absolute position j predicts input token j+1.\n"
        "                    start = prefix_lens[i] + extend_logprob_start_lens[i] - req.logprob_start_len + 1\n"
        "                    count = extend_lens[i] - extend_logprob_start_lens[i]\n"
        "                    rows = spec['ptd_positions'][start:start + count]\n"
        "                    self.token_ids_logprobs[i] = {'ptd_positions': rows + [[]] * (count - len(rows))}\n")

    name = "layers/logprob_processor.py"
    result[name] = _replace(result[name],
        "            if token_ids is not None:\n                row = logprobs[pt + j, token_ids]\n",
        "            if token_ids is not None:\n"
        "                " + MARKER + "\n"
        "                selected_ids = token_ids['ptd_positions'][split_pruned_len + j] if isinstance(token_ids, dict) else token_ids\n"
        "                row = logprobs[pt + j, selected_ids]\n")
    result[name] = _replace(result[name],
        "                val.append(row.tolist())\n                idx.append(token_ids)\n",
        "                val.append(row.tolist())\n                idx.append(selected_ids)\n")
    result[name] = _replace(result[name],
        "    vals, idxs = [], []\n    if stage == LogprobStage.DECODE:\n",
        "    vals, idxs = [], []\n"
        "    if stage == LogprobStage.DECODE:\n"
        "        token_ids_logprobs_list = [[] if isinstance(ids, dict) else ids for ids in token_ids_logprobs_list]\n")
    result[name] = _replace(result[name],
        "            token_ids_tensor = torch.tensor(token_ids, dtype=torch.long).to(\n"
        "                logprobs.device, non_blocking=True\n"
        "            )\n"
        "            pos_logprobs = logprobs[pt : pt + pruned_len, token_ids_tensor]\n",
        "            if isinstance(token_ids, dict):\n"
        "                rows = token_ids['ptd_positions']\n"
        "                selected = [logprobs[pt + j, row] for j, row in enumerate(rows)]\n"
        "                vals.append(selected if no_copy_to_cpu else [row.tolist() for row in selected])\n"
        "                idxs.append(rows)\n"
        "                pt += pruned_len\n"
        "                continue\n"
        "            token_ids_tensor = torch.tensor(token_ids, dtype=torch.long).to(\n"
        "                logprobs.device, non_blocking=True\n"
        "            )\n"
        "            pos_logprobs = logprobs[pt : pt + pruned_len, token_ids_tensor]\n")
    result[name] = _replace(result[name],
        "    batch_size = len(token_ids_logprobs)\n",
        "    token_ids_logprobs = [[] if isinstance(ids, dict) else ids for ids in token_ids_logprobs]\n"
        "    batch_size = len(token_ids_logprobs)\n")
    for name, source in result.items():
        compile(source, name, "exec")
    return result


def patch_mm_source(source):
    if MM_MARKER in source:
        return source
    source = _replace(source, "import math\n", "import copy\nimport math\n")
    source = _replace(source, "    async def process_mm_data_async(\n",
                      _MM_HELPER + "    async def process_mm_data_async(\n")
    source = _replace(source,
        "        self._mark_dp_encoder_features_for_deferred_reconstruction(mm_items)\n",
        "        ptd_exact_ids = getattr(request_obj, 'token_ids_logprob_positions', None) is not None\n"
        "        if ptd_exact_ids:\n"
        "            input_ids, mm_items = self._ptd_preserve_image_input(\n"
        "                request_obj.input_ids, input_ids, mm_items\n"
        "            )\n"
        "            base_output.input_ids = request_obj.input_ids\n"
        "        self._mark_dp_encoder_features_for_deferred_reconstruction(mm_items)\n")
    source = _replace(source,
        '        padded_input_ids = self._get_processor_output_value(ret, "padded_input_ids")\n',
        '        padded_input_ids = None if ptd_exact_ids else self._get_processor_output_value(ret, "padded_input_ids")\n')
    source = _replace(source,
        "        mrope_result = self._get_precomputed_mrope_from_output(ret)\n",
        "        mrope_result = None if ptd_exact_ids else self._get_precomputed_mrope_from_output(ret)\n")
    compile(source, MM_SOURCE, "exec")
    return source


def patch_sources(sources):
    # Upgrade sparse-v1 installations in place: their original four files remain
    # untouched while Qwen gains exact input-token preservation.
    result = {**sources, **patch_sparse_sources({name: sources[name] for name in SPARSE_SOURCES})}
    result[MM_SOURCE] = patch_mm_source(sources[MM_SOURCE])
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sglang-root", type=Path)
    args = parser.parse_args()
    root = args.sglang_root
    if root is None:
        root = Path(importlib.util.find_spec("sglang").origin).parent / "srt"
    names = (*SPARSE_SOURCES, MM_SOURCE)
    sources = {name: (root / name).read_text() for name in names}
    patched = patch_sources(sources)
    for name, source in patched.items():
        if source == sources[name]:
            continue
        path = root / name
        backup = path.with_suffix(path.suffix + ".ptd-po-backup")
        if backup.exists():
            raise ValueError(f"Refusing to overwrite original backup {backup}")
        backup.write_text(sources[name])
        path.write_text(source)
    print("SGLang PTD sparse scoring and exact image-token patches installed (or already present).")


if __name__ == "__main__":
    main()
