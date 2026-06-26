import math
import string
from argparse import Namespace

import pytest
from tests.ci.ci_register import register_cpu_ci

import miles.rollout.simct_opd as simct
from miles.rollout.simct_opd import (
    _align_positions,
    _byte_boundaries,
    _continuation_text,
    _realize,
    _reverse_kl,
    _simct_reverse_kls,
    post_process_rewards,
    reward_func,
)
from miles.utils.types import Sample

register_cpu_ci(est_time=60, suite="stage-a-cpu")


class FakeTokenizer:
    """Char-additive greedy tokenizer for deterministic CPU tests."""

    def __init__(self, merges):
        vocab = {ch: i for i, ch in enumerate(string.ascii_letters + " !.,")}
        for m in merges:
            vocab.setdefault(m, len(vocab))
        self.str_to_id = vocab
        self.id_to_str = {i: s for s, i in vocab.items()}
        self.units = sorted(vocab, key=len, reverse=True)

    def decode(self, ids, skip_special_tokens=False):
        return "".join(self.id_to_str[i] for i in ids)

    def encode(self, text, add_special_tokens=False):
        ids = []
        i = 0
        while i < len(text):
            for u in self.units:
                if u and text.startswith(u, i):
                    ids.append(self.str_to_id[u])
                    i += len(u)
                    break
            else:
                raise ValueError(f"cannot tokenize {text[i:]!r}")
        return ids

    def id(self, s):
        return self.str_to_id[s]


def _entry(prob, token_id):
    return [math.log(prob), token_id]


# --------------------------------------------------------------------------- #
# Pure helpers
# --------------------------------------------------------------------------- #


def test_byte_boundaries_cumulative():
    tok = FakeTokenizer(["ha", "pp"])
    assert _byte_boundaries(tok, tok.encode("happy")) == [0, 2, 4, 5]


def test_align_positions_happy_word():
    s = FakeTokenizer(["ha", "pp"])  # happy -> ha|pp|y
    t = FakeTokenizer(["hap", "py"])  # happy -> hap|py
    assert _align_positions(s.encode("happy"), t.encode("happy"), s, t) == {0: 0}


def test_align_positions_shared_trailing_boundary():
    s = FakeTokenizer(["ha", "pp"])  # happy! -> ha|pp|y|!
    t = FakeTokenizer(["hap", "py"])  # happy! -> hap|py|!
    assert _align_positions(s.encode("happy!"), t.encode("happy!"), s, t) == {0: 0, 3: 2}


def test_continuation_text_diffs_cumulative_decode():
    tok = FakeTokenizer(["ha"])
    assert _continuation_text(tok, [], tok.id("ha")) == "ha"
    assert _continuation_text(tok, tok.encode("ha"), tok.id("p")) == "p"


def test_realize_returns_continuation_tokens():
    tok = FakeTokenizer(["hap"])
    assert tok.decode(_realize(tok, [], "ha", "")) == "ha"


def test_reverse_kl_zero_when_equal():
    s = {"a": math.log(0.6), "b": math.log(0.4)}
    assert _reverse_kl(s, dict(s)) == pytest.approx(0.0)


def test_reverse_kl_matches_manual():
    s_s = {"a": math.log(0.6), "b": math.log(0.4)}
    s_t = {"a": math.log(0.3), "b": math.log(0.7)}
    expected = 0.6 * math.log(0.6 / 0.3) + 0.4 * math.log(0.4 / 0.7)
    assert _reverse_kl(s_s, s_t) == pytest.approx(expected)


def test_reverse_kl_singleton_is_zero():
    assert _reverse_kl({"a": 0.0}, {"a": 0.0}) == 0.0


def test_simct_reverse_kls_masks_non_aligned():
    positions = [{"t": 1, "scores_s": {"a": math.log(0.6), "b": math.log(0.4)},
                  "scores_t": {"a": math.log(0.3), "b": math.log(0.7)}}]
    out = _simct_reverse_kls(positions, response_length=3)
    assert len(out) == 3 and out[0] == 0.0 and out[2] == 0.0 and out[1] > 0.0


# --------------------------------------------------------------------------- #
# reward_func / post_process integration with stubbed tokenizers + HTTP
# --------------------------------------------------------------------------- #


@pytest.mark.asyncio
async def test_reward_func_end_to_end(monkeypatch):
    student = FakeTokenizer(["ha", "pp", "he"])
    teacher = FakeTokenizer(["hap", "py", "he"])

    monkeypatch.setattr(simct, "_student_tokenizer", lambda args: student)
    monkeypatch.setattr(simct, "_teacher_tokenizer", lambda args: teacher)
    monkeypatch.setattr(simct, "_teacher_prompt_ids", lambda tok, raw, tools: [])

    args = Namespace(
        rm_url="http://teacher/generate",
        sglang_router_ip="127.0.0.1",
        sglang_router_port=1234,
        opd_ct_candidate_k=20,
        opd_ct_max_continuation_len=4,
    )

    response_text = "happy"
    s_ids = student.encode(response_text)  # ha|pp|y -> response_length 3
    sample = Sample(prompt="", tokens=list(s_ids), response_length=len(s_ids), response=response_text)
    sample.metadata["opd_student_top_logprobs"] = [
        [_entry(0.6, student.id("ha")), _entry(0.4, student.id("he"))],
        [_entry(0.9, student.id("pp")), _entry(0.1, student.id("p"))],
        [_entry(0.9, student.id("y")), _entry(0.1, student.id("z"))],
    ]

    async def fake_post(url, payload):
        if "top_logprobs_num" in payload:
            return {
                "meta_info": {
                    "input_top_logprobs": [
                        None,
                        [_entry(0.7, teacher.id("hap")), _entry(0.3, teacher.id("he"))],
                        [_entry(0.8, teacher.id("py")), _entry(0.2, teacher.id("p"))],
                    ]
                }
            }
        cont = payload["input_ids"][payload["logprob_start_len"] + 1 :]
        return {"meta_info": {"input_token_logprobs": [[0.0, 0]] + [[math.log(0.5), 0] for _ in cont]}}

    monkeypatch.setattr(simct, "_post_json", fake_post)

    out = await reward_func(args, sample)
    rkl = out["ctopd_reverse_kl"]
    assert len(rkl) == sample.response_length
    assert rkl[1] == 0.0 and rkl[2] == 0.0
    assert math.isfinite(rkl[0]) and rkl[0] >= 0.0

    sample.reward = out
    rewards, _ = post_process_rewards(args, [sample])
    assert rewards == [0.0]
    assert sample.opd_reverse_kl.shape[0] == sample.response_length
    sample.validate()
