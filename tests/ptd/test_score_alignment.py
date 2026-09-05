import pytest

from miles.rollout.ptd import extract_score_rows


def _response(second_row):
    return {"meta_info": {
        "input_token_logprobs": [[None, 8], [-1.0, 2], second_row],
        "input_top_logprobs": [None, [[-1.0, 2]], [[-1.0, 3]]],
    }}


def test_alignment_reports_first_numeric_mismatch():
    with pytest.raises(ValueError) as error:
        extract_score_rows(_response([-1.0, 0]), {"response_tokens": [2, 3]}, "input_top_logprobs")
    assert "response_index=1, expected_token_id=3, actual_token_id=0" in str(error.value)
    assert "actual_type=int, response_length=2" in str(error.value)


@pytest.mark.parametrize("bad_id", ["private-returned-text", True, 3.0, None])
def test_alignment_rejects_noninteger_ids_without_logging_returned_content(bad_id):
    with pytest.raises(ValueError) as error:
        extract_score_rows(_response([-1.0, bad_id]), {"response_tokens": [2, 3]}, "input_top_logprobs")
    message = str(error.value)
    assert "response_index=1, expected_token_id=3, actual_token_id=None" in message
    assert f"actual_type={type(bad_id).__name__}" in message
    assert "private-returned-text" not in message


@pytest.mark.parametrize("bad_row", [None, [], [-1.0], "private-returned-text"])
def test_alignment_rejects_malformed_rows_without_logging_returned_content(bad_row):
    with pytest.raises(ValueError, match="malformed response score row at response_index=1") as error:
        extract_score_rows(_response(bad_row), {"response_tokens": [2, 3]}, "input_top_logprobs")
    assert "private-returned-text" not in str(error.value)


def test_aligned_response_is_unchanged():
    response = _response([-1.0, 3])
    assert extract_score_rows(response, {"response_tokens": [2, 3]}, "input_top_logprobs") == [
        [[-1.0, 2]], [[-1.0, 3]],
    ]
