from llm_eval_lab.checks import contains_citation, is_valid_json


def test_deterministic_checks() -> None:
    assert contains_citation("依据 [synthetic-doc]：结论")
    assert is_valid_json('{"score": 1}')
    assert not is_valid_json("not json")
