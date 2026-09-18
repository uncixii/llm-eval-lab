from __future__ import annotations

import json
import re


def normalized_tokens(text: str) -> set[str]:
    return set(re.findall(r"[A-Za-z0-9_]+|[\u4e00-\u9fff]", text.lower()))


def token_coverage(response: str, reference: str) -> float:
    reference_tokens = normalized_tokens(reference)
    response_tokens = normalized_tokens(response)
    if not reference_tokens:
        return 1.0
    return len(reference_tokens & response_tokens) / len(reference_tokens)


def contains_citation(response: str) -> bool:
    return bool(re.search(r"\[[^\]]+\]|来源|依据|文档", response))


def is_valid_json(response: str) -> bool:
    try:
        json.loads(response)
        return True
    except (TypeError, json.JSONDecodeError):
        return False
