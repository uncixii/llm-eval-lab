import time

from llm_eval_lab.providers import ProviderResponse, ResilientModelGateway


class FailingProvider:
    name = "failing"

    def complete(self, prompt: str, model: str) -> str:
        raise TimeoutError("synthetic timeout")


class WorkingProvider:
    name = "working"

    def complete(self, prompt: str, model: str) -> str:
        return "synthetic answer"


def test_model_gateway_retries_and_falls_back_between_providers() -> None:
    result = ResilientModelGateway(
        [FailingProvider(), WorkingProvider()],
        retries_per_provider=1,
        backoff_seconds=0,
    ).complete("prompt", "provider/model")
    assert result.provider == "working"
    assert result.attempts == 3
    assert len(result.failures) == 2
    assert len(result.attempt_trace) == 3


def test_model_gateway_tracks_usage_cost_and_latency() -> None:
    class MeteredProvider:
        name = "metered"

        def complete(self, prompt: str, model: str) -> ProviderResponse:
            return ProviderResponse(
                "answer",
                {"prompt_tokens": 1000, "completion_tokens": 500, "total_tokens": 1500},
            )

    result = ResilientModelGateway(
        [MeteredProvider()],
        retries_per_provider=0,
        pricing_per_million_tokens={"provider/model": (2.0, 4.0)},
    ).complete("prompt", "provider/model")
    assert result.usage["total_tokens"] == 1500
    assert result.estimated_cost_usd == 0.004
    assert result.latency_ms >= 0


def test_model_gateway_times_out_and_falls_back() -> None:
    class SlowProvider:
        name = "slow"

        def complete(self, prompt: str, model: str) -> str:
            time.sleep(0.02)
            return "late"

    result = ResilientModelGateway(
        [SlowProvider(), WorkingProvider()],
        retries_per_provider=0,
        timeout_seconds=0.001,
        backoff_seconds=0,
    ).complete("prompt", "provider/model")
    assert result.provider == "working"
    assert "TimeoutError" in result.failures[0]


def test_timeout_workers_are_bounded_and_failures_keep_trace() -> None:
    import threading

    import pytest

    from llm_eval_lab.providers import GatewayError

    release = threading.Event()
    started = threading.Event()
    calls = []

    class Blocked:
        name = "blocked"

        def complete(self, prompt, model):
            calls.append(prompt)
            started.set()
            release.wait(2)
            return "late"

    gateway = ResilientModelGateway(
        [Blocked()], retries_per_provider=0, timeout_seconds=0.01, max_in_flight=1
    )
    try:
        for _ in range(3):
            with pytest.raises(GatewayError) as exc:
                gateway.complete("q", "model")
            assert exc.value.attempt_trace
        assert started.is_set() and len(calls) == 1
        assert "in-flight limit" in exc.value.attempt_trace[0].error
    finally:
        release.set()
        gateway.close()


def test_missing_prices_and_failed_attempts_do_not_claim_complete_cost():
    with ResilientModelGateway([WorkingProvider()], retries_per_provider=0) as gateway:
        result = gateway.complete("q", "unknown-price")
        assert result.estimated_cost_usd is None and not result.cost_complete


def test_usage_from_empty_response_counts_towards_known_cost():
    class Metered:
        name = "metered"

        def __init__(self):
            self.calls = 0

        def complete(self, prompt, model):
            self.calls += 1
            return ProviderResponse(
                "" if self.calls == 1 else "answer",
                {"prompt_tokens": 1000, "completion_tokens": 0},
            )

    with ResilientModelGateway(
        [Metered()], backoff_seconds=0, pricing_per_million_tokens={"m": (1, 1)}
    ) as gateway:
        result = gateway.complete("q", "m")
        assert result.estimated_cost_usd == 0.002
        assert result.cost_complete
        assert result.attempt_trace[0].usage["prompt_tokens"] == 1000


def test_litellm_adapter_forwards_deadline_and_retains_unknown_usage(monkeypatch):
    import sys
    from types import SimpleNamespace

    from llm_eval_lab.providers import LiteLLMProvider

    calls = []

    def completion(**kwargs):
        calls.append(kwargs)
        return SimpleNamespace(
            usage=None, choices=[SimpleNamespace(message=SimpleNamespace(content=None))]
        )

    monkeypatch.setitem(sys.modules, "litellm", SimpleNamespace(completion=completion))
    result = LiteLLMProvider(timeout_seconds=7).complete("q", "synthetic/model")
    assert result.text == "" and result.usage == {}
    assert calls[0]["timeout"] == 7 and calls[0]["num_retries"] == 0
