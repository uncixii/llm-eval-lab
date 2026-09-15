from llm_eval_lab.providers import ProviderResponse, ResilientModelGateway
import time


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
        [FailingProvider(), WorkingProvider()], retries_per_provider=1, backoff_seconds=0
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
