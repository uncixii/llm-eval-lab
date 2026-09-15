from __future__ import annotations

from llm_eval_lab.providers import ProviderResponse, ResilientModelGateway


class SyntheticPrimary:
    name = "synthetic-primary"

    def complete(self, prompt: str, model: str) -> str:
        raise TimeoutError("synthetic timeout")


class SyntheticFallback:
    name = "synthetic-fallback"

    def complete(self, prompt: str, model: str) -> ProviderResponse:
        return ProviderResponse(
            "synthetic answer",
            {"prompt_tokens": 120, "completion_tokens": 30, "total_tokens": 150},
        )


def main() -> None:
    gateway = ResilientModelGateway(
        [SyntheticPrimary(), SyntheticFallback()],
        retries_per_provider=1,
        backoff_seconds=0,
        pricing_per_million_tokens={"synthetic/model": (1.0, 2.0)},
    )
    result = gateway.complete("synthetic prompt", "synthetic/model")
    print(f"provider={result.provider}, attempts={result.attempts}")
    print(f"usage={result.usage}, cost_usd={result.estimated_cost_usd:.6f}")
    for attempt in result.attempt_trace:
        print(attempt)


if __name__ == "__main__":
    main()
