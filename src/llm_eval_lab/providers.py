from __future__ import annotations

import time
from concurrent.futures import ThreadPoolExecutor, TimeoutError as FutureTimeoutError
from dataclasses import dataclass, field
from typing import Callable, Protocol


@dataclass(frozen=True)
class ProviderResponse:
    text: str
    usage: dict[str, int] = field(default_factory=dict)


class CompletionProvider(Protocol):
    name: str

    def complete(self, prompt: str, model: str) -> str | ProviderResponse: ...


@dataclass(frozen=True)
class AttemptRecord:
    provider: str
    attempt: int
    ok: bool
    latency_ms: float
    error: str | None = None


@dataclass(frozen=True)
class CompletionResult:
    text: str
    provider: str
    model: str
    attempts: int
    failures: tuple[str, ...] = ()
    usage: dict[str, int] = field(default_factory=dict)
    estimated_cost_usd: float = 0.0
    latency_ms: float = 0.0
    attempt_trace: tuple[AttemptRecord, ...] = ()


class ResilientModelGateway:
    """统一执行 timeout、retry/backoff、跨 provider fallback 与 usage/cost 记录。"""

    def __init__(
        self,
        providers: list[CompletionProvider],
        retries_per_provider: int = 1,
        timeout_seconds: float = 30.0,
        backoff_seconds: float = 0.25,
        sleeper: Callable[[float], None] = time.sleep,
        pricing_per_million_tokens: dict[str, tuple[float, float]] | None = None,
    ) -> None:
        if not providers:
            raise ValueError("至少需要一个 completion provider")
        if retries_per_provider < 0 or timeout_seconds <= 0 or backoff_seconds < 0:
            raise ValueError("retry、timeout 或 backoff 配置无效")
        self.providers = providers
        self.retries_per_provider = retries_per_provider
        self.timeout_seconds = timeout_seconds
        self.backoff_seconds = backoff_seconds
        self.sleeper = sleeper
        self.pricing = dict(pricing_per_million_tokens or {})

    def _call_with_timeout(
        self, provider: CompletionProvider, prompt: str, model: str
    ) -> str | ProviderResponse:
        executor = ThreadPoolExecutor(max_workers=1)
        future = executor.submit(provider.complete, prompt, model)
        try:
            return future.result(timeout=self.timeout_seconds)
        except FutureTimeoutError as exc:
            if future.done():
                raise
            future.cancel()
            raise TimeoutError(f"provider timeout after {self.timeout_seconds}s") from exc
        finally:
            executor.shutdown(wait=False, cancel_futures=True)

    def _estimate_cost(self, model: str, usage: dict[str, int]) -> float:
        input_price, output_price = self.pricing.get(model, (0.0, 0.0))
        return (
            usage.get("prompt_tokens", 0) * input_price
            + usage.get("completion_tokens", 0) * output_price
        ) / 1_000_000

    def complete(self, prompt: str, model: str) -> CompletionResult:
        failures: list[str] = []
        trace: list[AttemptRecord] = []
        attempts = 0
        started = time.perf_counter()
        for provider in self.providers:
            for retry_index in range(self.retries_per_provider + 1):
                attempts += 1
                attempt_started = time.perf_counter()
                try:
                    raw = self._call_with_timeout(provider, prompt, model)
                    response = raw if isinstance(raw, ProviderResponse) else ProviderResponse(str(raw))
                    text = response.text.strip()
                    if not text:
                        raise ValueError("empty response")
                    latency = (time.perf_counter() - attempt_started) * 1000
                    trace.append(AttemptRecord(provider.name, attempts, True, round(latency, 3)))
                    return CompletionResult(
                        text=text,
                        provider=provider.name,
                        model=model,
                        attempts=attempts,
                        failures=tuple(failures),
                        usage=dict(response.usage),
                        estimated_cost_usd=self._estimate_cost(model, response.usage),
                        latency_ms=round((time.perf_counter() - started) * 1000, 3),
                        attempt_trace=tuple(trace),
                    )
                except Exception as exc:  # provider boundary intentionally catches SDK errors
                    latency = (time.perf_counter() - attempt_started) * 1000
                    error = f"{provider.name}: {type(exc).__name__}: {exc}"
                    failures.append(error)
                    trace.append(AttemptRecord(provider.name, attempts, False, round(latency, 3), error))
                    if retry_index < self.retries_per_provider and self.backoff_seconds:
                        self.sleeper(self.backoff_seconds * (2**retry_index))
        raise RuntimeError("所有 completion providers 均失败: " + " | ".join(failures))


class LiteLLMProvider:
    """LiteLLM adapter，可按 model identifier 路由 OpenAI/Anthropic/Gemini/DeepSeek。"""

    name = "litellm"

    def complete(self, prompt: str, model: str) -> ProviderResponse:  # pragma: no cover - external API
        try:
            from litellm import completion
        except ImportError as exc:
            raise RuntimeError("请安装 llm-eval-lab[providers] 后使用 LiteLLMProvider") from exc
        response = completion(model=model, messages=[{"role": "user", "content": prompt}])
        usage = getattr(response, "usage", None)
        usage_payload = {
            "prompt_tokens": int(getattr(usage, "prompt_tokens", 0) or 0),
            "completion_tokens": int(getattr(usage, "completion_tokens", 0) or 0),
            "total_tokens": int(getattr(usage, "total_tokens", 0) or 0),
        }
        return ProviderResponse(str(response.choices[0].message.content), usage_payload)
