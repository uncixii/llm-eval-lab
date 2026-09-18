from __future__ import annotations

import time
from concurrent.futures import ThreadPoolExecutor
from concurrent.futures import TimeoutError as FutureTimeoutError
from dataclasses import dataclass, field
from threading import BoundedSemaphore
from typing import Callable, Protocol

from .validation import number


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
    usage: dict[str, int] = field(default_factory=dict)
    estimated_cost_usd: float | None = None


@dataclass(frozen=True)
class CompletionResult:
    text: str
    provider: str
    model: str
    attempts: int
    failures: tuple[str, ...] = ()
    usage: dict[str, int] = field(default_factory=dict)
    estimated_cost_usd: float | None = None
    latency_ms: float = 0.0
    attempt_trace: tuple[AttemptRecord, ...] = ()
    cost_complete: bool = False


class GatewayError(RuntimeError):
    def __init__(self, trace: tuple[AttemptRecord, ...]):
        self.attempt_trace = trace
        super().__init__(
            "All completion providers failed: "
            + " | ".join(t.error or "" for t in trace)
        )


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
        max_in_flight: int = 4,
    ) -> None:
        if not providers:
            raise ValueError("至少需要一个 completion provider")
        number(timeout_seconds, "timeout_seconds")
        number(backoff_seconds, "backoff_seconds")
        if type(max_in_flight) is not int or max_in_flight <= 0:
            raise ValueError("max_in_flight must be positive")
        if (
            type(retries_per_provider) is not int
            or retries_per_provider < 0
            or timeout_seconds <= 0
        ):
            raise ValueError("retry、timeout 或 backoff 配置无效")
        self.providers = providers
        self.retries_per_provider = retries_per_provider
        self.timeout_seconds = timeout_seconds
        self.backoff_seconds = backoff_seconds
        self.sleeper = sleeper
        self.pricing = dict(pricing_per_million_tokens or {})
        for prices in self.pricing.values():
            if len(prices) != 2:
                raise ValueError("pricing requires input/output rates")
            for price in prices:
                number(price, "token price")
        self._slots = BoundedSemaphore(max_in_flight)
        self._executor = ThreadPoolExecutor(max_workers=max_in_flight)
        self._closed = False

    def close(self) -> None:
        self._closed = True
        self._executor.shutdown(wait=False, cancel_futures=True)

    def __enter__(self):
        return self

    def __exit__(self, *_):
        self.close()

    def _call_with_timeout(
        self, provider: CompletionProvider, prompt: str, model: str
    ) -> str | ProviderResponse:
        if self._closed:
            raise RuntimeError("gateway is closed")
        if not self._slots.acquire(blocking=False):
            raise RuntimeError(
                "gateway in-flight limit reached; timed-out calls may still be running"
            )
        try:
            future = self._executor.submit(provider.complete, prompt, model)
        except Exception:
            self._slots.release()
            raise
        future.add_done_callback(lambda _: self._slots.release())
        try:
            return future.result(timeout=self.timeout_seconds)
        except FutureTimeoutError as exc:
            if future.done():
                raise
            future.cancel()
            raise TimeoutError(
                f"provider timeout after {self.timeout_seconds}s"
            ) from exc

    def _estimate_cost(self, model: str, usage: dict[str, int]) -> float | None:
        if (
            model not in self.pricing
            or not {"prompt_tokens", "completion_tokens"} <= usage.keys()
        ):
            return None
        input_price, output_price = self.pricing[model]
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
                usage = {}
                cost = None
                try:
                    raw = self._call_with_timeout(provider, prompt, model)
                    if not isinstance(raw, (str, ProviderResponse)):
                        raise ValueError(
                            "provider must return text or ProviderResponse"
                        )
                    response = (
                        raw
                        if isinstance(raw, ProviderResponse)
                        else ProviderResponse(raw)
                    )
                    usage = dict(response.usage)
                    if any(type(v) is not int or v < 0 for v in usage.values()):
                        usage = {}
                        raise ValueError("invalid provider usage")
                    cost = self._estimate_cost(model, usage)
                    text = response.text.strip()
                    if not text:
                        raise ValueError("empty response")
                    latency = (time.perf_counter() - attempt_started) * 1000
                    trace.append(
                        AttemptRecord(
                            provider.name,
                            attempts,
                            True,
                            round(latency, 3),
                            usage=usage,
                            estimated_cost_usd=cost,
                        )
                    )
                    return CompletionResult(
                        text=text,
                        provider=provider.name,
                        model=model,
                        attempts=attempts,
                        failures=tuple(failures),
                        usage=dict(response.usage),
                        estimated_cost_usd=sum(t.estimated_cost_usd or 0 for t in trace)
                        if any(t.estimated_cost_usd is not None for t in trace)
                        else None,
                        latency_ms=round((time.perf_counter() - started) * 1000, 3),
                        attempt_trace=tuple(trace),
                        cost_complete=all(
                            t.estimated_cost_usd is not None for t in trace
                        ),
                    )
                except (
                    Exception
                ) as exc:  # provider boundary intentionally catches SDK errors
                    latency = (time.perf_counter() - attempt_started) * 1000
                    error = f"{provider.name}: {type(exc).__name__}: {exc}"
                    failures.append(error)
                    trace.append(
                        AttemptRecord(
                            provider.name,
                            attempts,
                            False,
                            round(latency, 3),
                            error,
                            usage,
                            cost,
                        )
                    )
                    if retry_index < self.retries_per_provider and self.backoff_seconds:
                        self.sleeper(self.backoff_seconds * (2**retry_index))
        raise GatewayError(tuple(trace))


class LiteLLMProvider:
    """LiteLLM adapter，可按 model identifier 路由 OpenAI/Anthropic/Gemini/DeepSeek。"""

    name = "litellm"

    def __init__(self, timeout_seconds: float = 25.0):
        number(timeout_seconds, "provider timeout")
        if timeout_seconds <= 0:
            raise ValueError("provider timeout must be positive")
        self.timeout_seconds = timeout_seconds

    def complete(
        self, prompt: str, model: str
    ) -> ProviderResponse:  # pragma: no cover - external API
        try:
            from litellm import completion
        except ImportError as exc:
            raise RuntimeError(
                "请安装 llm-eval-lab[providers] 后使用 LiteLLMProvider"
            ) from exc
        response = completion(
            model=model,
            messages=[{"role": "user", "content": prompt}],
            timeout=self.timeout_seconds,
            num_retries=0,
        )
        usage = getattr(response, "usage", None)
        usage_payload = {
            key: getattr(usage, key)
            for key in ("prompt_tokens", "completion_tokens", "total_tokens")
            if usage is not None and getattr(usage, key, None) is not None
        }
        return ProviderResponse(
            response.choices[0].message.content or "", usage_payload
        )
