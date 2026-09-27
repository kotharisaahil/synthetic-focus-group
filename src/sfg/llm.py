"""Provider adapters and a thin wrapper that logs every call and enforces JSON contracts."""

from __future__ import annotations

import inspect
import json
import os
import re
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from dataclasses import asdict, dataclass, field
from typing import Any, Callable, Optional, Protocol, TypeVar

from pydantic import BaseModel, ValidationError

T = TypeVar("T", bound=BaseModel)

# Which configured model each agent role uses.
ROLE_MODEL = {
    "backstory": "participant",
    "rating": "participant",
    "participant": "participant",
    "moderator": "moderator",
    "analyst": "analyst",
}

# OpenAI reasoning families: hidden reasoning consumes completion tokens and custom
# temperature is rejected.
REASONING_PREFIXES = ("gpt-5", "o1", "o3", "o4")

# When a reply is cut off at its token limit, the next attempt doubles the limit, up to this.
MAX_TOKEN_CEILING = 32000

# Errors that retrying will not fix.
_FATAL_ERROR_HINTS = ("Authentication", "PermissionDenied", "NotFound", "BadRequest", "Invalid")


class LLMError(RuntimeError):
    pass


class Provider(Protocol):
    name: str

    def complete(
        self,
        *,
        model: str,
        system: str,
        messages: list[dict[str, str]],
        temperature: Optional[float],
        max_tokens: int,
        meta: dict[str, Any],
    ) -> "str | tuple[str, dict[str, int]]": ...
    # Return the reply text, or (text, {"input": n, "output": n}) when token usage is known.


class AnthropicProvider:
    name = "anthropic"

    def __init__(self) -> None:
        try:
            import anthropic
        except ImportError as e:  # pragma: no cover - depends on environment
            raise LLMError("Install the Anthropic SDK: pip install 'synthetic-focus-group[anthropic]'") from e
        if not os.environ.get("ANTHROPIC_API_KEY"):
            raise LLMError("ANTHROPIC_API_KEY is not set (see .env.example)")
        self._client = anthropic.Anthropic()
        # Recent SDK versions (1.x) removed the temperature parameter from messages.create.
        # Check once instead of failing on every call.
        self.supports_temperature = "temperature" in inspect.signature(self._client.messages.create).parameters

    def complete(self, *, model, system, messages, temperature, max_tokens, meta):  # pragma: no cover
        kwargs: dict[str, Any] = dict(model=model, system=system, messages=messages, max_tokens=max_tokens)
        if temperature is not None and self.supports_temperature:
            kwargs["temperature"] = temperature
        resp = self._client.messages.create(**kwargs)
        # Only text blocks are the answer; models may also return reasoning ("thinking") blocks,
        # which count toward max_tokens but are not part of the reply.
        text = "".join(
            getattr(block, "text", "") for block in resp.content if getattr(block, "type", None) == "text"
        ).strip()
        u = getattr(resp, "usage", None)
        usage = {
            "input": getattr(u, "input_tokens", 0) or 0,
            "output": getattr(u, "output_tokens", 0) or 0,
            "truncated": getattr(resp, "stop_reason", None) == "max_tokens",
        }
        return text, usage


class OpenAIProvider:
    name = "openai"

    def __init__(self) -> None:
        try:
            import openai
        except ImportError as e:  # pragma: no cover
            raise LLMError("Install the OpenAI SDK: pip install 'synthetic-focus-group[openai]'") from e
        if not os.environ.get("OPENAI_API_KEY"):
            raise LLMError("OPENAI_API_KEY is not set (see .env.example)")
        self._client = openai.OpenAI()

    def complete(self, *, model, system, messages, temperature, max_tokens, meta):  # pragma: no cover
        reasoning = model.startswith(REASONING_PREFIXES)
        kwargs: dict[str, Any] = dict(
            model=model,
            messages=[{"role": "system", "content": system}, *messages],
            max_completion_tokens=max_tokens * (6 if reasoning else 1),
        )
        if temperature is not None and not reasoning:
            kwargs["temperature"] = temperature
        resp = self._client.chat.completions.create(**kwargs)
        u = getattr(resp, "usage", None)
        usage = {
            "input": getattr(u, "prompt_tokens", 0) or 0,
            "output": getattr(u, "completion_tokens", 0) or 0,
            "truncated": resp.choices[0].finish_reason == "length",
        }
        return (resp.choices[0].message.content or "").strip(), usage


def parallel_map(fn: Callable[[Any], Any], items: list[Any], workers: int) -> list[Any]:
    """Map in a thread pool, preserving input order. Used for calls that don't depend on each other."""
    if workers <= 1 or len(items) <= 1:
        return [fn(x) for x in items]
    with ThreadPoolExecutor(max_workers=workers) as pool:
        return list(pool.map(fn, items))


def make_provider(name: str) -> Provider:
    if name == "anthropic":
        return AnthropicProvider()
    if name == "openai":
        return OpenAIProvider()
    if name == "mock":
        from .mock import MockProvider

        return MockProvider()
    raise LLMError(f"unknown provider '{name}'")


# ---------------------------------------------------------------------------
# JSON extraction
# ---------------------------------------------------------------------------


def extract_json(text: str) -> Any:
    """Pull the first complete JSON object out of a model reply.

    Handles code fences and prose before or after the object, and braces inside strings.
    """
    if not text or "{" not in text:
        raise ValueError("no JSON object found in reply")
    cleaned = re.sub(r"```(?:json)?", "", text)
    start = cleaned.index("{")
    depth, in_str, esc = 0, False, False
    for i in range(start, len(cleaned)):
        ch = cleaned[i]
        if in_str:
            if esc:
                esc = False
            elif ch == "\\":
                esc = True
            elif ch == '"':
                in_str = False
            continue
        if ch == '"':
            in_str = True
        elif ch == "{":
            depth += 1
        elif ch == "}":
            depth -= 1
            if depth == 0:
                return json.loads(cleaned[start : i + 1])
    raise ValueError("unterminated JSON object in reply")


def _short(err: Exception, limit: int = 300) -> str:
    msg = str(err).replace("\n", " ")
    return msg if len(msg) <= limit else msg[:limit] + "..."


# ---------------------------------------------------------------------------
# Wrapper
# ---------------------------------------------------------------------------


@dataclass
class CallRecord:
    role: str
    model: str
    system: str
    messages: list[dict[str, str]]
    response: str
    latency_ms: int
    meta: dict[str, Any] = field(default_factory=dict)
    usage: dict[str, int] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


class LLM:
    """Routes each agent role to its model, retries transient failures, and logs every call."""

    def __init__(
        self,
        provider: Provider,
        models: dict[str, str],
        sleep: Callable[[float], None] = time.sleep,
        max_attempts: int = 4,
        log_path: "Optional[os.PathLike[str] | str]" = None,
    ) -> None:
        self.provider = provider
        self.models = models
        self.calls: list[CallRecord] = []
        self._lock = threading.Lock()
        self._log_path = log_path  # when set, every call is appended here as it happens
        self._sleep = sleep
        self._max_attempts = max_attempts

    def usage(self) -> dict[str, dict[str, int]]:
        """Token totals per model, e.g. {"claude-haiku-...": {"calls": 180, "input": ..., "output": ...}}."""
        totals: dict[str, dict[str, int]] = {}
        with self._lock:
            for c in self.calls:
                t = totals.setdefault(c.model, {"calls": 0, "input": 0, "output": 0})
                t["calls"] += 1
                t["input"] += c.usage.get("input", 0)
                t["output"] += c.usage.get("output", 0)
        return totals

    @property
    def is_mock(self) -> bool:
        return self.provider.name == "mock"

    def model_for(self, role: str) -> str:
        return self.models[ROLE_MODEL[role]]

    def _call(self, role, system, messages, temperature, max_tokens, meta) -> tuple[str, bool]:
        """One provider call with retries for transient errors. Returns (text, truncated)."""
        model = self.model_for(role)
        last: Optional[Exception] = None
        for attempt in range(self._max_attempts):
            t0 = time.perf_counter()
            try:
                reply = self.provider.complete(
                    model=model,
                    system=system,
                    messages=messages,
                    temperature=temperature,
                    max_tokens=max_tokens,
                    meta={**(meta or {}), "role": role},
                )
            except LLMError:
                raise
            except Exception as e:  # provider SDK errors
                last = e
                if any(h in type(e).__name__ for h in _FATAL_ERROR_HINTS):
                    raise LLMError(f"{role} call failed: {_short(e)}") from e
                if attempt < self._max_attempts - 1:
                    self._sleep(min(30, 3 * 2**attempt))  # 3s, 6s, 12s: enough to ride out a rate limit
                continue
            out, usage = reply if isinstance(reply, tuple) else (reply, {})
            record = CallRecord(
                    role=role,
                    model=model,
                    system=system,
                    messages=list(messages),
                    response=out,
                    latency_ms=int((time.perf_counter() - t0) * 1000),
                    meta={**(meta or {}), "max_tokens": max_tokens},
                    usage=dict(usage),
                )
            with self._lock:
                self.calls.append(record)
                if self._log_path:
                    with open(self._log_path, "a", encoding="utf-8") as f:
                        f.write(json.dumps(record.to_dict(), default=str) + "\n")
            return out, bool(usage.get("truncated"))
        raise LLMError(f"{role} call failed after {self._max_attempts} attempts: {_short(last)}")

    def text(self, role, system, user, *, temperature=None, max_tokens=500, meta=None, retries: int = 2) -> str:
        """Plain-text reply. Empty replies are retried; replies cut off at the token limit are
        retried with double the limit (models can spend their budget on hidden reasoning)."""
        budget, out = max_tokens, ""
        for attempt in range(retries + 1):
            out, truncated = self._call(
                role, system, [{"role": "user", "content": user}], temperature, budget, {**(meta or {}), "attempt": attempt}
            )
            out = (out or "").strip()
            if out and not truncated:
                return out
            if truncated:
                budget = min(budget * 2, MAX_TOKEN_CEILING)
        if out:  # a reply cut short is better than stopping the whole session
            return out
        raise LLMError(f"{role}: empty reply after {retries + 1} attempts")

    def json(
        self,
        role: str,
        system: str,
        user: str,
        schema: type[T],
        *,
        temperature: Optional[float] = None,
        max_tokens: int = 800,
        meta: Optional[dict] = None,
        retries: int = 2,
    ) -> T:
        """Ask for JSON matching `schema`.

        A reply cut off at the token limit (or empty) is retried with double the limit. A complete
        reply that fails validation is shown back to the model with the error, and it tries again.
        """
        messages = [{"role": "user", "content": user}]
        budget = max_tokens
        last: Optional[Exception] = None
        for attempt in range(retries + 1):
            raw, truncated = self._call(role, system, messages, temperature, budget, {**(meta or {}), "attempt": attempt})
            if truncated or not raw.strip():
                last = ValueError("reply was cut off at the token limit" if truncated else "empty reply")
                if truncated:
                    budget = min(budget * 2, MAX_TOKEN_CEILING)
                continue
            try:
                return schema.model_validate(extract_json(raw))
            except (ValueError, ValidationError) as e:
                last = e
                messages = messages + [
                    {"role": "assistant", "content": raw},
                    {
                        "role": "user",
                        "content": f"That reply could not be used ({_short(e)}). "
                        "Reply again with valid JSON only, in exactly the requested format.",
                    },
                ]
        raise LLMError(f"{role}: no valid JSON after {retries + 1} attempts: {_short(last)}")
