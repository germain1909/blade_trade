#!/usr/bin/env python3
"""
Model/provider abstraction for the LLM decision layer.

Today there is exactly one provider — an OpenAI-compatible local endpoint
(LM Studio serving Qwen3-VL-4B). Everything that varies (base_url, model
name, temperature, prompt version) is configuration, never hardcoded, so
swapping to a different local model, a larger Qwen model, or eventually a
remote API means constructing a different LLMConfig / provider instance —
`decision_strategy.LLMStrategy` and everything upstream of it never changes.
"""

from __future__ import annotations

import json
import time
from dataclasses import dataclass
from typing import Optional, Protocol


@dataclass
class LLMConfig:
    base_url: str = "http://localhost:1234/v1"   # LM Studio's OpenAI-compatible endpoint
    model: str = "qwen3-vl-4b-instruct"
    temperature: float = 0.2
    max_tokens: int = 512
    timeout_s: float = 30.0
    prompt_version: str = "v1"
    seed: Optional[int] = None       # untested whether this server honors it — probe, don't assume
    max_retries: int = 2
    vision: bool = True


class LLMProvider(Protocol):
    def complete(self, system: str, user: str) -> str: ...


class OpenAICompatProvider:
    """Wraps any OpenAI-compatible chat-completions endpoint — LM Studio,
    vLLM, Ollama's OpenAI-compat mode, or a real OpenAI/Anthropic-compatible
    proxy later. Requires the `openai` package (pip install openai) purely
    as an HTTP client; no OpenAI account or API key is used against LM
    Studio, which ignores the key value."""

    def __init__(self, config: LLMConfig | None = None):
        self.config = config or LLMConfig()
        try:
            from openai import OpenAI
        except ImportError as exc:
            raise ImportError(
                "The 'openai' package is required for OpenAICompatProvider "
                "(used only as an HTTP client against LM Studio's local endpoint). "
                "Install with: pip install openai"
            ) from exc

        self._client = OpenAI(
            base_url=self.config.base_url,
            api_key="lm-studio",  # LM Studio does not check this; some clients require a non-empty string
            timeout=self.config.timeout_s,
        )

        # Populated by the most recent complete_structured/complete_vision_structured
        # call, for callers to persist alongside the result (experiment_store.py's
        # analyses table wants latency/retry_count/parse_errors per analysis).
        self.last_latency_s: Optional[float] = None
        self.last_retry_count: int = 0
        self.last_parse_error: Optional[str] = None

    def complete(self, system: str, user: str) -> str:
        try:
            resp = self._client.chat.completions.create(
                model=self.config.model,
                temperature=self.config.temperature,
                max_tokens=self.config.max_tokens,
                messages=[
                    {"role": "system", "content": system},
                    {"role": "user", "content": user},
                ],
            )
        except Exception as exc:
            # Surface a clear, single error rather than a raw httpx/httpcore traceback —
            # callers (LLMStrategy / run_experiment.py) can catch this to abort the
            # whole run cleanly instead of crashing on the first candidate, or writing
            # hundreds of "connection failed" rows into the experiment store.
            raise ConnectionError(
                f"Could not reach the local LLM at {self.config.base_url} "
                f"(model={self.config.model}). Is LM Studio running with that model loaded? "
                f"Underlying error: {exc}"
            ) from exc
        return resp.choices[0].message.content or ""

    def _kwargs(self, seed_supported: bool = True) -> dict:
        kw = {"temperature": self.config.temperature, "max_tokens": self.config.max_tokens}
        if seed_supported and self.config.seed is not None:
            kw["seed"] = self.config.seed
        return kw

    def _response_format(self, schema: dict, schema_name: str) -> dict:
        # Confirmed against this LM Studio build (Phase 2 Step 0): bare
        # {"type": "json_object"} is rejected outright (400 error). Grammar-
        # constrained json_schema mode is the reliable path — not best-effort
        # prompting, an actual output-shape guarantee.
        return {
            "type": "json_schema",
            "json_schema": {"name": schema_name, "strict": True, "schema": schema},
        }

    def _complete_with_retries(self, messages: list[dict], schema: dict, schema_name: str) -> tuple[dict, str]:
        attempts = self.config.max_retries + 1
        last_error: Optional[str] = None
        self.last_retry_count = 0
        t0 = time.time()
        for attempt in range(attempts):
            try:
                resp = self._client.chat.completions.create(
                    model=self.config.model,
                    messages=messages,
                    response_format=self._response_format(schema, schema_name),
                    **self._kwargs(),
                )
                raw = resp.choices[0].message.content or ""
                parsed = json.loads(raw)
                self.last_latency_s = time.time() - t0
                self.last_parse_error = None
                return parsed, raw
            except json.JSONDecodeError as exc:
                last_error = f"JSONDecodeError: {exc}"
            except Exception as exc:
                last_error = f"{type(exc).__name__}: {exc}"
            self.last_retry_count = attempt + 1

        self.last_latency_s = time.time() - t0
        self.last_parse_error = last_error
        raise ValueError(
            f"complete_structured failed after {attempts} attempt(s) against "
            f"{self.config.base_url} (model={self.config.model}): {last_error}"
        )

    def complete_structured(self, system: str, user: str, schema: dict, schema_name: str = "response") -> tuple[dict, str]:
        """json_schema-constrained structured output. Returns (parsed_dict, raw_response_text).
        Retries up to config.max_retries times on parse/transport failure before raising.
        Latency/retry-count/parse-error from the most recent call are on
        self.last_latency_s / self.last_retry_count / self.last_parse_error."""
        messages = [
            {"role": "system", "content": system},
            {"role": "user", "content": user},
        ]
        return self._complete_with_retries(messages, schema, schema_name)

    def complete_vision_structured(
        self, system: str, user_text: str, image_b64: str, schema: dict, schema_name: str = "response",
    ) -> tuple[dict, str]:
        """Same as complete_structured, but with an image attached to the user
        message via the OpenAI-style image_url data-URI content block (the
        exact format Phase 2 Step 0's Test 3 confirmed this server accepts)."""
        messages = [
            {"role": "system", "content": system},
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": user_text},
                    {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{image_b64}"}},
                ],
            },
        ]
        return self._complete_with_retries(messages, schema, schema_name)
