"""
Multi-provider LLM client with automatic fallback.

Priority order:
  1. Qazcode Hub (GPT-OSS) — QAZCODE_HUB_URL + QAZCODE_HUB_API_KEY
  2. Groq (free tier) — GROQ_API_KEY
     Model: llama-3.3-70b-versatile (30 RPM, 1K RPD, 12K TPM, 100K TPD)
  3. Google Gemini (free tier) — GEMINI_API_KEY
     Model: gemini-1.5-flash (15 RPM, 1M TPM)
  4. OpenAI (paid fallback) — OPENAI_API_KEY
     Model: gpt-4o-mini

All providers expose an OpenAI-compatible chat completions interface.
Gemini is accessed via its OpenAI-compatible endpoint.
"""

from __future__ import annotations

import json
import logging
import os
import time
from typing import Optional

logger = logging.getLogger(__name__)


def _load_env():
    """Load .env file into os.environ if present."""
    env_path = os.path.join(os.path.dirname(__file__), "..", ".env")
    env_path = os.path.normpath(env_path)
    if os.path.exists(env_path):
        with open(env_path) as f:
            for line in f:
                line = line.strip()
                if not line or line.startswith("#") or "=" not in line:
                    continue
                key, _, val = line.partition("=")
                key = key.strip()
                val = val.strip()
                if key and key not in os.environ:
                    os.environ[key] = val


_load_env()


class ProviderConfig:
    """Configuration for one LLM provider."""

    def __init__(self, name: str, base_url: str, api_key: str, model: str):
        self.name = name
        self.base_url = base_url
        self.api_key = api_key
        self.model = model

    def is_available(self) -> bool:
        return bool(self.api_key)


def _get_providers() -> list[ProviderConfig]:
    """Build ordered provider list from environment."""
    providers = []

    # 1. Qazcode Hub (GPT-OSS)
    hub_url = os.environ.get("QAZCODE_HUB_URL", "")
    hub_key = os.environ.get("QAZCODE_HUB_API_KEY", "")
    hub_model = os.environ.get("QAZCODE_HUB_MODEL", "oss-120b")
    if hub_url and hub_key:
        providers.append(ProviderConfig("qazcode_hub", hub_url, hub_key, hub_model))

    # 2. Groq — llama-3.3-70b-versatile on free tier
    groq_key = os.environ.get("GROQ_API_KEY", "")
    groq_model = os.environ.get("GROQ_MODEL", "llama-3.3-70b-versatile")
    if groq_key:
        providers.append(
            ProviderConfig(
                "groq", "https://api.groq.com/openai/v1", groq_key, groq_model
            )
        )

    # 3. Gemini — gemini-1.5-flash via OpenAI-compatible endpoint
    gemini_key = os.environ.get("GEMINI_API_KEY", "")
    gemini_model = os.environ.get("GEMINI_MODEL", "gemini-1.5-flash")
    if gemini_key:
        providers.append(
            ProviderConfig(
                "gemini",
                "https://generativelanguage.googleapis.com/v1beta/openai/",
                gemini_key,
                gemini_model,
            )
        )

    # 4. OpenAI (paid fallback)
    openai_key = os.environ.get("OPENAI_API_KEY", "")
    openai_model = os.environ.get("OPENAI_MODEL", "gpt-4o-mini")
    if openai_key:
        providers.append(
            ProviderConfig(
                "openai", "https://api.openai.com/v1", openai_key, openai_model
            )
        )

    return providers


class LLMClient:
    """
    OpenAI-compatible chat completions client with automatic provider fallback.

    Usage:
        client = LLMClient()
        response = client.chat(messages=[{"role": "user", "content": "Hello"}])
        text = response["choices"][0]["message"]["content"]
    """

    def __init__(self, max_retries: int = 2, retry_delay: float = 2.0):
        self.providers = _get_providers()
        self.max_retries = max_retries
        self.retry_delay = retry_delay
        if not self.providers:
            raise RuntimeError(
                "No LLM provider configured. Set at least one of: "
                "QAZCODE_HUB_API_KEY, GROQ_API_KEY, GEMINI_API_KEY, OPENAI_API_KEY"
            )
        logger.info(
            "LLMClient initialized with providers: %s", [p.name for p in self.providers]
        )

    def chat(
        self,
        messages: list[dict],
        max_tokens: int = 300,
        temperature: float = 0.05,
        json_mode: bool = False,
        provider_override: Optional[str] = None,
    ) -> dict:
        """
        Call chat completions with automatic fallback across providers.

        Returns the raw API response dict with a 'choices' key.
        Raises RuntimeError if all providers fail.
        """
        from openai import OpenAI, RateLimitError, APIStatusError

        providers = self.providers
        if provider_override:
            providers = [p for p in self.providers if p.name == provider_override]
            if not providers:
                raise ValueError(f"Provider '{provider_override}' not configured")

        last_error = None
        for provider in providers:
            for attempt in range(self.max_retries + 1):
                try:
                    client = OpenAI(
                        base_url=provider.base_url,
                        api_key=provider.api_key,
                    )
                    kwargs: dict = dict(
                        model=provider.model,
                        messages=messages,
                        max_tokens=max_tokens,
                        temperature=temperature,
                    )
                    if json_mode:
                        kwargs["response_format"] = {"type": "json_object"}

                    response = client.chat.completions.create(**kwargs)
                    logger.debug("Provider %s succeeded", provider.name)
                    # Return as dict for uniform access
                    return response.model_dump()

                except RateLimitError as e:
                    last_error = e
                    logger.warning(
                        "Rate limit on %s (attempt %d/%d): %s",
                        provider.name,
                        attempt + 1,
                        self.max_retries + 1,
                        e,
                    )
                    if attempt < self.max_retries:
                        time.sleep(self.retry_delay * (attempt + 1))
                    else:
                        logger.warning(
                            "Exhausted retries on %s, trying next provider",
                            provider.name,
                        )
                        break

                except APIStatusError as e:
                    last_error = e
                    logger.warning(
                        "API error on %s: %s — trying next provider", provider.name, e
                    )
                    break

                except Exception as e:
                    last_error = e
                    logger.warning(
                        "Unexpected error on %s: %s — trying next provider",
                        provider.name,
                        e,
                    )
                    break

        raise RuntimeError(f"All LLM providers failed. Last error: {last_error}")

    def chat_text(self, messages: list[dict], **kwargs) -> str:
        """Convenience: returns just the text content of the first choice."""
        response = self.chat(messages, **kwargs)
        return response["choices"][0]["message"]["content"]

    def chat_json(self, messages: list[dict], **kwargs) -> dict:
        """Convenience: returns parsed JSON from the first choice."""
        kwargs["json_mode"] = True
        text = self.chat_text(messages, **kwargs)
        # Strip markdown code fences if present
        text = text.strip()
        if text.startswith("```"):
            text = "\n".join(text.split("\n")[1:])
            if text.endswith("```"):
                text = text[:-3]
        return json.loads(text.strip())

    @property
    def active_provider(self) -> str:
        """Name of the first available provider."""
        return self.providers[0].name if self.providers else "none"


# Module-level singleton (lazy-initialized)
_client: Optional[LLMClient] = None


def get_client() -> LLMClient:
    global _client
    if _client is None:
        _client = LLMClient()
    return _client
