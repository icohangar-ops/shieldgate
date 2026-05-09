"""Qwen API client using OpenAI-compatible interface.

Uses Alibaba Cloud DashScope with automatic China/Singapore region failover.
"""

from __future__ import annotations

import json
import os
from typing import Optional

import httpx


# Default API keys and endpoints (China primary, Singapore backup)
_CHINA_API_KEY = "sk-6daeb1e1fa5349df9ab31ab1b76657a4"
_CHINA_BASE_URL = "https://dashscope.aliyuncs.com/compatible-mode/v1"

_SG_API_KEY = "sk-a6e8ffd8f6ed4afe8f77440d6ad3dfff"
_SG_BASE_URL = "https://dashscope-intl.aliyuncs.com/compatible-mode/v1"

_DEFAULT_MODEL = "qwen3.6-flash"


class QwenClient:
    """Async client for the Qwen API via DashScope (OpenAI-compatible)."""

    def __init__(
        self,
        api_key: Optional[str] = None,
        base_url: Optional[str] = None,
        model: str = _DEFAULT_MODEL,
        timeout: float = 60.0,
    ):
        # Allow override via params, then env vars, then defaults
        self.api_key = api_key or os.environ.get(
            "QWEN_API_KEY", _CHINA_API_KEY
        )
        self.base_url = (base_url or os.environ.get(
            "QWEN_BASE_URL", _CHINA_BASE_URL
        )).rstrip("/")

        # Backup endpoint for failover
        self.backup_api_key = os.environ.get(
            "QWEN_BACKUP_API_KEY", _SG_API_KEY
        )
        self.backup_base_url = os.environ.get(
            "QWEN_BACKUP_BASE_URL", _SG_BASE_URL
        ).rstrip("/")

        self.model = model
        self.timeout = timeout

    async def analyze(self, prompt: str, system: str = "") -> str:
        """Send a chat completion request and return the assistant's text.

        Automatically fails over to the Singapore region on error.
        """
        result = await self._call_api(
            self.base_url, self.api_key, prompt, system
        )
        return result

    async def _call_api(
        self, base_url: str, api_key: str, prompt: str, system: str
    ) -> str:
        headers = {
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
        }
        messages: list[dict] = []
        if system:
            messages.append({"role": "system", "content": system})
        messages.append({"role": "user", "content": prompt})

        payload = {
            "model": self.model,
            "messages": messages,
            "temperature": 0.3,
            "max_tokens": 2000,
        }

        try:
            async with httpx.AsyncClient(timeout=self.timeout) as client:
                resp = await client.post(
                    f"{base_url}/chat/completions",
                    headers=headers,
                    json=payload,
                )
                resp.raise_for_status()
                data = resp.json()
                return data["choices"][0]["message"]["content"]
        except (httpx.HTTPError, KeyError, IndexError):
            # Failover to backup region
            if base_url != self.backup_base_url:
                return await self._call_api(
                    self.backup_base_url, self.backup_api_key, prompt, system
                )
            raise

    async def health_check(self) -> bool:
        """Verify connectivity to the Qwen API."""
        try:
            result = await self.analyze("Say OK", "")
            return bool(result and "OK" in result.upper())
        except Exception:
            return False
