from __future__ import annotations

import time
from typing import Optional

import requests

from app.assistants.base import BaseAssistant
from app.config import config


class OSSAssistant(BaseAssistant):


    MODEL_TYPE = "oss"
    MODEL_NAME: str = config.OSS_MODEL_ID

    def __init__(self, session_id: Optional[str] = None):
        super().__init__(session_id)
        self._api_url = config.OSS_API_URL
        self._headers = config.oss_headers
        self._timeout = 60  # seconds — serverless cold-start can be slow

    def _generate_response(self) -> str:
        messages = self.memory.get_messages()

        payload = {
            "model": config.OSS_MODEL_ID,
            "messages": messages,
            "max_tokens": config.OSS_MAX_TOKENS,
            "temperature": config.OSS_TEMPERATURE,
            "stream": False,
        }

        try:
            resp = requests.post(
                self._api_url,
                headers=self._headers,
                json=payload,
                timeout=self._timeout,
            )
        except requests.exceptions.Timeout:
            raise RuntimeError(
                "HuggingFace Inference API timed out. "
                "The model may be cold-starting — please wait 20–30 seconds and retry."
            )
        except requests.exceptions.ConnectionError as exc:
            raise RuntimeError(f"Network error connecting to HF API: {exc}")

        if resp.status_code == 503:
            raise RuntimeError(
                "Model is loading (cold start). "
                "HF free-tier models sleep after inactivity. "
                "Please retry in 20–30 seconds."
            )

        if resp.status_code == 401:
            raise RuntimeError(
                "HuggingFace API token is invalid or missing. "
                "Set HF_API_TOKEN in your .env file."
            )

        if resp.status_code != 200:
            raise RuntimeError(
                f"HF Inference API error {resp.status_code}: {resp.text[:200]}"
            )

        data = resp.json()

        # OpenAI-compatible response format
        try:
            return data["choices"][0]["message"]["content"].strip()
        except (KeyError, IndexError) as exc:
            raise RuntimeError(
                f"Unexpected response format from HF API: {exc}. "
                f"Raw: {str(data)[:300]}"
            )

    def warmup(self) -> str:
        """
        Send a cheap warmup request to trigger model loading.
        Returns status message.
        """
        if not config.HF_API_TOKEN:
            return " HF_API_TOKEN not set — OSS model will not work."
        try:
            t0 = time.perf_counter()
            resp = requests.post(
                self._api_url,
                headers=self._headers,
                json={
                    "model": config.OSS_MODEL_ID,
                    "messages": [{"role": "user", "content": "Hi"}],
                    "max_tokens": 10,
                },
                timeout=45,
            )
            elapsed = (time.perf_counter() - t0) * 1000
            if resp.status_code == 200:
                return f" OSS model ready ({elapsed:.0f} ms)"
            return f" Warmup HTTP {resp.status_code}: {resp.text[:100]}"
        except Exception as exc:
            return f" Warmup failed: {exc}"
