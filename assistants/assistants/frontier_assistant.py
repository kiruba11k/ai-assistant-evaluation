"""
Frontier Model Assistant — Anthropic Claude Sonnet
===================================================
Uses the official Anthropic Python SDK with streaming support.

Supported frontier models (set FRONTIER_MODEL in config.py):
  - claude-sonnet-4-20250514   (recommended: balanced quality + cost)
  - claude-opus-4-5            (highest quality, higher cost)
  - claude-haiku-4-5           (fastest, lowest cost)

Cost & Latency (Claude Sonnet 4)
----------------------------------
| Metric          | Value (approx)                       |
|-----------------|--------------------------------------|
| Input cost      | $3 / 1M tokens                       |
| Output cost     | $15 / 1M tokens                      |
| p50 latency     | 0.5 – 1.5 s (first token)            |
| p95 latency     | 2 – 5 s                              |
| Max context     | 200 000 tokens                       |
| Throughput      | ~800 tok/s                           |
"""

from __future__ import annotations

from typing import Optional

import anthropic

from app.assistants.base import BaseAssistant
from app.config import config


class FrontierAssistant(BaseAssistant):
    """
    Personal assistant powered by Claude Sonnet (Anthropic).

    Architecture
    ------------
    - Inference: Anthropic Messages API via official SDK
    - Memory:    Sliding-window conversation history (inherited)
    - Tools:     Heuristic pre-check + result injection (inherited)
    - Safety:    Two-stage guardrails + Anthropic's own Constitutional AI (inherited + built-in)
    - Logging:   SQLite + JSONL (inherited)
    """

    MODEL_TYPE = "frontier"
    MODEL_NAME: str = config.FRONTIER_MODEL

    def __init__(self, session_id: Optional[str] = None):
        super().__init__(session_id)
        if not config.ANTHROPIC_API_KEY:
            raise ValueError(
                "ANTHROPIC_API_KEY is not set. "
                "Add it to your .env file to use the Frontier assistant."
            )
        self._client = anthropic.Anthropic(api_key=config.ANTHROPIC_API_KEY)

    def _generate_response(self) -> str:
        system_prompt, messages = self.memory.get_anthropic_messages()

        # Filter out any empty messages (safety)
        messages = [m for m in messages if m.get("content", "").strip()]

        try:
            response = self._client.messages.create(
                model=config.FRONTIER_MODEL,
                system=system_prompt,
                messages=messages,
                max_tokens=config.FRONTIER_MAX_TOKENS,
                temperature=config.FRONTIER_TEMPERATURE,
            )
            return response.content[0].text.strip()

        except anthropic.AuthenticationError:
            raise RuntimeError(
                "Anthropic API key is invalid. "
                "Please check ANTHROPIC_API_KEY in your .env file."
            )
        except anthropic.RateLimitError:
            raise RuntimeError(
                "Anthropic rate limit exceeded. "
                "Please wait a moment and try again."
            )
        except anthropic.APIStatusError as exc:
            raise RuntimeError(
                f"Anthropic API error {exc.status_code}: {exc.message}"
            )
        except Exception as exc:
            raise RuntimeError(f"Unexpected error calling Anthropic API: {exc}")

    def stream_response(self, user_message: str):
        """
        Generator that yields text tokens as they arrive (streaming mode).
        Useful for real-time UI updates.
        NOTE: Bypasses the full pipeline (safety/logging) — use `chat()` for production.
        """
        system_prompt, messages = self.memory.get_anthropic_messages()
        messages = [m for m in messages if m.get("content", "").strip()]
        messages.append({"role": "user", "content": user_message})

        with self._client.messages.stream(
            model=config.FRONTIER_MODEL,
            system=system_prompt,
            messages=messages,
            max_tokens=config.FRONTIER_MAX_TOKENS,
        ) as stream:
            for text in stream.text_stream:
                yield text
