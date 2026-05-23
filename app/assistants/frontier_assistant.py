from __future__ import annotations
from typing import Optional, Iterator
from groq import Groq, RateLimitError, AuthenticationError, APIStatusError

from app.assistants.base import BaseAssistant
from app.config import config


class FrontierAssistant(BaseAssistant):
    MODEL_TYPE = "frontier"
    MODEL_NAME: str = config.FRONTIER_MODEL

    def __init__(self, session_id: Optional[str] = None):
        super().__init__(session_id)
        if not config.GROQ_API_KEY:
            raise ValueError(
                "GROQ_API_KEY is not set.\n"
                "Get your FREE key at https://console.groq.com\n"
                "Then add it to your .env file: GROQ_API_KEY=gsk_..."
            )
        self._client = Groq(api_key=config.GROQ_API_KEY)

    def _generate_response(self) -> str:
        # Groq uses OpenAI-compatible format — get_messages() works directly
        messages = self.memory.get_messages()
        messages = [m for m in messages if m.get("content", "").strip()]

        try:
            response = self._client.chat.completions.create(
                model=config.FRONTIER_MODEL,
                messages=messages,
                max_tokens=config.FRONTIER_MAX_TOKENS,
                temperature=config.FRONTIER_TEMPERATURE,
            )
            return response.choices[0].message.content.strip()

        except AuthenticationError:
            raise RuntimeError(
                "Invalid GROQ_API_KEY. Check your key at https://console.groq.com"
            )
        except RateLimitError:
            raise RuntimeError(
                "Groq rate limit hit. Free tier: 6K–20K tokens/min. "
                "Wait 60 seconds and retry, or switch to llama-3.1-8b-instant."
            )
        except APIStatusError as exc:
            raise RuntimeError(f"Groq API error {exc.status_code}: {exc.message}")
        except Exception as exc:
            raise RuntimeError(f"Unexpected error calling Groq API: {exc}")

    def stream_response(self, user_message: str) -> Iterator[str]:
        """Stream tokens as they arrive (for real-time UI)."""
        messages = self.memory.get_messages()
        messages = [m for m in messages if m.get("content", "").strip()]
        messages.append({"role": "user", "content": user_message})

        with self._client.chat.completions.stream(
            model=config.FRONTIER_MODEL,
            messages=messages,
            max_tokens=config.FRONTIER_MAX_TOKENS,
        ) as stream:
            for chunk in stream:
                delta = chunk.choices[0].delta.content
                if delta:
                    yield delta
