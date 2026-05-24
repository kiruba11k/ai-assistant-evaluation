from __future__ import annotations

from typing import Optional

from openai import OpenAI

from app.assistants.base import BaseAssistant
from app.config import config


class OSSAssistant(BaseAssistant):

    MODEL_TYPE = "oss"

    MODEL_NAME: str = config.OSS_MODEL_ID

    def __init__(self, session_id: Optional[str] = None):
        super().__init__(session_id)

        if not config.HF_API_TOKEN:
            raise ValueError(
                "HF_API_TOKEN is not set.\n"
                "Create a token at:\n"
                "https://huggingface.co/settings/tokens"
            )

        self._client = OpenAI(
            base_url="https://router.huggingface.co/v1",
            api_key=config.HF_API_TOKEN,
        )

    def _generate_response(self) -> str:

        messages = self.memory.get_messages()

        messages = [
            m for m in messages
            if m.get("content", "").strip()
        ]

        try:

            response = self._client.chat.completions.create(
                model=config.OSS_MODEL_ID,
                messages=messages,
                max_tokens=config.OSS_MAX_TOKENS,
                temperature=config.OSS_TEMPERATURE,
            )

            content = response.choices[0].message.content

            if not content:
                return "No response generated."

            return content.strip()

        except Exception as exc:
            raise RuntimeError(
                f"Error calling HuggingFace Router API: {exc}"
            )

    def warmup(self) -> str:

        try:

            response = self._client.chat.completions.create(
                model=config.OSS_MODEL_ID,
                messages=[
                    {
                        "role": "user",
                        "content": "Hello"
                    }
                ],
                max_tokens=10,
                temperature=0.1,
            )

            if response:
                return "OSS model ready"

            return "OSS warmup failed"

        except Exception as exc:
            return f"Warmup failed: {exc}"
