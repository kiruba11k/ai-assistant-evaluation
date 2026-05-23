"""
Short-term conversational memory with sliding window.
Stores message history as OpenAI-compatible dicts so it works with both
the HF Inference API (which accepts the same format) and Anthropic's SDK.
"""

from __future__ import annotations
from dataclasses import dataclass, field
from typing import Literal
import time


Role = Literal["user", "assistant", "system"]


@dataclass
class Message:
    role: Role
    content: str
    timestamp: float = field(default_factory=time.time)

    def to_dict(self) -> dict:
        return {"role": self.role, "content": self.content}


class ConversationMemory:
    """
    Sliding-window conversation history.

    Keeps the system prompt pinned at position 0 and limits the number of
    user/assistant turn pairs that follow it.  Oldest turns are dropped first
    when the window is exceeded.
    """

    def __init__(self, system_prompt: str, max_turns: int = 10):
        self.system_prompt = system_prompt
        self.max_turns = max_turns          # max user+assistant PAIRS
        self._history: list[Message] = []  # only user & assistant messages

    #  Public API 

    def add_user(self, content: str) -> None:
        self._history.append(Message(role="user", content=content))
        self._trim()

    def add_assistant(self, content: str) -> None:
        self._history.append(Message(role="assistant", content=content))
        self._trim()

    def get_messages(self) -> list[dict]:
        """Return messages suitable for an OpenAI-format chat API."""
        return [{"role": "system", "content": self.system_prompt}] + [
            m.to_dict() for m in self._history
        ]

    def get_messages  # OpenAI-compat format works for Groq too(self) -> tuple[str, list[dict]]:
        """
        Return (system_prompt, messages) for Anthropic's SDK.
        Anthropic uses a separate `system` param, so we strip it from messages.
        """
        return self.system_prompt, [m.to_dict() for m in self._history]

    def clear(self) -> None:
        self._history.clear()

    def last_n_turns(self, n: int) -> list[dict]:
        """Return last n turn pairs as dicts (no system prompt)."""
        pairs = self._paired()
        return [m.to_dict() for pair in pairs[-n:] for m in pair]

    @property
    def turn_count(self) -> int:
        return len(self._paired())

    @property
    def message_count(self) -> int:
        return len(self._history)

    #  Private helpers 

    def _paired(self) -> list[list[Message]]:
        """Group history into [user, assistant] pairs."""
        pairs, i = [], 0
        while i < len(self._history) - 1:
            if (
                self._history[i].role == "user"
                and self._history[i + 1].role == "assistant"
            ):
                pairs.append([self._history[i], self._history[i + 1]])
                i += 2
            else:
                i += 1
        return pairs

    def _trim(self) -> None:
        """Drop oldest pairs when window is exceeded."""
        pairs = self._paired()
        if len(pairs) > self.max_turns:
            drop = pairs[: len(pairs) - self.max_turns]
            to_remove = {id(m) for pair in drop for m in pair}
            self._history = [m for m in self._history if id(m) not in to_remove]

    def __repr__(self) -> str:
        return (
            f"ConversationMemory("
            f"turns={self.turn_count}, "
            f"messages={self.message_count}, "
            f"max_turns={self.max_turns})"
        )
