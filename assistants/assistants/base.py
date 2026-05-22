"""
Abstract base class for all assistants.
Concrete implementations override `_generate_response()`.
"""

from __future__ import annotations

import time
import uuid
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Optional

from app.config import config
from app.guardrails.safety import (
    SafetyDecision, SafetyResult, check_input, check_output, get_refusal
)
from app.memory.conversation_memory import ConversationMemory
from app.observability.tracker import tracker
from app.tools.tool_registry import detect_and_call_tool


@dataclass
class AssistantResponse:
    text: str
    latency_ms: float
    tool_used: Optional[str]
    tool_result: Optional[str]
    safety_input: SafetyResult
    safety_output: SafetyResult
    error: Optional[str]
    session_id: str
    model_name: str
    model_type: str

    @property
    def was_blocked(self) -> bool:
        return self.safety_input.decision == SafetyDecision.BLOCKED

    @property
    def display_text(self) -> str:
        """What to show in the UI."""
        if self.was_blocked:
            return get_refusal(self.safety_input)
        return self.safety_output.display_text or self.text


class BaseAssistant(ABC):
    """
    Common pipeline for both OSS and Frontier assistants:

        1. Input safety check
        2. (Optional) Tool pre-check
        3. Generate response via model
        4. Output safety check
        5. Log to observability tracker
        6. Return AssistantResponse
    """

    MODEL_TYPE: str = "base"
    MODEL_NAME: str = "unknown"

    def __init__(self, session_id: Optional[str] = None):
        self.session_id = session_id or str(uuid.uuid4())
        self.memory = ConversationMemory(
            system_prompt=config.SYSTEM_PROMPT,
            max_turns=config.MAX_HISTORY_TURNS,
        )
        tracker.start_session(self.session_id, self.MODEL_TYPE)

    # ── Public interface ────────────────────────────────────────────────────

    def chat(self, user_message: str) -> AssistantResponse:
        """Process one user turn and return the full response object."""
        t_start = time.perf_counter()

        # 1. Input safety
        safety_in = check_input(
            user_message,
            use_llm_judge=config.ENABLE_INPUT_GUARDRAILS,
            api_key=config.ANTHROPIC_API_KEY,
        )

        if safety_in.is_blocked:
            response_text = get_refusal(safety_in)
            latency = (time.perf_counter() - t_start) * 1000
            safety_out = SafetyResult(
                decision=SafetyDecision.SAFE, reason="Blocked before generation",
                stage=0, latency_ms=0, original_text="",
            )
            resp = AssistantResponse(
                text=response_text, latency_ms=latency,
                tool_used=None, tool_result=None,
                safety_input=safety_in, safety_output=safety_out,
                error=None, session_id=self.session_id,
                model_name=self.MODEL_NAME, model_type=self.MODEL_TYPE,
            )
            tracker.record_turn(
                session_id=self.session_id, model_type=self.MODEL_TYPE,
                model_name=self.MODEL_NAME, user_message=user_message,
                assistant_message=response_text, latency_ms=latency,
                safety_input=safety_in, safety_output=safety_out,
            )
            return resp

        # 2. Tool pre-check (lightweight, no API call)
        tool_result: Optional[str] = None
        tool_name: Optional[str] = None
        tool_result = detect_and_call_tool(user_message)
        if tool_result:
            tool_name = "auto-detect"

        # 3. Add user message to memory
        self.memory.add_user(user_message)

        # 4. Generate response
        error: Optional[str] = None
        try:
            if tool_result:
                # Inject tool result into context
                augmented_message = (
                    f"{user_message}\n\n[Tool result: {tool_result}]"
                )
                # Temporarily swap last user message
                self.memory._history[-1].content = augmented_message
                response_text = self._generate_response()
                self.memory._history[-1].content = user_message
            else:
                response_text = self._generate_response()
        except Exception as exc:
            error = str(exc)
            response_text = (
                "I apologize — I encountered an error generating a response. "
                "Please try again."
            )

        latency = (time.perf_counter() - t_start) * 1000

        # 5. Output safety
        safety_out = check_output(
            response_text,
            scrub_pii_data=True,
            use_llm_judge=False,   # Output judge is off by default to save cost
            api_key=config.ANTHROPIC_API_KEY,
        )
        final_text = safety_out.display_text or response_text

        # 6. Add assistant response to memory
        self.memory.add_assistant(final_text)

        # 7. Log
        tracker.record_turn(
            session_id=self.session_id, model_type=self.MODEL_TYPE,
            model_name=self.MODEL_NAME, user_message=user_message,
            assistant_message=final_text, latency_ms=latency,
            safety_input=safety_in, safety_output=safety_out,
            tool_called=tool_name, tool_result=tool_result,
            error=error,
        )

        return AssistantResponse(
            text=final_text, latency_ms=latency,
            tool_used=tool_name, tool_result=tool_result,
            safety_input=safety_in, safety_output=safety_out,
            error=error, session_id=self.session_id,
            model_name=self.MODEL_NAME, model_type=self.MODEL_TYPE,
        )

    def reset(self) -> None:
        """Clear conversation memory."""
        self.memory.clear()

    def get_history(self) -> list[tuple[str, str]]:
        """Return [(user, assistant), ...] pairs for Gradio Chatbot."""
        pairs = self.memory._paired()
        return [(p[0].content, p[1].content) for p in pairs]

    # ── Abstract ────────────────────────────────────────────────────────────

    @abstractmethod
    def _generate_response(self) -> str:
        """
        Generate assistant response using self.memory.
        Must return a plain string.
        """
        ...
