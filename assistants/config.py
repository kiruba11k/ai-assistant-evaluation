import os
from dataclasses import dataclass, field
from typing import Optional


@dataclass
class ModelConfig:
    """Configuration for a single model."""
    name: str
    api_url: str
    max_tokens: int = 512
    temperature: float = 0.7
    top_p: float = 0.9


@dataclass
class AppConfig:
    # ── API Keys ──────────────────────────────────────────────────────────────
    ANTHROPIC_API_KEY: str = field(default_factory=lambda: os.getenv("ANTHROPIC_API_KEY", ""))
    HF_API_TOKEN: str = field(default_factory=lambda: os.getenv("HF_API_TOKEN", ""))
    OPENAI_API_KEY: str = field(default_factory=lambda: os.getenv("OPENAI_API_KEY", ""))

    # ── OSS Model (Qwen2.5 via HF Inference API) ──────────────────────────────
    OSS_MODEL_ID: str = "Qwen/Qwen2.5-0.5B-Instruct"
    OSS_API_URL: str = "https://api-inference.huggingface.co/models/Qwen/Qwen2.5-0.5B-Instruct/v1/chat/completions"
    OSS_MAX_TOKENS: int = 512
    OSS_TEMPERATURE: float = 0.7

    # ── Frontier Model (Claude Sonnet) ────────────────────────────────────────
    FRONTIER_MODEL: str = "claude-sonnet-4-20250514"
    FRONTIER_MAX_TOKENS: int = 1024
    FRONTIER_TEMPERATURE: float = 0.7

    # ── Memory ────────────────────────────────────────────────────────────────
    MAX_HISTORY_TURNS: int = 10          # number of user+assistant turn pairs kept
    SYSTEM_PROMPT: str = (
        "You are a helpful, harmless, and honest AI personal assistant. "
        "Answer clearly and concisely. If you are unsure, say so rather than guessing. "
        "Never provide dangerous, illegal, or harmful information."
    )

    # ── Safety / Guardrails ───────────────────────────────────────────────────
    ENABLE_INPUT_GUARDRAILS: bool = True
    ENABLE_OUTPUT_GUARDRAILS: bool = True
    SAFETY_JUDGE_MODEL: str = "claude-sonnet-4-20250514"

    # ── Observability ─────────────────────────────────────────────────────────
    LOG_DIR: str = "logs"
    LOG_FILE: str = "logs/conversations.jsonl"
    DB_FILE: str = "logs/metrics.db"
    ENABLE_LOGGING: bool = True

    # ── Evaluation ────────────────────────────────────────────────────────────
    EVAL_OUTPUT_DIR: str = "evaluation/results"
    EVAL_JUDGE_MODEL: str = "claude-sonnet-4-20250514"

    # ── App ───────────────────────────────────────────────────────────────────
    APP_TITLE: str = "AI Assistant Evaluation Suite"
    APP_PORT: int = 7860
    DEBUG: bool = os.getenv("DEBUG", "false").lower() == "true"

    def __post_init__(self):
        os.makedirs(self.LOG_DIR, exist_ok=True)
        os.makedirs(self.EVAL_OUTPUT_DIR, exist_ok=True)

    @property
    def oss_headers(self) -> dict:
        return {
            "Authorization": f"Bearer {self.HF_API_TOKEN}",
            "Content-Type": "application/json",
        }


# Singleton config instance
config = AppConfig()
