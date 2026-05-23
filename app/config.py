import os
from dataclasses import dataclass, field


@dataclass
class AppConfig:
    #  API Keys
    GROQ_API_KEY: str = field(default_factory=lambda: os.getenv("GROQ_API_KEY", ""))
    HF_API_TOKEN: str = field(default_factory=lambda: os.getenv("HF_API_TOKEN", ""))
    OPENWEATHER_API_KEY: str = field(default_factory=lambda: os.getenv("OPENWEATHER_API_KEY", ""))

    #  OSS Model (Qwen2.5 via HF Inference API — free) 
    OSS_MODEL_ID: str = "Qwen/Qwen2.5-0.5B-Instruct"
    OSS_API_URL: str = "https://api-inference.huggingface.co/models/Qwen/Qwen2.5-0.5B-Instruct/v1/chat/completions"
    OSS_MAX_TOKENS: int = 512
    OSS_TEMPERATURE: float = 0.7

    #  Frontier Model (Groq — free, ultra-fast) 
    # Options (all FREE on Groq):
    #   llama-3.3-70b-versatile   ← best quality
    #   llama-3.1-8b-instant      ← fastest
    #   mixtral-8x7b-32768        ← good balance
    #   gemma2-9b-it              ← lightweight
    FRONTIER_MODEL: str = os.getenv("GROQ_MODEL", "llama-3.3-70b-versatile")
    FRONTIER_MAX_TOKENS: int = 1024
    FRONTIER_TEMPERATURE: float = 0.7

    # Judge model (used for eval scoring + safety — pick fast model)
    JUDGE_MODEL: str = "llama-3.1-8b-instant"

    #  Memory
    MAX_HISTORY_TURNS: int = 10
    SYSTEM_PROMPT: str = (
        "You are a helpful, harmless, and honest AI personal assistant. "
        "Answer clearly and concisely. If you are unsure, say so rather than guessing. "
        "Never provide dangerous, illegal, or harmful information."
    )

    #  Safety / Guardrails 
    ENABLE_INPUT_GUARDRAILS: bool = True
    ENABLE_OUTPUT_GUARDRAILS: bool = True

    #  Observability 
    LOG_DIR: str = "logs"
    LOG_FILE: str = "logs/conversations.jsonl"
    DB_FILE: str = "logs/metrics.db"
    ENABLE_LOGGING: bool = True

    #  Evaluation 
    EVAL_OUTPUT_DIR: str = "evaluation/results"

    #  App
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

    @property
    def groq_available(self) -> bool:
        return bool(self.GROQ_API_KEY)

    @property
    def hf_available(self) -> bool:
        return bool(self.HF_API_TOKEN)


config = AppConfig()
