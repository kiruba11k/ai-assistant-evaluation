import os
from dataclasses import dataclass, field


@dataclass
class AppConfig:
    GROQ_API_KEY: str = field(
        default_factory=lambda: os.getenv("GROQ_API_KEY", "")
    )

    HF_API_TOKEN: str = field(
        default_factory=lambda: os.getenv("HF_API_TOKEN", "")
    )

    OPENWEATHER_API_KEY: str = field(
        default_factory=lambda: os.getenv("OPENWEATHER_API_KEY", "")
    )

    OSS_MODEL_ID: str = "Qwen/Qwen2.5-0.5B-Instruct"

    OSS_API_URL: str = "https://router.huggingface.co/v1"

    OSS_MAX_TOKENS: int = 512

    OSS_TEMPERATURE: float = 0.7

    FRONTIER_MODEL: str = os.getenv(
        "GROQ_MODEL",
        "llama-3.3-70b-versatile"
    )

    FRONTIER_MAX_TOKENS: int = 1024

    FRONTIER_TEMPERATURE: float = 0.7

    JUDGE_MODEL: str = "llama-3.1-8b-instant"

    MAX_HISTORY_TURNS: int = 10

    SYSTEM_PROMPT: str = (
        "You are a helpful, harmless, and honest AI personal assistant. "
        "Answer clearly and concisely. "
        "If unsure, say so instead of hallucinating. "
        "Never provide dangerous, illegal, or harmful content."
    )

    ENABLE_INPUT_GUARDRAILS: bool = True

    ENABLE_OUTPUT_GUARDRAILS: bool = True

    LOG_DIR: str = "logs"

    LOG_FILE: str = "logs/conversations.jsonl"

    DB_FILE: str = "logs/metrics.db"

    ENABLE_LOGGING: bool = True

    EVAL_OUTPUT_DIR: str = "evaluation/results"

    APP_TITLE: str = "AI Assistant Evaluation Suite"

    APP_PORT: int = 7860

    DEBUG: bool = (
        os.getenv("DEBUG", "false").lower() == "true"
    )

    def __post_init__(self):
        os.makedirs(self.LOG_DIR, exist_ok=True)
        os.makedirs(self.EVAL_OUTPUT_DIR, exist_ok=True)

    @property
    def hf_headers(self) -> dict:
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

    def validate(self):
        warnings = []

        if not self.GROQ_API_KEY:
            warnings.append("Missing GROQ_API_KEY")

        if not self.HF_API_TOKEN:
            warnings.append("Missing HF_API_TOKEN")

        return warnings


config = AppConfig()


if __name__ == "__main__":

    print("=" * 50)
    print(config.APP_TITLE)
    print("=" * 50)

    warnings = config.validate()

    if warnings:
        print("\nConfiguration Warnings:")

        for warning in warnings:
            print(f" - {warning}")

    else:
        print("\nAll API keys loaded successfully.")

    print("\nConfiguration Loaded.")
