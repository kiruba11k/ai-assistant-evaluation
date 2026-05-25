from __future__ import annotations

import time
from typing import Optional

import torch
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
)

from app.assistants.base import BaseAssistant
from app.config import config


class OSSAssistant(BaseAssistant):

    MODEL_TYPE = "oss"
    MODEL_NAME: str = config.OSS_MODEL_ID

    _tokenizer = None
    _model = None

    def __init__(self, session_id: Optional[str] = None):
        super().__init__(session_id)

        if OSSAssistant._tokenizer is None:
            print(f"Loading model: {self.MODEL_NAME}")

            OSSAssistant._tokenizer = AutoTokenizer.from_pretrained(
                self.MODEL_NAME
            )

            OSSAssistant._model = AutoModelForCausalLM.from_pretrained(
                self.MODEL_NAME,
                torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
                device_map="auto",
            )

            print("Model loaded successfully.")

    @property
    def tokenizer(self):
        return OSSAssistant._tokenizer

    @property
    def model(self):
        return OSSAssistant._model

    def _generate_response(self) -> str:
        messages = self.memory.get_messages()

        t0 = time.perf_counter()

        try:
            prompt = self.tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True,
            )

            inputs = self.tokenizer(
                prompt,
                return_tensors="pt",
            ).to(self.model.device)

            outputs = self.model.generate(
                **inputs,
                max_new_tokens=config.OSS_MAX_TOKENS,
                temperature=config.OSS_TEMPERATURE,
                do_sample=True,
                top_p=0.9,
                repetition_penalty=1.1,
            )

            generated_tokens = outputs[0][inputs.input_ids.shape[-1]:]

            text = self.tokenizer.decode(
                generated_tokens,
                skip_special_tokens=True,
            )

            elapsed = (time.perf_counter() - t0) * 1000

            print(f"OSS latency: {elapsed:.0f} ms")

            return text.strip()

        except Exception as exc:
            raise RuntimeError(f"OSS model generation failed: {exc}")

    def warmup(self) -> str:
        try:
            t0 = time.perf_counter()

            prompt = self.tokenizer.apply_chat_template(
                [{"role": "user", "content": "Hi"}],
                tokenize=False,
                add_generation_prompt=True,
            )

            inputs = self.tokenizer(
                prompt,
                return_tensors="pt",
            ).to(self.model.device)

            _ = self.model.generate(
                **inputs,
                max_new_tokens=10,
            )

            elapsed = (time.perf_counter() - t0) * 1000

            return f"OSS model ready ({elapsed:.0f} ms)"

        except Exception as exc:
            return f"Warmup failed: {exc}"
