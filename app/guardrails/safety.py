"""
Guardrails / Safety Layer
Stage 1 — Fast keyword + regex filter (no API call, microseconds)
Stage 2 — LLM-based safety judge via Groq (fast, free)
"""

from __future__ import annotations
import re, time
from dataclasses import dataclass
from enum import Enum
from typing import Optional


class SafetyDecision(str, Enum):
    SAFE    = "safe"
    BLOCKED = "blocked"
    WARN    = "warn"


@dataclass
class SafetyResult:
    decision: SafetyDecision
    reason: str
    stage: int
    latency_ms: float
    original_text: str
    sanitized_text: Optional[str] = None

    @property
    def is_blocked(self) -> bool:
        return self.decision == SafetyDecision.BLOCKED

    @property
    def display_text(self) -> str:
        return self.sanitized_text or self.original_text


#  Stage 1 patterns 

_HARD_BLOCK = [re.compile(p, re.I) for p in [
    r"\b(synthesize|make|create|build)\b.{0,40}\b(nerve agent|sarin|vx|ricin|botulinum|anthrax|smallpox)\b",
    r"\b(bomb|explosive|ied)\b.{0,30}\b(make|build|create|instructions|how to)\b",
    r"\bchild\b.{0,20}\b(sexual|nude|naked|porn|abuse|exploit)\b",
    r"\b(csam|cp)\b",
    r"\b(ransomware|keylogger|rootkit|rat|trojan)\b.{0,30}\b(code|script|write|create|build)\b",
    r"\b(how to|best way to|method for)\b.{0,30}\b(kill myself|commit suicide|self harm)\b",
]]

_SOFT_BLOCK = [re.compile(p, re.I) for p in [
    r"\b(ignore|forget|disregard)\b.{0,30}\b(your instructions|system prompt|guidelines|rules)\b",
    r"\bdan mode\b|\byou are now\b.{0,20}\b(unrestricted|jailbroken|without limits)\b",
    r"\bpretend you (have no|don't have)\b.{0,20}\b(restrictions|ethics|guidelines)\b",
    r"\b(hack|crack|exploit)\b.{0,30}\b(password|account|system|network|server)\b",
]]

_WARN = [re.compile(p, re.I) for p in [
    r"\b(overdose|drug|dosage)\b",
    r"\b(weapons|firearms|gun)\b",
    r"\b(steal|theft|robbery)\b",
]]

_PII = [
    (re.compile(r"\b\d{3}-\d{2}-\d{4}\b"),                              "[SSN REDACTED]"),
    (re.compile(r"\b(?:\d{4}[- ]?){3}\d{4}\b"),                         "[CC# REDACTED]"),
    (re.compile(r"\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}\b"),"[EMAIL REDACTED]"),
    (re.compile(r"\b(\+?\d{1,3}[\s\-]?)?\(?\d{3}\)?[\s\-]?\d{3}[\s\-]?\d{4}\b"), "[PHONE REDACTED]"),
]


def _keyword_check(text: str) -> SafetyResult:
    t0 = time.perf_counter()
    for pat in _HARD_BLOCK:
        if pat.search(text):
            return SafetyResult(SafetyDecision.BLOCKED, f"Hard-blocked: {pat.pattern[:50]}", 1,
                                (time.perf_counter()-t0)*1000, text)
    for pat in _SOFT_BLOCK:
        if pat.search(text):
            return SafetyResult(SafetyDecision.BLOCKED, f"Jailbreak attempt detected", 1,
                                (time.perf_counter()-t0)*1000, text)
    for pat in _WARN:
        if pat.search(text):
            return SafetyResult(SafetyDecision.WARN, f"Flagged: {pat.pattern[:50]}", 1,
                                (time.perf_counter()-t0)*1000, text)
    return SafetyResult(SafetyDecision.SAFE, "Passed keyword filter", 1,
                        (time.perf_counter()-t0)*1000, text)


#  Stage 2: Groq LLM judge 

_SAFETY_SYSTEM = """You are a safety classifier. Return ONLY JSON, no markdown:
{"decision": "safe|warn|block", "reason": "brief reason under 15 words"}

block = illegal, harmful instructions, jailbreak attempt
warn  = sensitive but not directly harmful
safe  = everything else"""


def _groq_judge(text: str, api_key: str) -> SafetyResult:
    t0 = time.perf_counter()
    try:
        from groq import Groq
        import json
        client = Groq(api_key=api_key)
        resp = client.chat.completions.create(
            model="llama-3.1-8b-instant",   # fastest model for safety checks
            max_tokens=60,
            messages=[
                {"role": "system", "content": _SAFETY_SYSTEM},
                {"role": "user", "content": f"Classify:\n{text[:600]}"},
            ],
        )
        raw = resp.choices[0].message.content.strip()
        raw = raw.replace("```json","").replace("```","").strip()
        data = json.loads(raw)
        decision_map = {"block": SafetyDecision.BLOCKED, "warn": SafetyDecision.WARN,
                        "safe": SafetyDecision.SAFE}
        decision = decision_map.get(data.get("decision","safe"), SafetyDecision.SAFE)
        return SafetyResult(decision, data.get("reason","LLM judge"), 2,
                            (time.perf_counter()-t0)*1000, text)
    except Exception as exc:
        # Fail open on errors
        return SafetyResult(SafetyDecision.SAFE, f"Judge error (fail-open): {exc}", 2,
                            (time.perf_counter()-t0)*1000, text)


#  Public API 

def scrub_pii(text: str) -> str:
    for pat, repl in _PII:
        text = pat.sub(repl, text)
    return text


def check_input(text: str, *, use_llm_judge: bool = True, api_key: str = "") -> SafetyResult:
    result = _keyword_check(text)
    if result.decision == SafetyDecision.BLOCKED:
        return result
    if use_llm_judge and api_key and result.decision == SafetyDecision.WARN:
        return _groq_judge(text, api_key)
    return result


def check_output(text: str, *, scrub_pii_data: bool = True,
                 use_llm_judge: bool = False, api_key: str = "") -> SafetyResult:
    sanitized = scrub_pii(text) if scrub_pii_data else text
    result = _keyword_check(sanitized)
    result.sanitized_text = sanitized if sanitized != text else None
    return result


REFUSAL_MESSAGES = {
    SafetyDecision.BLOCKED: (
        "I'm not able to help with that request — it appears to violate safety guidelines. "
        "Please rephrase or ask something else."
    ),
}

def get_refusal(result: SafetyResult) -> str:
    return REFUSAL_MESSAGES.get(result.decision,
        "I can respond but please note this is a sensitive topic.")
