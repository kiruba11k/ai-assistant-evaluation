from __future__ import annotations
import os, re, time
import gradio as gr
from huggingface_hub import InferenceClient
from groq import Groq, RateLimitError, AuthenticationError

#  Config
HF_TOKEN       = os.getenv("HF_TOKEN", "")
GROQ_API_KEY   = os.getenv("GROQ_API_KEY", "")
OSS_MODEL      = os.getenv("OSS_MODEL",      "Qwen/Qwen2.5-0.5B-Instruct")
FRONTIER_MODEL = os.getenv("FRONTIER_MODEL", "llama-3.3-70b-versatile")
MAX_TOKENS     = int(os.getenv("MAX_TOKENS", "512"))
MAX_TURNS      = int(os.getenv("MAX_TURNS",  "8"))

SYSTEM_PROMPT = (
    "You are Qwen, created by Alibaba Cloud. You are a helpful assistant. "
    "Never provide dangerous, illegal, or harmful information."
)

#  InferenceClient with provider="hf-inference"
def get_hf_client() -> InferenceClient | None:
    if not HF_TOKEN:
        return None
    return InferenceClient(
        provider="hf-inference",   
        api_key=HF_TOKEN,
    )

_hf_client: InferenceClient | None = get_hf_client()

#  Safety filter
_BLOCK = [re.compile(p, re.I) for p in [
    r"\b(synthesize|make|create)\b.{0,30}\b(sarin|ricin|nerve agent)\b",
    r"\b(bomb|explosive)\b.{0,30}\b(make|build|how to)\b",
    r"\bchild\b.{0,20}\b(sexual|nude|exploit)\b",
    r"\b(ignore|disregard)\b.{0,30}\b(instructions|system prompt|guidelines)\b",
    r"\bdan mode\b",
]]
REFUSAL = "I cannot help with that request. Please ask something else."

def is_safe(text: str) -> bool:
    return not any(p.search(text) for p in _BLOCK)

#  Message builder
def build_messages(history: list, user_msg: str) -> list[dict]:
    msgs = [{"role": "system", "content": SYSTEM_PROMPT}]
    for u, a in history[-MAX_TURNS:]:
        if u: msgs.append({"role": "user",      "content": u})
        if a and not a.startswith("["): msgs.append({"role": "assistant", "content": a})
    msgs.append({"role": "user", "content": user_msg})
    return msgs

#  OSS: Qwen2.5 via HF Inference API (router.huggingface.co) 
def call_oss(messages: list[dict]) -> tuple[str, float]:
    if not HF_TOKEN:
        return (
            "[HF_TOKEN not set]\n"
            "Add it in Space Settings → Secrets → New Secret\n"
            "Name: HF_TOKEN | Value: your token from huggingface.co/settings/tokens",
            0.0,
        )

    global _hf_client
    if _hf_client is None:
        _hf_client = get_hf_client()

    t0 = time.perf_counter()
    try:
        completion = _hf_client.chat.completions.create(
            model=OSS_MODEL,
            messages=messages,
            max_tokens=MAX_TOKENS,
        )
        latency = (time.perf_counter() - t0) * 1000
        return completion.choices[0].message.content.strip(), latency

    except Exception as exc:
        latency = (time.perf_counter() - t0) * 1000
        err = str(exc)
        if "401" in err or "403" in err:
            return "[Invalid HF_TOKEN — generate a new Read token at huggingface.co/settings/tokens]", latency
        if "429" in err or "rate" in err.lower():
            return "[HF rate limit hit — wait 60s and retry]", latency
        if "503" in err or "loading" in err.lower():
            return "[Model cold-starting on HF — wait 20s and retry]", latency
        return f"[HF API error: {err[:150]}]", latency

#  Frontier: Llama-3.3-70B via Groq
def call_frontier(messages: list[dict]) -> tuple[str, float]:
    if not GROQ_API_KEY:
        return (
            "[GROQ_API_KEY not set]\n"
            "Add it in Space Settings → Secrets → New Secret\n"
            "Name: GROQ_API_KEY | Value: your key from console.groq.com",
            0.0,
        )
    t0 = time.perf_counter()
    try:
        client = Groq(api_key=GROQ_API_KEY)
        resp   = client.chat.completions.create(
            model=FRONTIER_MODEL,
            messages=messages,
            max_tokens=MAX_TOKENS,
        )
        latency = (time.perf_counter() - t0) * 1000
        return resp.choices[0].message.content.strip(), latency
    except AuthenticationError:
        return "[Invalid GROQ_API_KEY — check console.groq.com → API Keys]", 0.0
    except RateLimitError:
        return "[Groq rate limit — wait 60s and retry]", 0.0
    except Exception as e:
        return f"[Groq error: {str(e)[:150]}]", 0.0

#  Chat handlers
def oss_chat(message: str, history: list):
    if not message.strip(): return history, ""
    if not is_safe(message):
        history.append((message, REFUSAL)); return history, "Blocked"
    msgs = build_messages(history, message)
    text, lat = call_oss(msgs)
    history.append((message, text))
    return history, f"{lat:.0f} ms  |  {OSS_MODEL}  |  HF Inference API"

def frontier_chat(message: str, history: list):
    if not message.strip(): return history, ""
    if not is_safe(message):
        history.append((message, REFUSAL)); return history, "Blocked"
    msgs = build_messages(history, message)
    text, lat = call_frontier(msgs)
    history.append((message, text))
    return history, f"{lat:.0f} ms  |  {FRONTIER_MODEL}  |  Groq"

def h2h_chat(message: str, oh: list, fh: list):
    if not message.strip(): return oh, fh, "", ""
    if not is_safe(message):
        oh.append((message, REFUSAL)); fh.append((message, REFUSAL))
        return oh, fh, "Blocked", "Blocked"
    ot, ol = call_oss(build_messages(oh, message))
    ft, fl = call_frontier(build_messages(fh, message))
    oh.append((message, ot)); fh.append((message, ft))
    return (
        oh, fh,
        f"{ol:.0f} ms  |  {OSS_MODEL}  |  HF Inference API",
        f"{fl:.0f} ms  |  {FRONTIER_MODEL}  |  Groq",
    )

#  Gradio UI
with gr.Blocks(
    title="AI Assistant Evaluation Suite",
    theme=gr.themes.Soft(primary_hue="indigo"),
    css="footer{display:none!important}",
) as demo:

    gr.Markdown(
        "# AI Assistant Evaluation Suite\n"
        f"**OSS:** `{OSS_MODEL}` via HF Inference API (router.huggingface.co)\n\n"
        f"**Frontier:** `{FRONTIER_MODEL}` via Groq"
    )

    with gr.Tabs():

        with gr.Tab("OSS — Qwen2.5-0.5B-Instruct"):
            oss_bot  = gr.Chatbot(label=OSS_MODEL, height=440)
            oss_stat = gr.Markdown("Ready — requires HF_TOKEN in Space Secrets")
            with gr.Row():
                oss_inp  = gr.Textbox(placeholder="Ask anything…", show_label=False, scale=5)
                oss_send = gr.Button("Send", variant="primary", scale=1)
                oss_clr  = gr.Button("Clear", scale=1)
            oss_send.click(oss_chat, [oss_inp, oss_bot], [oss_bot, oss_stat]).then(lambda:"", outputs=oss_inp)
            oss_inp.submit(oss_chat, [oss_inp, oss_bot], [oss_bot, oss_stat]).then(lambda:"", outputs=oss_inp)
            oss_clr.click(lambda: ([], "Ready"), outputs=[oss_bot, oss_stat])

        with gr.Tab("Frontier — Llama-3.3-70B (Groq)"):
            fr_bot  = gr.Chatbot(label=FRONTIER_MODEL, height=440)
            fr_stat = gr.Markdown("Ready — requires GROQ_API_KEY in Space Secrets")
            with gr.Row():
                fr_inp  = gr.Textbox(placeholder="Ask anything…", show_label=False, scale=5)
                fr_send = gr.Button("Send", variant="primary", scale=1)
                fr_clr  = gr.Button("Clear", scale=1)
            fr_send.click(frontier_chat, [fr_inp, fr_bot], [fr_bot, fr_stat]).then(lambda:"", outputs=fr_inp)
            fr_inp.submit(frontier_chat, [fr_inp, fr_bot], [fr_bot, fr_stat]).then(lambda:"", outputs=fr_inp)
            fr_clr.click(lambda: ([], "Ready"), outputs=[fr_bot, fr_stat])

        with gr.Tab("Head-to-Head"):
            gr.Markdown("Same prompt sent to both models simultaneously.")
            h2h_inp = gr.Textbox(placeholder="Compare both models…", label="Prompt")
            with gr.Row():
                h2h_send = gr.Button("Compare Both", variant="primary")
                h2h_clr  = gr.Button("Clear Both")
            with gr.Row():
                with gr.Column():
                    gr.Markdown(f"**OSS — {OSS_MODEL}**")
                    h2h_oss  = gr.Chatbot(height=360)
                    h2h_os   = gr.Markdown("Ready")
                with gr.Column():
                    gr.Markdown(f"**Frontier — {FRONTIER_MODEL}**")
                    h2h_fr   = gr.Chatbot(height=360)
                    h2h_fs   = gr.Markdown("Ready")
            h2h_send.click(h2h_chat, [h2h_inp, h2h_oss, h2h_fr], [h2h_oss, h2h_fr, h2h_os, h2h_fs]).then(lambda:"", outputs=h2h_inp)
            h2h_inp.submit(h2h_chat, [h2h_inp, h2h_oss, h2h_fr], [h2h_oss, h2h_fr, h2h_os, h2h_fs]).then(lambda:"", outputs=h2h_inp)
            h2h_clr.click(lambda: ([], [], "Ready", "Ready"), outputs=[h2h_oss, h2h_fr, h2h_os, h2h_fs])

        with gr.Tab("Info"):
            gr.Markdown(f"""
## Why the DNS Error Happened

```
OLD (deprecated — DNS removed by HuggingFace):
  api-inference.huggingface.co  ← this hostname no longer resolves

NEW (active):
  router.huggingface.co         ← HF migrated all inference here
```

`InferenceClient(provider="hf-inference")` automatically uses the new URL.
No model weights are downloaded. Pure API call — fast, lightweight.

## Secrets Required

| Secret | Tab | Where to get |
|--------|-----|--------------|
| `HF_TOKEN` | OSS | huggingface.co/settings/tokens (free Read token) |
| `GROQ_API_KEY` | Frontier | console.groq.com (free) |

## Models

| | OSS | Frontier |
|--|--|--|
| Model | `{OSS_MODEL}` | `{FRONTIER_MODEL}` |
| Endpoint | router.huggingface.co | Groq API |
| Cost | Free (HF serverless) | Free (Groq free tier) |
| Latency | 2-10s (shared GPU) | ~200ms |
| Weights downloaded | None | None |
            """)

if __name__ == "__main__":
    demo.launch(server_name="0.0.0.0", server_port=7860)
