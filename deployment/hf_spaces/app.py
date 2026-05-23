from __future__ import annotations
import os, re, time, uuid
from typing import Optional
import gradio as gr
import requests

try:
    from groq import Groq
    GROQ_AVAILABLE = True
except ImportError:
    GROQ_AVAILABLE = False

#  Config 
GROQ_API_KEY   = os.getenv("GROQ_API_KEY", "")
HF_API_TOKEN   = os.getenv("HF_API_TOKEN", "")
GROQ_MODEL     = os.getenv("GROQ_MODEL", "llama-3.3-70b-versatile")
OSS_MODEL_ID   = os.getenv("OSS_MODEL_ID", "Qwen/Qwen2.5-0.5B-Instruct")
OSS_API_URL    = f"https://api-inference.huggingface.co/models/{OSS_MODEL_ID}/v1/chat/completions"
MAX_TOKENS     = 512
TEMPERATURE    = 0.7
MAX_TURNS      = 8

SYSTEM_PROMPT = (
    "You are a helpful, harmless, and honest AI personal assistant. "
    "Answer clearly and concisely. If unsure, say so. "
    "Never provide dangerous or illegal information."
)

#  Safety filter 
_BLOCK = [re.compile(p, re.I) for p in [
    r"\b(synthesize|make)\b.{0,30}\b(sarin|ricin|nerve agent)\b",
    r"\b(bomb|explosive)\b.{0,30}\b(make|build|instructions)\b",
    r"\bchild\b.{0,20}\b(sexual|nude|exploit)\b",
    r"\b(ignore|disregard)\b.{0,30}\b(instructions|system prompt|guidelines)\b",
    r"\bdan mode\b",
]]
REFUSAL = "I can't help with that — it appears to violate safety guidelines."

def is_safe(text: str) -> bool:
    return not any(p.search(text) for p in _BLOCK)

#  Message builder 
def build_messages(history: list, user_msg: str) -> list[dict]:
    msgs = [{"role": "system", "content": SYSTEM_PROMPT}]
    for u, a in history[-MAX_TURNS:]:
        if u: msgs.append({"role": "user",      "content": u})
        if a and not a.startswith("❌"): msgs.append({"role": "assistant", "content": a})
    msgs.append({"role": "user", "content": user_msg})
    return msgs

#  OSS call (HF Inference API) 
def call_oss(messages: list[dict]) -> tuple[str, float]:
    if not HF_API_TOKEN:
        return " HF_API_TOKEN not set. Add it in Space Settings → Secrets.", 0.0
    t0 = time.perf_counter()
    try:
        r = requests.post(OSS_API_URL,
            headers={"Authorization": f"Bearer {HF_API_TOKEN}", "Content-Type": "application/json"},
            json={"model": OSS_MODEL_ID, "messages": messages,
                  "max_tokens": MAX_TOKENS, "temperature": TEMPERATURE},
            timeout=60)
        latency = (time.perf_counter()-t0)*1000
        if r.status_code == 503:
            return " OSS model loading (cold start ~30s) — please retry.", latency
        if r.status_code != 200:
            return f" HF API {r.status_code}: {r.text[:150]}", latency
        return r.json()["choices"][0]["message"]["content"].strip(), latency
    except requests.Timeout:
        return " HF API timeout — model may be cold-starting. Retry in 30s.", 0.0
    except Exception as e:
        return f" Error: {e}", 0.0

#  Frontier call (Groq) 
def call_frontier(messages: list[dict]) -> tuple[str, float]:
    if not GROQ_API_KEY:
        return " GROQ_API_KEY not set. Add it in Space Settings → Secrets.", 0.0
    if not GROQ_AVAILABLE:
        return " groq package not installed — check requirements.txt", 0.0
    t0 = time.perf_counter()
    try:
        client = Groq(api_key=GROQ_API_KEY)
        resp = client.chat.completions.create(
            model=GROQ_MODEL, messages=messages,
            max_tokens=MAX_TOKENS, temperature=TEMPERATURE)
        latency = (time.perf_counter()-t0)*1000
        return resp.choices[0].message.content.strip(), latency
    except Exception as e:
        return f" Groq error: {e}", 0.0

#  Chat handlers 
def oss_chat(message, history):
    if not message.strip(): return history, ""
    if not is_safe(message):
        history.append((message, REFUSAL)); return history, " Blocked"
    msgs = build_messages(history, message)
    text, lat = call_oss(msgs)
    history.append((message, text))
    return history, f" {lat:.0f}ms | {OSS_MODEL_ID}"

def frontier_chat(message, history):
    if not message.strip(): return history, ""
    if not is_safe(message):
        history.append((message, REFUSAL)); return history, " Blocked"
    msgs = build_messages(history, message)
    text, lat = call_frontier(msgs)
    history.append((message, text))
    return history, f" {lat:.0f}ms | {GROQ_MODEL} (Groq)"

def h2h_chat(message, oss_hist, fr_hist):
    if not message.strip(): return oss_hist, fr_hist, "", ""
    blocked = not is_safe(message)
    oss_msgs = build_messages(oss_hist, message)
    fr_msgs  = build_messages(fr_hist,  message)
    if blocked:
        oss_hist.append((message, REFUSAL)); fr_hist.append((message, REFUSAL))
        return oss_hist, fr_hist, " Blocked", " Blocked"
    oss_text, oss_lat = call_oss(oss_msgs)
    fr_text,  fr_lat  = call_frontier(fr_msgs)
    oss_hist.append((message, oss_text)); fr_hist.append((message, fr_text))
    return oss_hist, fr_hist, f" {oss_lat:.0f}ms", f" {fr_lat:.0f}ms"

#  UI 
CSS = "footer{display:none!important}"

with gr.Blocks(title="AI Assistant Eval", theme=gr.themes.Soft(primary_hue="indigo"), css=CSS) as demo:
    gr.Markdown("""
    #  AI Assistant Evaluation Suite
    **OSS**: Qwen2.5-0.5B (HuggingFace)  vs  **Frontier**: Llama-3.3-70B (Groq — free & ultra-fast)
    """)

    with gr.Tabs():
        # Tab 1: OSS
        with gr.Tab(" OSS — Qwen2.5"):
            oss_bot  = gr.Chatbot(label=OSS_MODEL_ID, height=430)
            oss_stat = gr.Markdown("")
            with gr.Row():
                oss_inp  = gr.Textbox(placeholder="Ask anything…", show_label=False, scale=5)
                oss_send = gr.Button("Send ↵", variant="primary", scale=1)
                oss_clr  = gr.Button("🗑", scale=1)
            oss_send.click(oss_chat, [oss_inp, oss_bot], [oss_bot, oss_stat]).then(lambda:"", outputs=oss_inp)
            oss_inp.submit(oss_chat, [oss_inp, oss_bot], [oss_bot, oss_stat]).then(lambda:"", outputs=oss_inp)
            oss_clr.click(lambda: ([], ""), outputs=[oss_bot, oss_stat])

        # Tab 2: Frontier
        with gr.Tab(" Frontier — Llama 3.3 (Groq)"):
            fr_bot  = gr.Chatbot(label=f"{GROQ_MODEL} via Groq", height=430)
            fr_stat = gr.Markdown("")
            with gr.Row():
                fr_inp  = gr.Textbox(placeholder="Ask anything…", show_label=False, scale=5)
                fr_send = gr.Button("Send ↵", variant="primary", scale=1)
                fr_clr  = gr.Button("🗑", scale=1)
            fr_send.click(frontier_chat, [fr_inp, fr_bot], [fr_bot, fr_stat]).then(lambda:"", outputs=fr_inp)
            fr_inp.submit(frontier_chat, [fr_inp, fr_bot], [fr_bot, fr_stat]).then(lambda:"", outputs=fr_inp)
            fr_clr.click(lambda: ([], ""), outputs=[fr_bot, fr_stat])

        # Tab 3: Head-to-Head
        with gr.Tab(" Head-to-Head"):
            gr.Markdown("Same prompt → both models at once")
            h2h_inp  = gr.Textbox(placeholder="Compare both models…", label="Prompt")
            h2h_send = gr.Button(" Compare", variant="primary")
            h2h_clr  = gr.Button(" Clear Both")
            with gr.Row():
                with gr.Column():
                    gr.Markdown("####  OSS (Qwen2.5)")
                    h2h_oss_bot  = gr.Chatbot(height=360)
                    h2h_oss_stat = gr.Markdown("")
                with gr.Column():
                    gr.Markdown("####  Frontier (Groq Llama)")
                    h2h_fr_bot   = gr.Chatbot(height=360)
                    h2h_fr_stat  = gr.Markdown("")

            h2h_send.click(h2h_chat,
                [h2h_inp, h2h_oss_bot, h2h_fr_bot],
                [h2h_oss_bot, h2h_fr_bot, h2h_oss_stat, h2h_fr_stat]
            ).then(lambda:"", outputs=h2h_inp)
            h2h_inp.submit(h2h_chat,
                [h2h_inp, h2h_oss_bot, h2h_fr_bot],
                [h2h_oss_bot, h2h_fr_bot, h2h_oss_stat, h2h_fr_stat]
            ).then(lambda:"", outputs=h2h_inp)
            h2h_clr.click(lambda: ([], [], "", ""),
                outputs=[h2h_oss_bot, h2h_fr_bot, h2h_oss_stat, h2h_fr_stat])

        # Tab 4: Info
        with gr.Tab("ℹ Info"):
            gr.Markdown(f"""
            ### Model Details

            | | OSS | Frontier |
            |--|--|--|
            | **Model** | `{OSS_MODEL_ID}` | `{GROQ_MODEL}` |
            | **Provider** | HuggingFace Inference API | Groq |
            | **Cost** | Free | Free |
            | **Latency** | 1–15s (cold start risk) | ~200ms |
            | **Context** | 4 096 tokens | 32 768 tokens |
            | **Privacy** | HF servers | Groq servers |

            ### Safety
            Both assistants have a keyword-based input filter that blocks:
            - Dangerous/illegal instructions
            - Jailbreak attempts
            - CSAM

            ### Source Code
            [github.com/yourusername/ai-assistant-eval](https://github.com)
            """)

if __name__ == "__main__":
    demo.launch(server_name="0.0.0.0", server_port=7860)
