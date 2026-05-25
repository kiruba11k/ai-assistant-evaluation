
from __future__ import annotations
import os, re, time
import gradio as gr
from huggingface_hub import InferenceClient

try:
    from groq import Groq
    GROQ_AVAILABLE = True
except ImportError:
    GROQ_AVAILABLE = False

#  Config 
GROQ_API_KEY = os.getenv("GROQ_API_KEY", "")
HF_API_TOKEN = os.getenv("HF_API_TOKEN", "")
GROQ_MODEL   = os.getenv("GROQ_MODEL",   "llama-3.3-70b-versatile")
OSS_MODEL_ID = os.getenv("OSS_MODEL_ID", "Qwen/Qwen2.5-0.5B-Instruct")
MAX_TOKENS   = int(os.getenv("MAX_TOKENS",  "512"))
TEMPERATURE  = float(os.getenv("TEMPERATURE", "0.7"))
MAX_TURNS    = int(os.getenv("MAX_TURNS",   "8"))

SYSTEM_PROMPT = (
    "You are a helpful, harmless, and honest AI personal assistant. "
    "Answer clearly and concisely. If unsure, say so. "
    "Never provide dangerous or illegal information."
)

#  InferenceClient (replaces raw requests — works inside HF Spaces) 
def get_oss_client() -> InferenceClient | None:
    if not HF_API_TOKEN:
        return None
    return InferenceClient(model=OSS_MODEL_ID, token=HF_API_TOKEN)

# Initialise once at startup
_oss_client: InferenceClient | None = get_oss_client()

#  Safety filter 
_BLOCK = [re.compile(p, re.I) for p in [
    r"\b(synthesize|make)\b.{0,30}\b(sarin|ricin|nerve agent)\b",
    r"\b(bomb|explosive)\b.{0,30}\b(make|build|instructions)\b",
    r"\bchild\b.{0,20}\b(sexual|nude|exploit)\b",
    r"\b(ignore|disregard)\b.{0,30}\b(instructions|system prompt|guidelines)\b",
    r"\bdan mode\b",
]]
REFUSAL = (
    "I cannot help with that request as it appears to violate safety guidelines. "
    "Please ask something else."
)

def is_safe(text: str) -> bool:
    return not any(p.search(text) for p in _BLOCK)

#  Message builder 
def build_messages(history: list[tuple], user_msg: str) -> list[dict]:
    msgs = [{"role": "system", "content": SYSTEM_PROMPT}]
    for u, a in history[-MAX_TURNS:]:
        if u:
            msgs.append({"role": "user",      "content": u})
        if a and not a.startswith("Error") and not a.startswith("I cannot"):
            msgs.append({"role": "assistant", "content": a})
    msgs.append({"role": "user", "content": user_msg})
    return msgs

#  OSS call via InferenceClient ─
def call_oss(messages: list[dict]) -> tuple[str, float]:
    if not HF_API_TOKEN:
        return (
            "HF_API_TOKEN not set. "
            "Add it in Space Settings -> Secrets -> New Secret -> HF_API_TOKEN",
            0.0,
        )

    global _oss_client
    if _oss_client is None:
        _oss_client = get_oss_client()

    t0 = time.perf_counter()
    try:
        response = _oss_client.chat.completions.create(
            messages=messages,
            max_tokens=MAX_TOKENS,
            temperature=TEMPERATURE,
            stream=False,
        )
        latency = (time.perf_counter() - t0) * 1000
        return response.choices[0].message.content.strip(), latency

    except Exception as exc:
        latency = (time.perf_counter() - t0) * 1000
        err = str(exc)

        # Model loading (cold start on free tier)
        if "loading" in err.lower() or "503" in err:
            return (
                "OSS model is loading (cold start on HF free tier). "
                "Please wait 20-30 seconds and send your message again.",
                latency,
            )
        # Token / auth error
        if "401" in err or "403" in err or "token" in err.lower():
            return (
                "HF_API_TOKEN is invalid or expired. "
                "Generate a new Read token at huggingface.co/settings/tokens",
                latency,
            )
        # Rate limit
        if "429" in err or "rate" in err.lower():
            return (
                "HF Inference API rate limit reached. "
                "Wait a minute and try again, or upgrade your HF account.",
                latency,
            )
        return f"OSS model error: {err[:200]}", latency

#  Frontier call via Groq ─
def call_frontier(messages: list[dict]) -> tuple[str, float]:
    if not GROQ_API_KEY:
        return (
            "GROQ_API_KEY not set. "
            "Add it in Space Settings -> Secrets -> New Secret -> GROQ_API_KEY",
            0.0,
        )
    if not GROQ_AVAILABLE:
        return "groq package not installed. Check requirements.txt.", 0.0

    t0 = time.perf_counter()
    try:
        client = Groq(api_key=GROQ_API_KEY)
        resp = client.chat.completions.create(
            model=GROQ_MODEL,
            messages=messages,
            max_tokens=MAX_TOKENS,
            temperature=TEMPERATURE,
        )
        latency = (time.perf_counter() - t0) * 1000
        return resp.choices[0].message.content.strip(), latency

    except Exception as exc:
        latency = (time.perf_counter() - t0) * 1000
        err = str(exc)

        if "401" in err or "auth" in err.lower():
            return (
                "GROQ_API_KEY is invalid. "
                "Check your key at console.groq.com -> API Keys",
                latency,
            )
        if "429" in err or "rate" in err.lower():
            return (
                "Groq free tier rate limit hit (100K tokens/day for 70B model). "
                "Wait 60 seconds or switch GROQ_MODEL to llama-3.1-8b-instant "
                "in Space Secrets for a higher daily limit.",
                latency,
            )
        return f"Groq error: {err[:200]}", latency

#  Chat handlers 
def oss_chat(message: str, history: list) -> tuple[list, str]:
    if not message.strip():
        return history, ""
    if not is_safe(message):
        history.append((message, REFUSAL))
        return history, "Blocked by safety filter"

    msgs = build_messages(history, message)
    text, lat = call_oss(msgs)
    history.append((message, text))
    status = f"{lat:.0f}ms  |  {OSS_MODEL_ID}" if lat > 0 else "Error"
    return history, status


def frontier_chat(message: str, history: list) -> tuple[list, str]:
    if not message.strip():
        return history, ""
    if not is_safe(message):
        history.append((message, REFUSAL))
        return history, "Blocked by safety filter"

    msgs = build_messages(history, message)
    text, lat = call_frontier(msgs)
    history.append((message, text))
    status = f"{lat:.0f}ms  |  {GROQ_MODEL} via Groq" if lat > 0 else "Error"
    return history, status


def h2h_chat(
    message: str,
    oss_hist: list,
    fr_hist: list,
) -> tuple[list, list, str, str]:
    if not message.strip():
        return oss_hist, fr_hist, "", ""

    if not is_safe(message):
        oss_hist.append((message, REFUSAL))
        fr_hist.append((message, REFUSAL))
        return oss_hist, fr_hist, "Blocked", "Blocked"

    oss_msgs = build_messages(oss_hist, message)
    fr_msgs  = build_messages(fr_hist,  message)

    oss_text, oss_lat = call_oss(oss_msgs)
    fr_text,  fr_lat  = call_frontier(fr_msgs)

    oss_hist.append((message, oss_text))
    fr_hist.append((message,  fr_text))

    return (
        oss_hist, fr_hist,
        f"{oss_lat:.0f}ms  |  {OSS_MODEL_ID}",
        f"{fr_lat:.0f}ms  |  {GROQ_MODEL} via Groq",
    )

#  Gradio UI ─
CSS = "footer { display: none !important; }"

with gr.Blocks(
    title="AI Assistant Evaluation Suite",
    theme=gr.themes.Soft(primary_hue="indigo"),
    css=CSS,
) as demo:

    gr.Markdown(
        "# AI Assistant Evaluation Suite\n"
        "**OSS:** Qwen2.5-0.5B (HuggingFace)  vs  "
        "**Frontier:** Llama-3.3-70B (Groq)  —  both on free APIs"
    )

    with gr.Tabs():

        #  Tab 1: OSS 
        with gr.Tab("OSS — Qwen2.5-0.5B"):
            oss_bot  = gr.Chatbot(label=OSS_MODEL_ID, height=430)
            oss_stat = gr.Markdown("Ready")
            with gr.Row():
                oss_inp  = gr.Textbox(
                    placeholder="Ask anything... (first message may take 30s to cold-start)",
                    show_label=False, scale=5,
                )
                oss_send = gr.Button("Send", variant="primary", scale=1)
                oss_clr  = gr.Button("Clear", scale=1)

            oss_send.click(
                oss_chat, [oss_inp, oss_bot], [oss_bot, oss_stat]
            ).then(lambda: "", outputs=oss_inp)

            oss_inp.submit(
                oss_chat, [oss_inp, oss_bot], [oss_bot, oss_stat]
            ).then(lambda: "", outputs=oss_inp)

            oss_clr.click(lambda: ([], "Ready"), outputs=[oss_bot, oss_stat])

        #  Tab 2: Frontier ─
        with gr.Tab("Frontier — Llama-3.3-70B (Groq)"):
            fr_bot  = gr.Chatbot(label=f"{GROQ_MODEL} via Groq", height=430)
            fr_stat = gr.Markdown("Ready")
            with gr.Row():
                fr_inp  = gr.Textbox(
                    placeholder="Ask anything...",
                    show_label=False, scale=5,
                )
                fr_send = gr.Button("Send", variant="primary", scale=1)
                fr_clr  = gr.Button("Clear", scale=1)

            fr_send.click(
                frontier_chat, [fr_inp, fr_bot], [fr_bot, fr_stat]
            ).then(lambda: "", outputs=fr_inp)

            fr_inp.submit(
                frontier_chat, [fr_inp, fr_bot], [fr_bot, fr_stat]
            ).then(lambda: "", outputs=fr_inp)

            fr_clr.click(lambda: ([], "Ready"), outputs=[fr_bot, fr_stat])

        #  Tab 3: Head-to-Head ─
        with gr.Tab("Head-to-Head"):
            gr.Markdown("Same prompt sent to both models at the same time.")
            h2h_inp  = gr.Textbox(placeholder="Enter a prompt to compare both models...", label="Prompt")
            h2h_send = gr.Button("Compare Both", variant="primary")
            h2h_clr  = gr.Button("Clear Both")

            with gr.Row():
                with gr.Column():
                    gr.Markdown(f"**OSS — {OSS_MODEL_ID}**")
                    h2h_oss_bot  = gr.Chatbot(height=360)
                    h2h_oss_stat = gr.Markdown("Ready")
                with gr.Column():
                    gr.Markdown(f"**Frontier — {GROQ_MODEL} (Groq)**")
                    h2h_fr_bot   = gr.Chatbot(height=360)
                    h2h_fr_stat  = gr.Markdown("Ready")

            h2h_send.click(
                h2h_chat,
                [h2h_inp, h2h_oss_bot, h2h_fr_bot],
                [h2h_oss_bot, h2h_fr_bot, h2h_oss_stat, h2h_fr_stat],
            ).then(lambda: "", outputs=h2h_inp)

            h2h_inp.submit(
                h2h_chat,
                [h2h_inp, h2h_oss_bot, h2h_fr_bot],
                [h2h_oss_bot, h2h_fr_bot, h2h_oss_stat, h2h_fr_stat],
            ).then(lambda: "", outputs=h2h_inp)

            h2h_clr.click(
                lambda: ([], [], "Ready", "Ready"),
                outputs=[h2h_oss_bot, h2h_fr_bot, h2h_oss_stat, h2h_fr_stat],
            )

        #  Tab 4: Model Info ─
        with gr.Tab("Model Info"):
            gr.Markdown(f"""
## Model Details

| | OSS | Frontier |
|--|--|--|
| Model | `{OSS_MODEL_ID}` | `{GROQ_MODEL}` |
| Provider | HuggingFace Inference API | Groq |
| Client | `huggingface_hub.InferenceClient` | `groq` SDK |
| Cost | Free | Free |
| Cold-start | 10-30 seconds (first call) | Under 1 second |
| Warm latency | 1-5 seconds | ~200 ms |
| Context window | 4,096 tokens | 32,768 tokens |

## Safety

Both assistants block the following via keyword filter:
- Dangerous / illegal instructions
- Jailbreak attempts (DAN mode, ignore instructions, etc.)
- Child safety violations

## Rate Limits (free tiers)

| Provider | Model | Daily Limit | Per-minute |
|--|--|--|--|
| HuggingFace | Qwen2.5-0.5B | No hard cap | Shared GPU |
| Groq | llama-3.3-70b-versatile | 100K tokens | 6K tokens |
| Groq | llama-3.1-8b-instant | 500K tokens | 20K tokens |

To switch to the faster Groq model, add a Space Secret:
`GROQ_MODEL = llama-3.1-8b-instant`
            """)

if __name__ == "__main__":
    demo.launch(server_name="0.0.0.0", server_port=7860)
