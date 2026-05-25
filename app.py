from __future__ import annotations
import os
import re
import time
import torch
import gradio as gr

from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
)

from groq import (
    Groq,
    RateLimitError,
    AuthenticationError,
)

GROQ_API_KEY = os.getenv("GROQ_API_KEY", "")

OSS_MODEL = "Qwen/Qwen2.5-0.5B-Instruct"
FRONTIER_MODEL = "llama-3.3-70b-versatile"

MAX_TOKENS = int(os.getenv("MAX_TOKENS", "512"))
MAX_TURNS = int(os.getenv("MAX_TURNS", "8"))

SYSTEM_PROMPT = (
    "You are Qwen, created by Alibaba Cloud. "
    "You are a helpful assistant. "
    "Never provide dangerous, illegal, or harmful information."
)

print("Loading Qwen model...")

tokenizer = AutoTokenizer.from_pretrained(
    OSS_MODEL
)

model = AutoModelForCausalLM.from_pretrained(
    OSS_MODEL,
    torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
    device_map="auto"
)

print("Qwen model loaded successfully.")

_BLOCK = [
    re.compile(p, re.I)
    for p in [
        r"\b(synthesize|make|create)\b.{0,30}\b(sarin|ricin|nerve agent)\b",
        r"\b(bomb|explosive)\b.{0,30}\b(make|build|how to)\b",
        r"\bchild\b.{0,20}\b(sexual|nude|exploit)\b",
        r"\b(ignore|disregard)\b.{0,30}\b(instructions|system prompt|guidelines)\b",
        r"\bdan mode\b",
    ]
]

REFUSAL = "I cannot help with that request. Please ask something else."


def is_safe(text: str) -> bool:
    return not any(p.search(text) for p in _BLOCK)


def build_messages(history: list, user_msg: str) -> list[dict]:
    msgs = [{"role": "system", "content": SYSTEM_PROMPT}]

    for u, a in history[-MAX_TURNS:]:
        if u:
            msgs.append({"role": "user", "content": u})

        if a and not a.startswith("["):
            msgs.append({"role": "assistant", "content": a})

    msgs.append({"role": "user", "content": user_msg})

    return msgs


def call_oss(messages: list[dict]) -> tuple[str, float]:
    t0 = time.perf_counter()

    try:
        prompt = tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
        )

        inputs = tokenizer(
            prompt,
            return_tensors="pt"
        ).to(model.device)

        outputs = model.generate(
            **inputs,
            max_new_tokens=MAX_TOKENS,
            temperature=0.7,
            do_sample=True,
            top_p=0.9,
            repetition_penalty=1.1,
        )

        generated_tokens = outputs[0][inputs.input_ids.shape[-1]:]

        text = tokenizer.decode(
            generated_tokens,
            skip_special_tokens=True
        )

        latency = (time.perf_counter() - t0) * 1000

        return text.strip(), latency

    except Exception as exc:
        latency = (time.perf_counter() - t0) * 1000
        return f"[Qwen error: {str(exc)[:200]}]", latency


def call_frontier(messages: list[dict]) -> tuple[str, float]:
    if not GROQ_API_KEY:
        return (
            "[GROQ_API_KEY not set]\n"
            "Add it in Space Settings → Secrets → New Secret\n"
            "Name: GROQ_API_KEY",
            0.0,
        )

    t0 = time.perf_counter()

    try:
        client = Groq(api_key=GROQ_API_KEY)

        resp = client.chat.completions.create(
            model=FRONTIER_MODEL,
            messages=messages,
            max_tokens=MAX_TOKENS,
        )

        latency = (time.perf_counter() - t0) * 1000

        return resp.choices[0].message.content.strip(), latency

    except AuthenticationError:
        return "[Invalid GROQ_API_KEY]", 0.0

    except RateLimitError:
        return "[Groq rate limit hit — wait and retry]", 0.0

    except Exception as e:
        return f"[Groq error: {str(e)[:200]}]", 0.0


def oss_chat(message: str, history: list):
    if not message.strip():
        return history, ""

    if not is_safe(message):
        history.append((message, REFUSAL))
        return history, "Blocked"

    msgs = build_messages(history, message)

    text, lat = call_oss(msgs)

    history.append((message, text))

    return history, f"{lat:.0f} ms  |  {OSS_MODEL}  |  Local Transformers"


def frontier_chat(message: str, history: list):
    if not message.strip():
        return history, ""

    if not is_safe(message):
        history.append((message, REFUSAL))
        return history, "Blocked"

    msgs = build_messages(history, message)

    text, lat = call_frontier(msgs)

    history.append((message, text))

    return history, f"{lat:.0f} ms  |  {FRONTIER_MODEL}  |  Groq"


def h2h_chat(message: str, oh: list, fh: list):
    if not message.strip():
        return oh, fh, "", ""

    if not is_safe(message):
        oh.append((message, REFUSAL))
        fh.append((message, REFUSAL))

        return oh, fh, "Blocked", "Blocked"

    ot, ol = call_oss(build_messages(oh, message))
    ft, fl = call_frontier(build_messages(fh, message))

    oh.append((message, ot))
    fh.append((message, ft))

    return (
        oh,
        fh,
        f"{ol:.0f} ms  |  {OSS_MODEL}",
        f"{fl:.0f} ms  |  {FRONTIER_MODEL}",
    )


with gr.Blocks(
    title="AI Assistant Evaluation Suite",
    theme=gr.themes.Soft(primary_hue="indigo"),
    css="footer{display:none!important}",
) as demo:

    gr.Markdown(
        f"""
# AI Assistant Evaluation Suite

### OSS Model
`{OSS_MODEL}` running locally in Hugging Face Spaces

### Frontier Model
`{FRONTIER_MODEL}` via Groq
"""
    )

    with gr.Tabs():

        with gr.Tab("OSS — Qwen2.5-0.5B-Instruct"):

            oss_bot = gr.Chatbot(
                label=OSS_MODEL,
                height=440,
            )

            oss_stat = gr.Markdown(
                "Ready — running locally inside Hugging Face Space"
            )

            with gr.Row():

                oss_inp = gr.Textbox(
                    placeholder="Ask anything...",
                    show_label=False,
                    scale=5,
                )

                oss_send = gr.Button(
                    "Send",
                    variant="primary",
                    scale=1,
                )

                oss_clr = gr.Button(
                    "Clear",
                    scale=1,
                )

            oss_send.click(
                oss_chat,
                [oss_inp, oss_bot],
                [oss_bot, oss_stat],
            ).then(lambda: "", outputs=oss_inp)

            oss_inp.submit(
                oss_chat,
                [oss_inp, oss_bot],
                [oss_bot, oss_stat],
            ).then(lambda: "", outputs=oss_inp)

            oss_clr.click(
                lambda: ([], "Ready"),
                outputs=[oss_bot, oss_stat],
            )

        with gr.Tab("Frontier — Llama-3.3-70B"):

            fr_bot = gr.Chatbot(
                label=FRONTIER_MODEL,
                height=440,
            )

            fr_stat = gr.Markdown(
                "Ready — requires GROQ_API_KEY"
            )

            with gr.Row():

                fr_inp = gr.Textbox(
                    placeholder="Ask anything...",
                    show_label=False,
                    scale=5,
                )

                fr_send = gr.Button(
                    "Send",
                    variant="primary",
                    scale=1,
                )

                fr_clr = gr.Button(
                    "Clear",
                    scale=1,
                )

            fr_send.click(
                frontier_chat,
                [fr_inp, fr_bot],
                [fr_bot, fr_stat],
            ).then(lambda: "", outputs=fr_inp)

            fr_inp.submit(
                frontier_chat,
                [fr_inp, fr_bot],
                [fr_bot, fr_stat],
            ).then(lambda: "", outputs=fr_inp)

            fr_clr.click(
                lambda: ([], "Ready"),
                outputs=[fr_bot, fr_stat],
            )

        with gr.Tab("Head-to-Head"):

            gr.Markdown(
                "Same prompt sent to both models simultaneously."
            )

            h2h_inp = gr.Textbox(
                placeholder="Compare both models...",
                label="Prompt",
            )

            with gr.Row():

                h2h_send = gr.Button(
                    "Compare Both",
                    variant="primary",
                )

                h2h_clr = gr.Button(
                    "Clear Both"
                )

            with gr.Row():

                with gr.Column():

                    gr.Markdown(f"## OSS — {OSS_MODEL}")

                    h2h_oss = gr.Chatbot(height=360)

                    h2h_os = gr.Markdown("Ready")

                with gr.Column():

                    gr.Markdown(f"## Frontier — {FRONTIER_MODEL}")

                    h2h_fr = gr.Chatbot(height=360)

                    h2h_fs = gr.Markdown("Ready")

            h2h_send.click(
                h2h_chat,
                [h2h_inp, h2h_oss, h2h_fr],
                [h2h_oss, h2h_fr, h2h_os, h2h_fs],
            ).then(lambda: "", outputs=h2h_inp)

            h2h_inp.submit(
                h2h_chat,
                [h2h_inp, h2h_oss, h2h_fr],
                [h2h_oss, h2h_fr, h2h_os, h2h_fs],
            ).then(lambda: "", outputs=h2h_inp)

            h2h_clr.click(
                lambda: ([], [], "Ready", "Ready"),
                outputs=[h2h_oss, h2h_fr, h2h_os, h2h_fs],
            )

        with gr.Tab("Info"):

            gr.Markdown(
                f"""
## Architecture

| Component | Technology |
|---|---|
| OSS Model | `{OSS_MODEL}` |
| OSS Runtime | Local Transformers |
| Frontier Model | `{FRONTIER_MODEL}` |
| Frontier Runtime | Groq API |
| UI | Gradio |
| Hosting | Hugging Face Spaces |

"""
            )

if __name__ == "__main__":
    demo.launch(
        server_name="0.0.0.0",
        server_port=7860,
    )
