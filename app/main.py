from __future__ import annotations

import os
import time
import uuid
from typing import Optional

import gradio as gr

from app.assistants.frontier_assistant import FrontierAssistant
from app.assistants.oss_assistant import OSSAssistant
from app.config import config
from app.observability.tracker import tracker
from app.tools.tool_registry import TOOL_DESCRIPTIONS

#  Session singletons (one per user via Gradio State)

def make_oss() -> OSSAssistant:
    return OSSAssistant(session_id=str(uuid.uuid4()))

def make_frontier() -> FrontierAssistant:
    return FrontierAssistant(session_id=str(uuid.uuid4()))


#  Chat handlers

def oss_chat(message: str, history: list, assistant_state: Optional[OSSAssistant]):
    if not message.strip():
        return history, history, assistant_state

    if assistant_state is None:
        try:
            assistant_state = make_oss()
        except Exception as exc:
            history.append((message, f" Init error: {exc}"))
            return history, history, assistant_state

    resp = assistant_state.chat(message)
    prefix = ""
    if resp.tool_used:
        prefix = f" *Used tool: {resp.tool_used}*\n\n"
    if resp.safety_input.decision.value == "warn":
        prefix += " *Sensitive topic — response may be limited.*\n\n"

    display = prefix + resp.display_text
    history.append((message, display))

    status = f"⏱ {resp.latency_ms:.0f}ms | 🛡 {resp.safety_input.decision.value}"
    if resp.error:
        status += f" |  {resp.error[:50]}"
    return history, history, assistant_state, status


def frontier_chat(message: str, history: list, assistant_state: Optional[FrontierAssistant]):
    if not message.strip():
        return history, history, assistant_state

    if assistant_state is None:
        try:
            assistant_state = make_frontier()
        except Exception as exc:
            history.append((message, f" Init error: {exc}"))
            return history, history, assistant_state, ""

    resp = assistant_state.chat(message)
    prefix = ""
    if resp.tool_used:
        prefix = f" *Used tool: {resp.tool_used}*\n\n"
    if resp.safety_input.decision.value == "warn":
        prefix += " *Sensitive topic — response may be limited.*\n\n"

    display = prefix + resp.display_text
    history.append((message, display))

    status = f" {resp.latency_ms:.0f}ms |  {resp.safety_input.decision.value}"
    if resp.error:
        status += f" |  {resp.error[:50]}"
    return history, history, assistant_state, status


def h2h_chat(message: str, oss_hist: list, frontier_hist: list,
             oss_state, frontier_state):
    """Head-to-head: run both models on the same prompt."""
    if not message.strip():
        return oss_hist, frontier_hist, oss_state, frontier_state, "", ""

    # Init if needed
    if oss_state is None:
        try:
            oss_state = make_oss()
        except Exception as exc:
            oss_hist.append((message, f" {exc}"))
            frontier_hist.append((message, ""))
            return oss_hist, frontier_hist, oss_state, frontier_state, "", ""

    if frontier_state is None:
        try:
            frontier_state = make_frontier()
        except Exception as exc:
            oss_hist.append((message, ""))
            frontier_hist.append((message, f" {exc}"))
            return oss_hist, frontier_hist, oss_state, frontier_state, "", ""

    oss_resp = oss_state.chat(message)
    frontier_resp = frontier_state.chat(message)

    oss_hist.append((message, oss_resp.display_text))
    frontier_hist.append((message, frontier_resp.display_text))

    oss_status = f"⏱ {oss_resp.latency_ms:.0f}ms | 🛡 {oss_resp.safety_input.decision.value}"
    frontier_status = f"⏱ {frontier_resp.latency_ms:.0f}ms | 🛡 {frontier_resp.safety_input.decision.value}"

    return oss_hist, frontier_hist, oss_state, frontier_state, oss_status, frontier_status


def get_stats() -> str:
    stats = tracker.get_stats()
    if not stats:
        return "No conversations logged yet."
    lines = ["###  Live Observability Stats\n"]
    for model_type, row in stats.items():
        label = " OSS (Qwen2.5)" if model_type == "oss" else " Frontier (Claude)"
        lines.append(f"**{label}**")
        lines.append(f"- Total turns: {row.get('total_turns', 0)}")
        lines.append(f"- Avg latency: {row.get('avg_latency_ms', 0):.0f} ms")
        lines.append(f"- Blocked inputs: {row.get('blocked_inputs', 0)}")
        lines.append(f"- Errors: {row.get('errors', 0)}")
        lines.append(f"- Avg response tokens (est.): {row.get('avg_completion_tokens', 0):.0f}")
        lines.append("")
    return "\n".join(lines)


#  CSS

CUSTOM_CSS = """
.tab-label { font-weight: 600; }
.status-bar { font-size: 0.8em; color: #6b7280; }
.chatbot .message { border-radius: 12px; }
footer { display: none !important; }
"""

TOOL_INFO = "\n".join(
    f"- **{name}**: {desc}" for name, desc in TOOL_DESCRIPTIONS.items()
)


#  Build UI

def build_app() -> gr.Blocks:
    with gr.Blocks(
        title=config.APP_TITLE,
        css=CUSTOM_CSS,
        theme=gr.themes.Soft(primary_hue="blue"),
    ) as demo:

        gr.Markdown(f"#  {config.APP_TITLE}")
        gr.Markdown(
            "Compare **OSS** (Qwen2.5 via HuggingFace) vs **Frontier** (Claude Sonnet) personal assistants. "
            "Both support multi-turn memory, tools, and safety guardrails."
        )

        with gr.Tabs():

            #  Tab 1: OSS Assistant
            with gr.Tab(" OSS Assistant (Qwen2.5)"):
                oss_state = gr.State(None)
                oss_history_state = gr.State([])

                with gr.Row():
                    with gr.Column(scale=3):
                        oss_chatbot = gr.Chatbot(
                            label="Qwen2.5-0.5B-Instruct",
                            height=450,
                            show_label=True,
                        )
                        oss_status = gr.Markdown("", elem_classes=["status-bar"])
                        with gr.Row():
                            oss_input = gr.Textbox(
                                placeholder="Ask me anything…",
                                show_label=False,
                                scale=4,
                            )
                            oss_send = gr.Button("Send ↵", variant="primary", scale=1)
                            oss_clear = gr.Button("🗑 Clear", scale=1)
                    with gr.Column(scale=1):
                        gr.Markdown("### ℹ Model Info")
                        gr.Markdown(
                            f"**Model:** `{config.OSS_MODEL_ID}`\n\n"
                            f"**API:** HuggingFace Inference API\n\n"
                            f"**Max turns in memory:** {config.MAX_HISTORY_TURNS}\n\n"
                            f"**Max tokens:** {config.OSS_MAX_TOKENS}\n\n"
                            "---\n###  Available Tools\n" + TOOL_INFO
                        )

                def oss_send_fn(msg, hist, state):
                    if not config.HF_API_TOKEN:
                        hist.append((msg, " HF_API_TOKEN not set. Add it to your .env file."))
                        return hist, hist, state, ""
                    result = oss_chat(msg, hist, state)
                    return result[0], result[1], result[2], result[3] if len(result) > 3 else ""

                oss_send.click(
                    oss_send_fn,
                    inputs=[oss_input, oss_history_state, oss_state],
                    outputs=[oss_chatbot, oss_history_state, oss_state, oss_status],
                ).then(lambda: "", outputs=oss_input)

                oss_input.submit(
                    oss_send_fn,
                    inputs=[oss_input, oss_history_state, oss_state],
                    outputs=[oss_chatbot, oss_history_state, oss_state, oss_status],
                ).then(lambda: "", outputs=oss_input)

                def oss_clear_fn(state):
                    if state:
                        state.reset()
                    return [], [], state, ""

                oss_clear.click(oss_clear_fn, inputs=[oss_state],
                                outputs=[oss_chatbot, oss_history_state, oss_state, oss_status])

            #  Tab 2: Frontier Assistant
            with gr.Tab(" Frontier Assistant (Claude)"):
                frontier_state = gr.State(None)
                frontier_history_state = gr.State([])

                with gr.Row():
                    with gr.Column(scale=3):
                        frontier_chatbot = gr.Chatbot(
                            label="Groq (Llama 3.3 70B)",
                            height=450,
                            show_label=True,
                        )
                        frontier_status = gr.Markdown("", elem_classes=["status-bar"])
                        with gr.Row():
                            frontier_input = gr.Textbox(
                                placeholder="Ask me anything…",
                                show_label=False,
                                scale=4,
                            )
                            frontier_send = gr.Button("Send ↵", variant="primary", scale=1)
                            frontier_clear = gr.Button("🗑 Clear", scale=1)
                    with gr.Column(scale=1):
                        gr.Markdown("###  Model Info")
                        gr.Markdown(
                            f"**Model:** `{config.FRONTIER_MODEL}`\n\n"
                            f"**API:** Anthropic Messages API\n\n"
                            f"**Max turns in memory:** {config.MAX_HISTORY_TURNS}\n\n"
                            f"**Max tokens:** {config.FRONTIER_MAX_TOKENS}\n\n"
                            "---\n###  Available Tools\n" + TOOL_INFO
                        )

                def frontier_send_fn(msg, hist, state):
                    if not config.GROQ_API_KEY:
                        hist.append((msg, " GROQ_API_KEY not set. Add it to your .env file."))
                        return hist, hist, state, ""
                    result = frontier_chat(msg, hist, state)
                    return result[0], result[1], result[2], result[3] if len(result) > 3 else ""

                frontier_send.click(
                    frontier_send_fn,
                    inputs=[frontier_input, frontier_history_state, frontier_state],
                    outputs=[frontier_chatbot, frontier_history_state, frontier_state, frontier_status],
                ).then(lambda: "", outputs=frontier_input)

                frontier_input.submit(
                    frontier_send_fn,
                    inputs=[frontier_input, frontier_history_state, frontier_state],
                    outputs=[frontier_chatbot, frontier_history_state, frontier_state, frontier_status],
                ).then(lambda: "", outputs=frontier_input)

                def frontier_clear_fn(state):
                    if state:
                        state.reset()
                    return [], [], state, ""

                frontier_clear.click(frontier_clear_fn, inputs=[frontier_state],
                                     outputs=[frontier_chatbot, frontier_history_state, frontier_state, frontier_status])

            #  Tab 3: Head-to-Head
            with gr.Tab(" Head-to-Head"):
                h2h_oss_state = gr.State(None)
                h2h_frontier_state = gr.State(None)
                h2h_oss_hist = gr.State([])
                h2h_frontier_hist = gr.State([])

                gr.Markdown("### Same prompt → both models simultaneously")
                h2h_input = gr.Textbox(
                    placeholder="Enter a prompt to send to both assistants…",
                    label="Prompt",
                )
                h2h_send = gr.Button(" Compare Both", variant="primary")
                h2h_clear = gr.Button("🗑 Clear Both")

                with gr.Row():
                    with gr.Column():
                        gr.Markdown("####  OSS (Qwen2.5)")
                        h2h_oss_bot = gr.Chatbot(height=380)
                        h2h_oss_status = gr.Markdown("", elem_classes=["status-bar"])
                    with gr.Column():
                        gr.Markdown("####  Frontier (Claude)")
                        h2h_frontier_bot = gr.Chatbot(height=380)
                        h2h_frontier_status = gr.Markdown("", elem_classes=["status-bar"])

                def h2h_send_fn(msg, oh, fh, os_, fs_):
                    if not msg.strip():
                        return oh, fh, os_, fs_, "", ""

                    # Init OSS
                    if os_ is None:
                        if not config.HF_API_TOKEN:
                            oh.append((msg, " HF_API_TOKEN not configured"))
                            os_ = None
                        else:
                            try:
                                os_ = make_oss()
                            except Exception as exc:
                                oh.append((msg, f" {exc}"))
                                os_ = None

                    # Init Frontier
                    if fs_ is None:
                        if not config.GROQ_API_KEY:
                            fh.append((msg, " GROQ_API_KEY not configured"))
                            fs_ = None
                        else:
                            try:
                                fs_ = make_frontier()
                            except Exception as exc:
                                fh.append((msg, f" {exc}"))
                                fs_ = None

                    return h2h_chat(msg, oh, fh, os_, fs_)

                h2h_send.click(
                    h2h_send_fn,
                    inputs=[h2h_input, h2h_oss_hist, h2h_frontier_hist,
                            h2h_oss_state, h2h_frontier_state],
                    outputs=[h2h_oss_bot, h2h_frontier_bot,
                             h2h_oss_state, h2h_frontier_state,
                             h2h_oss_status, h2h_frontier_status],
                ).then(lambda: "", outputs=h2h_input)

                h2h_input.submit(
                    h2h_send_fn,
                    inputs=[h2h_input, h2h_oss_hist, h2h_frontier_hist,
                            h2h_oss_state, h2h_frontier_state],
                    outputs=[h2h_oss_bot, h2h_frontier_bot,
                             h2h_oss_state, h2h_frontier_state,
                             h2h_oss_status, h2h_frontier_status],
                ).then(lambda: "", outputs=h2h_input)

                def h2h_clear_fn(os_, fs_):
                    if os_: os_.reset()
                    if fs_: fs_.reset()
                    return [], [], os_, fs_, "", ""

                h2h_clear.click(
                    h2h_clear_fn,
                    inputs=[h2h_oss_state, h2h_frontier_state],
                    outputs=[h2h_oss_bot, h2h_frontier_bot,
                             h2h_oss_state, h2h_frontier_state,
                             h2h_oss_status, h2h_frontier_status],
                )

            #  Tab 4: Live Stats
            with gr.Tab(" Live Stats"):
                gr.Markdown("### Observability Dashboard")
                gr.Markdown(
                    "Metrics are written to SQLite (`logs/metrics.db`) and JSONL (`logs/conversations.jsonl`). "
                    "Refresh to see latest stats."
                )
                stats_display = gr.Markdown(get_stats())
                refresh_btn = gr.Button(" Refresh Stats")
                refresh_btn.click(get_stats, outputs=stats_display)

                gr.Markdown("---")
                gr.Markdown(
                    "### 🏃 Run Full Evaluation\n"
                    "```bash\n"
                    "python -m evaluation.run_evaluation\n"
                    "```\n"
                    "Results saved to `evaluation/results/`"
                )

    return demo


if __name__ == "__main__":
    app = build_app()
    app.launch(
        server_name="0.0.0.0",
        server_port=config.APP_PORT,
        share=False,
        show_error=True,
    )
