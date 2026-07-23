import gradio as gr
import os

# from gradio_langgraph.backend.base_agent import base_agent
from gradio_langgraph.backend.rag_agent import base_agent
from gradio_langgraph.backend import chat

BASE_URL = os.getenv("BASE_URL", "http://localhost:8000/v1")


def create_app() -> gr.Blocks:
    with gr.Blocks(analytics_enabled=False) as demo:
        with gr.Row():
            with gr.Column(scale=1):
                with gr.Group():
                    vllm_url = gr.Textbox(
                        label="VLLM base url",
                        value=BASE_URL,
                        interactive=True
                    )
                    vllm_connect_btn = gr.Button(
                        value="Connect",
                        variant="primary"
                    )
            with gr.Column(scale=7):
                connection_status = gr.Label(
                    label="VLLM server connection",
                    show_label=True,
                    value=lambda: base_agent.model_name or "No connection"
                )

        vllm_connect_btn.click(
            fn=base_agent.set_connection,
            inputs=vllm_url,
            outputs=connection_status
        )

        with gr.Row():
            with gr.Column(scale=1):
                with gr.Group():
                    max_completion_tokens = gr.Slider(
                        label="Max completion tokens",
                        minimum=1, maximum=20000, step=1, value=20000
                    )
                    repetition_penalty = gr.Slider(
                        label="Repetition penalty",
                        minimum=0.0, maximum=1.0, step=0.1, value=1.0
                    )
                    temperature = gr.Slider(
                        label="Temperature",
                        minimum=0.0, maximum=2.0, step=0.1, value=0.3
                    )
            with gr.Column(scale=7):
                with gr.Tab("Chat"):
                    with gr.Accordion("System prompt", open=False):
                        system_textbox = gr.Textbox("", label="System prompt")
                    gr.ChatInterface(
                        chat.inference,
                        editable=True,
                        show_progress="full",
                        multimodal=False,
                        additional_inputs=[
                            system_textbox,
                            temperature
                        ],
                        concurrency_limit=5,
                        autofocus=True
                    ).chatbot.height = 600
    return demo
