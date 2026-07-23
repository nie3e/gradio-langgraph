import time
import gradio as gr
from langchain_core.messages import AIMessage, HumanMessage
# from gradio_langgraph.backend.base_agent import base_agent
from gradio_langgraph.backend.rag_agent import base_agent

YIELD_INTERVAL = 0.1


def filter_messages(chatbot: list) -> list:
    result = []
    for message in chatbot:
        if message.get("metadata"):
            continue
        if not message["content"]:
            continue
        if message["role"] == "assistant":
            result.append(AIMessage(content=message["content"]))
        elif message["role"] == "user":
            result.append(HumanMessage(content=message["content"]))
    return result


async def inference(
    message: str,
    chatbot: list,
    system_message: str,
    temperature: float,
):
    output: str = ""
    reasoning_content: str = ""
    tool_call: str = ""

    history = filter_messages(chatbot)

    current_message = None
    result_messages = []
    tool_messages: dict[str, gr.ChatMessage] = {}
    tool_name: dict[str, str] = {}
    current_type = None
    tool_id = None

    last_yield_time = time.monotonic()
    force_yield = False

    async for stream_mode, chunk in base_agent.astream(
        input={
            "system_message": system_message,
            "messages": history + [HumanMessage(content=message)],
            "temperature": temperature,
        },
        stream_mode=["values", "messages"],
    ):
        if stream_mode == "messages":
            msg, metadata = chunk
            if metadata["langgraph_node"] == "tool_node":
                if current_message:
                    result_messages.append(current_message)
                tool_messages[msg.tool_call_id].metadata = {
                    "title": f"🛠️ Used {tool_name[msg.tool_call_id]} tool",
                    "status": "done",
                    "id": msg.tool_call_id
                }
                output = ""
                tool_call = ""
                current_message = gr.ChatMessage(
                    role="assistant",
                    content=msg.content,
                    metadata={"title": "🛠️ result", "parent_id": msg.tool_call_id}
                )
                force_yield = True
            elif metadata["langgraph_node"] != "agent":
                continue
            if msg.type == "AIMessageChunk" and msg.content:
                if not current_type:
                    current_type = "content"
                elif current_type != "content":
                    if current_type == "reasoning":
                        current_message.metadata["status"] = "done"
                    result_messages.append(current_message)
                    reasoning_content = ""
                    current_type = "content"
                    force_yield = True
                output += msg.content
                current_message = gr.ChatMessage(
                    role="assistant",
                    content=output
                )

            if msg.type == "AIMessageChunk" and msg.additional_kwargs.get("reasoning_content"):
                if not current_type:
                    current_type = "reasoning"
                elif current_type != "reasoning":
                    result_messages.append(current_message)
                    output = ""
                    tool_call = ""
                    current_type = "reasoning"
                    force_yield = True
                reasoning_content += msg.additional_kwargs["reasoning_content"]
                current_message = gr.ChatMessage(
                    role="assistant",
                    content=reasoning_content,
                    metadata={"title": "🧠 thinking...", "status": "pending"},
                )

            if msg.type == "AIMessageChunk" and (msg.tool_calls or msg.tool_call_chunks):
                if not current_type:
                    current_type = "tool_call"
                elif current_type != "tool_call":
                    reasoning_content = ""
                    output = ""
                    result_messages.append(current_message)
                    current_message = None
                    current_type = "tool_call"
                    force_yield = True
                for c in msg.tool_call_chunks:
                    if c["id"]:
                        tool_id = c["id"]
                        tool_messages[tool_id] = gr.ChatMessage(
                            role="assistant",
                            content=""
                        )
                        if current_message:
                            result_messages.append(current_message)
                            current_message = None
                            output = ""
                    if c["name"]:
                        tool_call = c["name"]
                    if c["args"]:
                        output += c["args"]
                tool_messages[tool_id].content = output
                tool_messages[tool_id].metadata = {
                    "title": f"🛠️ Using {tool_call} tool",
                    "status": "pending",
                    "id": tool_id
                }
                tool_name[tool_id] = tool_call
                current_message = tool_messages[tool_id]

            now = time.monotonic()
            if force_yield or (now - last_yield_time >= YIELD_INTERVAL):
                if current_message is not None:
                    yield result_messages + [current_message]
                else:
                    yield result_messages
                last_yield_time = now
                force_yield = False

    if current_message is not None:
        if current_type == "reasoning":
            current_message.metadata["status"] = "done"
        yield result_messages + [current_message]
