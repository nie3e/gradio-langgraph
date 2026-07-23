import os
from dataclasses import dataclass

import requests

from typing import TypedDict, Annotated

from langchain_core.messages import SystemMessage
from langchain_qwq import ChatQwen
from langgraph.constants import END
from langgraph.graph import StateGraph
from langgraph.graph.message import add_messages
from langgraph.graph.state import CompiledStateGraph
from langgraph.runtime import Runtime

from langgraph.typing import ContextT, InputT, OutputT, StateT
from langgraph.types import (
    All,
    CachePolicy,
    Checkpointer,
    Command,
    Durability,
    GraphOutput,
    Interrupt,
    Send,
    StateSnapshot,
    StateUpdate,
    StreamMode,
    StreamPart,
    TimeoutPolicy,
    ensure_valid_checkpointer,
)
from langchain_core.runnables.config import RunnableConfig
from collections.abc import (
    AsyncIterator,
    Awaitable,
    Callable,
    Iterator,
    Mapping,
    Sequence,
)

BASE_URL = os.getenv("BASE_URL", "http://localhost:8000/v1")


class AgentState(TypedDict):
    system_message: str
    messages: Annotated[list, add_messages]

    temperature: float


@dataclass
class Context:
    llm: ChatQwen


async def agent(state: AgentState, runtime: Runtime[Context]) -> dict:
    response = await runtime.context.llm.ainvoke(
        [SystemMessage(content=state["system_message"])] + state["messages"],
        temperature=state["temperature"]
    )

    return {"messages": [response]}


def build_graph():
    builder = StateGraph(AgentState)
    builder.add_node("agent", agent)
    builder.set_entry_point("agent")
    builder.add_edge("agent", END)
    graph = builder.compile()
    return graph


class BaseAgent:
    def __init__(self) -> None:
        self._llm: ChatQwen | None = None
        self._base_url: str
        self._api_key: str
        self._model_name: str | None = None
        self._agent: CompiledStateGraph | None = None

    def set_connection(self, base_url: str, api_key: str = "k") -> str | None:
        try:
            resp = requests.get(
                f"{base_url.rstrip("/")}/models",
                headers={"Authorization": f"Bearer {api_key}"},
                timeout=1.0
            )
            resp.raise_for_status()

            self._model_name = resp.json()["data"][0]["id"]
            if not self._model_name:
                raise RuntimeError("No model")
            self._llm = ChatQwen(
                base_url=base_url, api_key=api_key, model=self._model_name
            )
            self._agent = build_graph()
        except Exception:
            self._llm = None
            return "No connection"
        return self._model_name

    @property
    def model_name(self) -> str | None:
        return self._model_name

    async def ainvoke(
        self, state: AgentState, *, context: Context | None = None, **kwargs
    ):
        if not self._llm:
            raise RuntimeError("LLM not initialized")
        if not self._agent:
            raise RuntimeError("Agent not initialized")
        return await self._agent.ainvoke(
            state,
            context=context or Context(llm=self._llm),
            **kwargs
        )

    def astream(
        self,
        input: InputT | Command | None,
        config: RunnableConfig | None = None,
        *,
        context: ContextT | None = None,
        **kwargs
    ) -> AsyncIterator:
        if not self._llm:
            raise RuntimeError("LLM not initialized")
        if not self._agent:
            raise RuntimeError("Agent not initialized")
        return self._agent.astream(
            input,
            config,
            context=context or Context(llm=self._llm),
            **kwargs
        )


base_agent = BaseAgent()
