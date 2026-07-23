import os
from dataclasses import dataclass

import requests

from typing import TypedDict, Annotated, Literal, Any

from langchain_core.messages import SystemMessage, ToolMessage
from langchain_openai import OpenAIEmbeddings
from langchain_qwq import ChatQwen
from langgraph.constants import END
from langgraph.graph import StateGraph
from langgraph.graph.message import add_messages
from langgraph.graph.state import CompiledStateGraph
from langgraph.runtime import Runtime

from langgraph.typing import ContextT, InputT
from langgraph.types import Command
from langchain_core.runnables.config import RunnableConfig
from collections.abc import AsyncIterator
from langchain_core.tools import tool, BaseTool, InjectedToolArg

import weaviate
from weaviate.classes.query import MetadataQuery


WEAVIATE_HOST = os.getenv("WEAVIATE_HOST", "localhost")
WEAVIATE_PORT = int(os.getenv("WEAVIATE_PORT", "8088"))
WEAVIATE_COLLECTION = os.getenv("WEAVIATE_COLLECTION", "my_collection")


EMBEDDING_BASE_URL = os.getenv("EMBEDDING_BASE_URL", "http://localhost:8001/v1")
EMBEDDING_MODEL_NAME = os.getenv("EMBEDDING_MODEL_NAME", "emb_model")


def embed(query: str, client: OpenAIEmbeddings) -> list[float]:
    QUERY_PREFIX = (
        "Instruct: Given a web search query, retrieve relevant "
        "passages that answer the query\nQuery: "
    )

    embeddings = client.embed_query(QUERY_PREFIX + query)
    return embeddings


class AgentState(TypedDict):
    system_message: str
    messages: Annotated[list, add_messages]

    temperature: float


@dataclass
class Context:
    llm: Any
    embeddings: OpenAIEmbeddings


def get_results(embeddings: list[float], top_n: int = 10) -> list[dict]:
    client = weaviate.connect_to_local(host=WEAVIATE_HOST, port=WEAVIATE_PORT)
    assert client.is_ready()

    collection = client.collections.use(WEAVIATE_COLLECTION)

    vector_results = collection.query.near_vector(
        near_vector=embeddings,
        limit=top_n,
        target_vector=["text_vec"],
        return_metadata=MetadataQuery(distance=True)
    )
    client.close()

    results = []
    for obj in vector_results.objects:
        results.append({
            "publication_id": obj.properties["publication_id"],
            "chunk_id": obj.properties["chunk_id"],
            "text": obj.properties["text"],
            "uuid": str(obj.uuid),
            "distance": obj.metadata.distance
        })
    return results


def results_to_string(rag_results: list[dict]) -> str:
    msg = "\n\n".join([
        f"uuid: {result["uuid"]}\n"
        f"distance: {result["distance"]}\n"
        f"text: {result["text"]}"
        for result in rag_results
    ])
    return msg


@tool
def search_in_database(query: str, embeddings_client: Annotated[OpenAIEmbeddings, InjectedToolArg]) -> str:
    """Wyszukuje w bazie danych"""
    query_embedding = embed(query, embeddings_client)
    rag_results = get_results(query_embedding, 10)
    return results_to_string(rag_results)


tools: list[BaseTool] = [search_in_database]
tools_by_name = {tool.name: tool for tool in tools}


async def agent(state: AgentState, runtime: Runtime[Context]) -> dict:
    system_prompt = """Jesteś przyjaznym agentem i masz dostęp do wektorowej bazy danych Biblioteka Nauki.
Użytkownik zada Ci pytanie - wyszukaj odpowiedzi w tej bazie - możesz przesłać tyle zapytań ile chcesz.
W odpowiedzi dostaniesz chunki tekstu, które najbardziej pasowały, razem z miarą podobieństwa:
0.0 - najbardziej zbliżony wynik
1.0 - najmniej zbliżony wynik
Poniżej 0.3: dokładny wynik
Między 0.3 a 0.5: wynik przeciętny
Większy niż 0.5: wynik niepowiązany
Z wyszukiwania możesz korzystać ile chcesz - jeśli pierwsze materiały nie dostarczą odpowiedzi, możesz dalej korygować swoje zapytania.
Postaraj się odpowiedzieć na pytanie użytkownika używając zwróconych danych.
"""
    state["system_message"] = system_prompt

    response = await runtime.context.llm.ainvoke(
        [SystemMessage(content=state["system_message"])] + state["messages"],
        temperature=state["temperature"]
    )

    return {"messages": [response]}


def tool_node(state: AgentState, runtime: Runtime[Context]):
    result = []
    for tool_call in state["messages"][-1].tool_calls:
        tool = tools_by_name[tool_call["name"]]
        observation = tool.invoke(tool_call["args"] | {"embeddings_client": runtime.context.embeddings})
        result.append(ToolMessage(
            content=observation, tool_call_id=tool_call["id"]
        ))

    return {"messages": result}


def should_continue(state: AgentState) -> Literal["tool_node", "__end__"]:
    messages = state["messages"]
    last_message = messages[-1]

    if last_message.tool_calls:
        return "tool_node"

    return END


def build_graph():
    builder = StateGraph(AgentState)

    builder.add_node("agent", agent)
    builder.add_node("tool_node", tool_node)

    builder.set_entry_point("agent")
    builder.add_conditional_edges(
        "agent",
        should_continue,
        ["tool_node", END]
    )
    builder.add_edge("tool_node", "agent")
    graph = builder.compile()
    return graph


class RAGAgent:
    def __init__(self) -> None:
        self._llm: Any | None = None
        self._embeddings: OpenAIEmbeddings | None = None
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

            resp = requests.get(
                f"{EMBEDDING_BASE_URL}/models",
                headers={"Authorization": f"Bearer {api_key}"},
                timeout=1.0
            )
            resp.raise_for_status()

            self._embeddings = OpenAIEmbeddings(
                base_url=EMBEDDING_BASE_URL,
                api_key="-",
                model=EMBEDDING_MODEL_NAME,
                check_embedding_ctx_length=False,
                tiktoken_enabled=False,
            )
            self._llm = ChatQwen(
                base_url=base_url, api_key=api_key, model=self._model_name
            ).bind_tools(tools)
            client = weaviate.connect_to_local(host=WEAVIATE_HOST,
                                               port=WEAVIATE_PORT)
            assert client.is_ready()
            client.close()
            self._agent = build_graph()
        except Exception:
            self._llm = None
            self._embeddings = None
            return "No connection"
        return self._model_name

    @property
    def model_name(self) -> str | None:
        return self._model_name

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
        if not self._embeddings:
            raise RuntimeError("Embeddings not initialized")
        return self._agent.astream(
            input,
            config,
            context=context or Context(llm=self._llm,
                                       embeddings=self._embeddings),
            **kwargs
        )


base_agent = RAGAgent()
