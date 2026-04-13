"""Простой чат-граф на базе локальной LLM через OpenAI-совместимый протокол."""

from __future__ import annotations

import asyncio
import os
from dataclasses import dataclass, field
from typing import Annotated, Any, Dict, List

from langchain_core.messages import BaseMessage, SystemMessage
from langchain_openai import ChatOpenAI
from langgraph.graph import StateGraph
from langgraph.graph.message import add_messages
from langgraph.prebuilt import ToolNode, tools_condition
from langgraph.runtime import Runtime
from typing_extensions import TypedDict
from dotenv import load_dotenv

from agent.mcp import load_mcp_tools
from agent.tools import TOOLS

load_dotenv()

# Поднимаем MCP-серверы и забираем их тулзы синхронно при импорте модуля.
# LangGraph-рантайм ожидает уже скомпилированный объект `graph`, поэтому не можем
# делать это лениво.
_mcp_tools = asyncio.run(load_mcp_tools())
ALL_TOOLS = TOOLS + _mcp_tools

llm = ChatOpenAI(
    base_url=os.getenv("LLM_BASE_URL"),
    api_key=os.getenv("LLM_KEY", "not-needed"),
    model=os.getenv("LLM_NAME", "local-model"),
    temperature=0.7,
    extra_body={"chat_template_kwargs": {"enable_thinking": False}},
).bind_tools(ALL_TOOLS)


class Context(TypedDict):
    """Параметры конфигурации графа."""
    system_prompt: str


@dataclass
class State:
    """Состояние графа — здесь живёт история диалога."""
    messages: Annotated[List[BaseMessage], add_messages] = field(default_factory=list)


async def call_model(state: State, runtime: Runtime[Context]) -> Dict[str, Any]:
    """Основной узел — передаём историю в модель и получаем ответ."""
    system_prompt = (runtime.context or {}).get(
        "system_prompt",
        "Ты полезный ИИ-ассистент. Отвечай чётко и по делу."
    )
    full_messages = [SystemMessage(content=system_prompt)] + state.messages
    response = await llm.ainvoke(full_messages)
    return {"messages": [response]}


graph = (
    StateGraph(State, context_schema=Context)
    .add_node(call_model)
    .add_node("tools", ToolNode(ALL_TOOLS))
    .add_edge("__start__", "call_model")
    .add_conditional_edges("call_model", tools_condition)
    .add_edge("tools", "call_model")
    .compile(name="Chat Graph")
)
