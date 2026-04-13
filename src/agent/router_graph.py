"""Граф с роутингом: router решает, какому агенту (или чату) отдать запрос."""

from __future__ import annotations

import asyncio
import os
from dataclasses import dataclass, field
from typing import Annotated, Any, Dict, List, Literal

from dotenv import load_dotenv
from langchain_core.messages import BaseMessage, SystemMessage
from langchain_openai import ChatOpenAI
from langgraph.graph import END, StateGraph
from langgraph.graph.message import add_messages
from langgraph.prebuilt import ToolNode, tools_condition
from langgraph.runtime import Runtime
from typing_extensions import TypedDict

from agent.mcp import load_mcp_tools
from agent.tools import get_crypto_price, get_iss_location, get_weather, search_wikipedia

load_dotenv()

# Забираем MCP-тулзы и раскладываем по именам — удобнее выбирать нужные по ключу.
_mcp_tools = asyncio.run(load_mcp_tools())
_mcp = {t.name: t for t in _mcp_tools}

# Специализация №1 — веб-исследователь: чтение интернета и энциклопедии.
WEB_TOOLS = [_mcp["fetch"], search_wikipedia, get_iss_location]

# Специализация №2 — риалтайм-данные: погода, крипта, время/таймзоны.
DATA_TOOLS = [
    get_weather,
    get_crypto_price,
    _mcp["get_current_time"],
    _mcp["convert_time"],
]

_llm_kwargs = dict(
    base_url=os.getenv("LLM_BASE_URL"),
    api_key=os.getenv("LLM_KEY", "not-needed"),
    model=os.getenv("LLM_NAME", "local-model"),
    extra_body={"chat_template_kwargs": {"enable_thinking": False}},
)

_chat_llm = ChatOpenAI(temperature=0.7, **_llm_kwargs)
_web_llm = ChatOpenAI(temperature=0.3, **_llm_kwargs).bind_tools(WEB_TOOLS)
_data_llm = ChatOpenAI(temperature=0.3, **_llm_kwargs).bind_tools(DATA_TOOLS)


# Роутер — отдельный LLM с низкой температурой. Без bind_tools/structured_output:
# qwen-шаблон ломается на function calling без явного user-запроса в нужной форме,
# поэтому просим модель вернуть одно слово текстом и парсим вручную.
_router_llm = ChatOpenAI(temperature=0.0, **_llm_kwargs)

_ROUTER_SYSTEM = (
    "Ты — роутер. Посмотри на последнее сообщение пользователя и выбери один маршрут "
    "из трёх. Ответь СТРОГО одним словом без кавычек и пояснений:\n"
    "- chat — обычная беседа, шутка, мнение, без внешних данных.\n"
    "- web — нужен поиск или чтение в интернете: произвольные URL, Википедия, факты о МКС.\n"
    "- data — нужны актуальные данные: погода, курсы криптовалют, текущее время, таймзоны."
)

_VALID_ROUTES = {"chat", "web", "data"}


class Context(TypedDict):
    """Параметры конфигурации графа."""
    system_prompt: str


@dataclass
class State:
    """Состояние графа: история сообщений + выбранный роут."""
    messages: Annotated[List[BaseMessage], add_messages] = field(default_factory=list)
    route: str = "chat"


async def router(state: State, runtime: Runtime[Context]) -> Dict[str, Any]:
    """Классифицирует последний запрос пользователя в один из трёх маршрутов."""
    system = SystemMessage(content=_ROUTER_SYSTEM)
    resp = await _router_llm.ainvoke([system] + state.messages)
    # Достаём первое слово из ответа и сверяем со списком допустимых маршрутов.
    word = (resp.content or "").strip().lower().split()[:1]
    destination = word[0] if word and word[0] in _VALID_ROUTES else "chat"
    return {"route": destination}


def pick_route(state: State) -> str:
    """Вытаскивает роут из стейта для conditional_edges."""
    return state.route


async def chat_node(state: State, runtime: Runtime[Context]) -> Dict[str, Any]:
    system = SystemMessage(content="Ты дружелюбный ассистент. Отвечай кратко и по делу.")
    resp = await _chat_llm.ainvoke([system] + state.messages)
    return {"messages": [resp]}


async def web_agent(state: State, runtime: Runtime[Context]) -> Dict[str, Any]:
    system = SystemMessage(
        content=(
            "Ты веб-исследователь. У тебя есть инструменты: fetch (скачать URL), "
            "search_wikipedia, get_iss_location. Используй их, чтобы собрать факты, "
            "потом дай итоговый ответ."
        )
    )
    resp = await _web_llm.ainvoke([system] + state.messages)
    return {"messages": [resp]}


async def data_agent(state: State, runtime: Runtime[Context]) -> Dict[str, Any]:
    system = SystemMessage(
        content=(
            "Ты специалист по актуальным данным. У тебя есть инструменты для погоды, "
            "цен криптовалют, текущего времени и конвертации таймзон. Используй их, "
            "а потом коротко ответь пользователю."
        )
    )
    resp = await _data_llm.ainvoke([system] + state.messages)
    return {"messages": [resp]}


graph = (
    StateGraph(State, context_schema=Context)
    .add_node("router", router)
    .add_node("chat", chat_node)
    .add_node("web_agent", web_agent)
    .add_node("web_tools", ToolNode(WEB_TOOLS))
    .add_node("data_agent", data_agent)
    .add_node("data_tools", ToolNode(DATA_TOOLS))
    .add_edge("__start__", "router")
    .add_conditional_edges(
        "router",
        pick_route,
        {"chat": "chat", "web": "web_agent", "data": "data_agent"},
    )
    .add_edge("chat", END)
    # Цикл реактивного агента для web_agent
    .add_conditional_edges("web_agent", tools_condition, {"tools": "web_tools", END: END})
    .add_edge("web_tools", "web_agent")
    # Цикл реактивного агента для data_agent
    .add_conditional_edges("data_agent", tools_condition, {"tools": "data_tools", END: END})
    .add_edge("data_tools", "data_agent")
    .compile(name="Router Graph")
)
