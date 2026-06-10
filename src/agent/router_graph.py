"""Граф с роутингом: router решает, какому агенту (или чату) отдать запрос."""

from __future__ import annotations

import asyncio
import os
from dataclasses import dataclass, field
from typing import Annotated, Any, Dict, List

from dotenv import load_dotenv
from langchain_core.messages import BaseMessage, SystemMessage
from langchain_openai import ChatOpenAI
from langgraph.graph import END, StateGraph
from langgraph.graph.message import add_messages
from langgraph.prebuilt import ToolNode, tools_condition
from langgraph.runtime import Runtime
from typing_extensions import TypedDict

from agent.mcp import load_mcp_tools
from agent.tools import (
    calculate,
    convert_currency,
    get_crypto_price,
    get_hacker_news_top,
    get_iss_location,
    get_weather,
    search_wikipedia,
)

load_dotenv()

# Забираем MCP-тулзы и раскладываем по именам — удобнее выбирать нужные по ключу.
_mcp_tools = asyncio.run(load_mcp_tools())
_mcp = {t.name: t for t in _mcp_tools}

# Специализация №1 — веб-исследователь: чтение интернета, энциклопедии и новостей.
WEB_TOOLS = [_mcp["fetch"], search_wikipedia, get_iss_location, get_hacker_news_top]

# Специализация №2 — риалтайм-данные: погода, крипта, валюты, время, расчёты.
DATA_TOOLS = [
    get_weather,
    get_crypto_price,
    convert_currency,
    calculate,
    _mcp["get_current_time"],
    _mcp["convert_time"],
]

# Прокси гоняет Claude с принудительно включённым thinking, а в этом режиме
# допустима только temperature=1. Поэтому фиксируем 1.0 на всех LLM и не шлём
# qwen-специфичный extra_body (он тут всё равно игнорируется).
_llm_kwargs = dict(
    base_url=os.getenv("LLM_BASE_URL"),
    api_key=os.getenv("LLM_API_KEY", "not-needed"),
    model=os.getenv("MODEL", "local-model"),
    temperature=1.0,
)

_chat_llm = ChatOpenAI(**_llm_kwargs)
_web_llm = ChatOpenAI(**_llm_kwargs).bind_tools(WEB_TOOLS)
_data_llm = ChatOpenAI(**_llm_kwargs).bind_tools(DATA_TOOLS)


# Роутер не биндит тулзы и не использует structured_output: просим модель вернуть
# одно слово текстом и парсим вручную — так роутинг не зависит от поддержки
# function calling и стабильно работает даже при temperature=1 (промпт строгий).
_router_llm = ChatOpenAI(**_llm_kwargs)

_ROUTER_SYSTEM = (
    "Ты — роутер. Посмотри на последнее сообщение пользователя и выбери один маршрут "
    "из трёх. Ответь СТРОГО одним словом без кавычек и пояснений:\n"
    "- chat — обычная беседа, шутка, мнение, без внешних данных.\n"
    "- web — нужен поиск или чтение в интернете: произвольные URL, Википедия, "
    "факты о МКС, новости Hacker News.\n"
    "- data — нужны актуальные данные или расчёт: погода, курсы криптовалют, "
    "конвертация валют, арифметика, текущее время, таймзоны."
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
    """Ветка обычной беседы — отвечает без инструментов."""
    system = SystemMessage(content="Ты дружелюбный ассистент. Отвечай кратко и по делу.")
    resp = await _chat_llm.ainvoke([system] + state.messages)
    return {"messages": [resp]}


async def web_agent(state: State, runtime: Runtime[Context]) -> Dict[str, Any]:
    """Ветка веб-исследователя — работает с чтением интернета и новостями."""
    system = SystemMessage(
        content=(
            "Ты веб-исследователь. У тебя есть инструменты: fetch (скачать URL), "
            "search_wikipedia, get_iss_location, get_hacker_news_top (топ новостей "
            "Hacker News). Используй их, чтобы собрать факты, потом дай итоговый ответ."
        )
    )
    resp = await _web_llm.ainvoke([system] + state.messages)
    return {"messages": [resp]}


async def data_agent(state: State, runtime: Runtime[Context]) -> Dict[str, Any]:
    """Ветка реалтайм-данных — погода, валюты, расчёты, время."""
    system = SystemMessage(
        content=(
            "Ты специалист по актуальным данным. У тебя есть инструменты для погоды, "
            "цен криптовалют, конвертации валют, точных вычислений (calculate), "
            "текущего времени и конвертации таймзон. Используй их, "
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
