"""Третий агент — узкопрофильный эксперт: кулинар-нутрициолог.

Простой ReAct-граф (как `agent`), но с фокусным набором кулинарных тулзов и
сильным доменным system-промптом. MCP здесь не подключаем — эксперту хватает
своих инструментов из `chef_tools`.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from typing import Annotated, Any, Dict, List

from dotenv import load_dotenv
from langchain_core.messages import BaseMessage, SystemMessage
from langchain_openai import ChatOpenAI
from langgraph.graph import StateGraph
from langgraph.graph.message import add_messages
from langgraph.prebuilt import ToolNode, tools_condition
from langgraph.runtime import Runtime
from typing_extensions import TypedDict

from agent.chef_tools import CHEF_TOOLS

load_dotenv()

# Прокси держит Claude с принудительным thinking — допустима только temperature=1.
llm = ChatOpenAI(
    base_url=os.getenv("LLM_BASE_URL"),
    api_key=os.getenv("LLM_API_KEY", "not-needed"),
    model=os.getenv("MODEL", "local-model"),
    temperature=1.0,
).bind_tools(CHEF_TOOLS)

# Доменный промпт делает из обычной LLM именно эксперта: задаёт роль, правила
# работы с инструментами и язык общения.
_EXPERT_PROMPT = (
    "Ты — шеф-повар и нутрициолог. Твоя зона — еда: состав продуктов, БЖУ и "
    "калорийность, подбор и разбор рецептов. Отвечай по-русски, тепло и по делу.\n\n"
    "Инструменты:\n"
    "- product_nutrition — состав и пищевая ценность продукта (на 100 г);\n"
    "- find_recipes_by_ingredient — какие блюда готовят из ингредиента;\n"
    "- get_recipe — полный рецепт блюда по названию;\n"
    "- random_meal — случайное блюдо с рецептом.\n\n"
    "Важно: инструменты надёжнее работают с английскими названиями (для рецептов "
    "и ингредиентов TheMealDB английский обязателен, для product_nutrition — "
    "желателен). Переводи запрос на английский, а результат — обратно на русский. "
    "Если инструмент ничего не нашёл, не повторяй тот же вызов несколько раз — "
    "попробуй другое название один раз или ответь из своих знаний, честно пометив "
    "это. Если вопрос не про еду и питание — мягко скажи, что это вне твоей "
    "специализации."
)


class Context(TypedDict):
    """Параметры конфигурации графа."""

    system_prompt: str


@dataclass
class State:
    """Состояние графа — история диалога с экспертом."""

    messages: Annotated[List[BaseMessage], add_messages] = field(default_factory=list)


async def call_model(state: State, runtime: Runtime[Context]) -> Dict[str, Any]:
    """Основной узел — прокидывает историю эксперту и возвращает ответ."""
    system_prompt = (runtime.context or {}).get("system_prompt", _EXPERT_PROMPT)
    full_messages = [SystemMessage(content=system_prompt)] + state.messages
    response = await llm.ainvoke(full_messages)
    return {"messages": [response]}


graph = (
    StateGraph(State, context_schema=Context)
    .add_node(call_model)
    .add_node("tools", ToolNode(CHEF_TOOLS))
    .add_edge("__start__", "call_model")
    .add_conditional_edges("call_model", tools_condition)
    .add_edge("tools", "call_model")
    .compile(name="Chef Expert Graph")
)
