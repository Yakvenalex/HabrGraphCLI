"""Набор инструментов для агента — дёргают публичные API без ключей."""

from __future__ import annotations

import ast
import asyncio
import operator
from typing import Callable, Dict, Type

import httpx
from langchain_core.tools import tool


@tool
async def get_weather(city: str) -> str:
    """Получить текущую погоду в городе.

    Args:
        city: Название города на любом языке (например, "Ташкент", "Tokyo").
    """
    async with httpx.AsyncClient(timeout=10) as client:
        geo = await client.get(
            "https://geocoding-api.open-meteo.com/v1/search",
            params={"name": city, "count": 1, "language": "ru"},
        )
        geo_data = geo.json().get("results") or []
        if not geo_data:
            return f"Город '{city}' не найден."

        place = geo_data[0]
        lat, lon = place["latitude"], place["longitude"]
        name = f"{place['name']}, {place.get('country', '')}".strip(", ")

        weather = await client.get(
            "https://api.open-meteo.com/v1/forecast",
            params={
                "latitude": lat,
                "longitude": lon,
                "current": "temperature_2m,relative_humidity_2m,wind_speed_10m,weather_code",
            },
        )
        current = weather.json()["current"]

    return (
        f"{name}: {current['temperature_2m']}°C, "
        f"влажность {current['relative_humidity_2m']}%, "
        f"ветер {current['wind_speed_10m']} км/ч "
        f"(код погоды {current['weather_code']})."
    )


@tool
async def get_crypto_price(coin_id: str, vs_currency: str = "usd") -> str:
    """Узнать текущую цену криптовалюты через CoinGecko.

    Args:
        coin_id: Идентификатор монеты на CoinGecko (например, "bitcoin", "ethereum", "solana").
        vs_currency: Валюта котировки (по умолчанию "usd", также "eur", "rub").
    """
    async with httpx.AsyncClient(timeout=10) as client:
        resp = await client.get(
            "https://api.coingecko.com/api/v3/simple/price",
            params={
                "ids": coin_id,
                "vs_currencies": vs_currency,
                "include_24hr_change": "true",
            },
        )
        data = resp.json()

    if coin_id not in data:
        return f"Монета '{coin_id}' не найдена. Используй id с CoinGecko (bitcoin, ethereum и т.д.)."

    price = data[coin_id][vs_currency]
    change = data[coin_id].get(f"{vs_currency}_24h_change", 0)
    arrow = "📈" if change >= 0 else "📉"
    return f"{coin_id}: {price} {vs_currency.upper()} {arrow} {change:+.2f}% за 24ч"


@tool
async def search_wikipedia(query: str, lang: str = "ru") -> str:
    """Найти краткую статью в Википедии по запросу.

    Args:
        query: Что ищем (название темы, персоны, события).
        lang: Код языка Википедии (по умолчанию "ru", также "en").
    """
    async with httpx.AsyncClient(timeout=10, follow_redirects=True) as client:
        url = f"https://{lang}.wikipedia.org/api/rest_v1/page/summary/{query}"
        resp = await client.get(url, headers={"User-Agent": "langgraph-agent/1.0"})

        if resp.status_code == 404:
            return f"Статья '{query}' не найдена в Википедии ({lang})."

        data = resp.json()

    title = data.get("title", query)
    extract = data.get("extract", "Описание отсутствует.")
    page_url = data.get("content_urls", {}).get("desktop", {}).get("page", "")
    return f"**{title}**\n\n{extract}\n\n{page_url}".strip()


@tool
async def get_iss_location() -> str:
    """Узнать текущие координаты Международной космической станции (МКС) и кто сейчас на борту."""
    async with httpx.AsyncClient(timeout=10) as client:
        pos_resp, crew_resp = await client.get("http://api.open-notify.org/iss-now.json"), await client.get(
            "http://api.open-notify.org/astros.json"
        )
        pos = pos_resp.json()["iss_position"]
        crew = crew_resp.json()

    iss_crew = [p["name"] for p in crew["people"] if p["craft"] == "ISS"]
    return (
        f"МКС сейчас над координатами: широта {pos['latitude']}, долгота {pos['longitude']}. "
        f"На борту {len(iss_crew)} человек: {', '.join(iss_crew) if iss_crew else 'нет данных'}."
    )


# Разрешённые операции для калькулятора. Всё, чего нет в этих словарях,
# трактуется как небезопасное и приводит к ошибке — так мы не пускаем в eval
# произвольный Python (вызовы функций, доступ к атрибутам и т.п.).
_BIN_OPS: Dict[Type[ast.operator], Callable[[float, float], float]] = {
    ast.Add: operator.add,
    ast.Sub: operator.sub,
    ast.Mult: operator.mul,
    ast.Div: operator.truediv,
    ast.FloorDiv: operator.floordiv,
    ast.Mod: operator.mod,
    ast.Pow: operator.pow,
}
_UNARY_OPS: Dict[Type[ast.unaryop], Callable[[float], float]] = {
    ast.UAdd: operator.pos,
    ast.USub: operator.neg,
}


def _eval_node(node: ast.AST) -> float:
    """Рекурсивно вычисляет узел AST, пропуская только арифметику."""
    if isinstance(node, ast.Constant) and isinstance(node.value, (int, float)):
        return node.value
    if isinstance(node, ast.BinOp) and type(node.op) in _BIN_OPS:
        return _BIN_OPS[type(node.op)](_eval_node(node.left), _eval_node(node.right))
    if isinstance(node, ast.UnaryOp) and type(node.op) in _UNARY_OPS:
        return _UNARY_OPS[type(node.op)](_eval_node(node.operand))
    raise ValueError("Выражение содержит недопустимую операцию.")


@tool
def calculate(expression: str) -> str:
    """Посчитать арифметическое выражение (точная математика, без округлений LLM).

    Поддерживает + - * / // % ** и скобки. Никаких переменных и функций.

    Args:
        expression: Например, "2 + 2 * 10" или "(1500 / 3) ** 0.5".
    """
    try:
        tree = ast.parse(expression, mode="eval")
        result = _eval_node(tree.body)
    except ZeroDivisionError:
        return "Ошибка: деление на ноль."
    except (ValueError, SyntaxError, TypeError) as exc:
        return f"Не смог вычислить '{expression}': {exc}"
    # Приводим целые результаты к int, чтобы не показывать "4.0" вместо "4".
    if isinstance(result, float) and result.is_integer():
        result = int(result)
    return f"{expression} = {result}"


@tool
async def convert_currency(amount: float, from_currency: str, to_currency: str) -> str:
    """Конвертировать сумму из одной фиатной валюты в другую по актуальному курсу.

    Поддерживаются валюты ЕЦБ (USD, EUR, GBP, JPY, CHF, CNY и др.), без RUB.

    Args:
        amount: Сумма в исходной валюте (например, 100).
        from_currency: Код исходной валюты ISO-4217 (например, "USD", "EUR", "GBP").
        to_currency: Код целевой валюты ISO-4217 (например, "EUR", "JPY", "CHF").
    """
    src, dst = from_currency.upper(), to_currency.upper()
    async with httpx.AsyncClient(timeout=10, follow_redirects=True) as client:
        resp = await client.get(
            "https://api.frankfurter.dev/v1/latest",
            params={"amount": amount, "from": src, "to": dst},
        )
        if resp.status_code == 404:
            return f"Валюта не поддерживается (одна из {src}, {dst}). Сервис работает с валютами ЕЦБ, без RUB."
        if resp.status_code != 200:
            return f"Не удалось получить курс {src}→{dst} (HTTP {resp.status_code})."
        data = resp.json()

    rates = data.get("rates") or {}
    if dst not in rates:
        return f"Валютная пара {src}→{dst} не поддерживается."
    converted = rates[dst]
    return f"{amount} {src} = {converted} {dst} (курс на {data.get('date', 'сегодня')})."


@tool
async def get_hacker_news_top(limit: int = 5) -> str:
    """Получить заголовки топовых новостей с Hacker News.

    Args:
        limit: Сколько новостей вернуть (1–10, по умолчанию 5).
    """
    limit = max(1, min(limit, 10))
    base = "https://hacker-news.firebaseio.com/v0"
    async with httpx.AsyncClient(timeout=10) as client:
        ids_resp = await client.get(f"{base}/topstories.json")
        ids = ids_resp.json()[:limit]
        # Тянем карточки новостей параллельно — так быстрее, чем по одной.
        items = await asyncio.gather(
            *(client.get(f"{base}/item/{story_id}.json") for story_id in ids)
        )

    lines = []
    for i, item in enumerate(items, 1):
        story = item.json() or {}
        title = story.get("title", "без заголовка")
        score = story.get("score", 0)
        url = story.get("url", f"https://news.ycombinator.com/item?id={story.get('id', '')}")
        lines.append(f"{i}. {title} ({score} баллов)\n   {url}")
    return "Топ Hacker News:\n" + "\n".join(lines)


TOOLS = [
    get_weather,
    get_crypto_price,
    search_wikipedia,
    get_iss_location,
    calculate,
    convert_currency,
    get_hacker_news_top,
]
