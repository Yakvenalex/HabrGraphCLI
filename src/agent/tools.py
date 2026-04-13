"""Набор инструментов для агента — дёргают публичные API без ключей."""

from __future__ import annotations

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


TOOLS = [get_weather, get_crypto_price, search_wikipedia, get_iss_location]
