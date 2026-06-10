"""Инструменты кулинара-нутрициолога — еда, состав продуктов и рецепты.

Все API публичные и без ключей:
- Open Food Facts — состав и БЖУ продуктов по названию.
- TheMealDB — рецепты по ингредиенту/названию и случайное блюдо.

TheMealDB понимает только английские названия, поэтому агент должен
переводить запрос на английский, а ответ — обратно на русский.
"""

from __future__ import annotations

import asyncio
from typing import Any, Dict, List

import httpx
from langchain_core.tools import tool

_OFF_URL = "https://world.openfoodfacts.org/api/v2/search"
_MEALDB = "https://www.themealdb.com/api/json/v1/1"
_HEADERS = {"User-Agent": "habr-graph-chef/1.0"}


async def _get_json(client: httpx.AsyncClient, url: str, **params: Any) -> Dict[str, Any]:
    """GET с ретраями на rate-limit и мягкой обработкой сбоев.

    Open Food Facts любит отдавать 503/429 под нагрузкой — пара повторов с
    backoff обычно решает. При не-JSON/окончательной ошибке возвращаем {}.
    """
    for attempt in range(3):
        try:
            resp = await client.get(url, params=params or None)
            if resp.status_code in (429, 503) and attempt < 2:
                await asyncio.sleep(0.6 * (attempt + 1))
                continue
            if resp.status_code != 200:
                return {}
            return resp.json()
        except (httpx.HTTPError, ValueError):
            if attempt < 2:
                await asyncio.sleep(0.6 * (attempt + 1))
                continue
            return {}
    return {}


def _format_meal(meal: Dict[str, Any]) -> str:
    """Собирает карточку блюда: заголовок, ингредиенты, шаги и фото."""
    name = meal.get("strMeal", "Без названия")
    category = meal.get("strCategory") or "—"
    area = meal.get("strArea") or meal.get("strCountry") or "—"

    # У TheMealDB ингредиенты разложены по 20 параллельным полям
    # strIngredient1..20 + strMeasure1..20 — собираем непустые пары.
    ingredients: List[str] = []
    for i in range(1, 21):
        ing = (meal.get(f"strIngredient{i}") or "").strip()
        measure = (meal.get(f"strMeasure{i}") or "").strip()
        if ing:
            ingredients.append(f"- {ing}: {measure}" if measure else f"- {ing}")

    parts = [f"**{name}** ({category}, кухня: {area})"]
    if ingredients:
        parts.append("Ингредиенты:\n" + "\n".join(ingredients))
    instructions = (meal.get("strInstructions") or "").strip()
    if instructions:
        parts.append("Приготовление:\n" + instructions)
    thumb = (meal.get("strMealThumb") or "").strip()
    if thumb:
        parts.append(f"Фото: {thumb}")
    return "\n\n".join(parts)


@tool
async def product_nutrition(name: str) -> str:
    """Узнать состав и пищевую ценность продукта (БЖУ, калории, NutriScore).

    Данные приведены на 100 г продукта.

    Args:
        name: Название продукта, например "nutella", "greek yogurt", "ржаной хлеб".
    """
    async with httpx.AsyncClient(timeout=10, headers=_HEADERS) as client:
        data = await _get_json(
            client,
            _OFF_URL,
            search_terms=name,
            page_size=1,
            fields="product_name,brands,nutriscore_grade,nutriments",
        )

    products = data.get("products") or []
    if not products:
        return f"Продукт '{name}' не найден в Open Food Facts."

    p = products[0]
    n = p.get("nutriments", {})
    title = p.get("product_name") or name
    brand = p.get("brands") or "—"
    grade = (p.get("nutriscore_grade") or "?").upper()

    def g(key: str) -> str:
        val = n.get(f"{key}_100g")
        return f"{val}" if val is not None else "н/д"

    return (
        f"**{title}** (бренд: {brand})\n"
        f"На 100 г:\n"
        f"- Калории: {g('energy-kcal')} ккал\n"
        f"- Белки: {g('proteins')} г\n"
        f"- Жиры: {g('fat')} г\n"
        f"- Углеводы: {g('carbohydrates')} г (сахара: {g('sugars')} г)\n"
        f"- Соль: {g('salt')} г\n"
        f"- NutriScore: {grade}"
    )


@tool
async def find_recipes_by_ingredient(ingredient: str) -> str:
    """Найти блюда, которые готовят из заданного ингредиента.

    Args:
        ingredient: Ингредиент НА АНГЛИЙСКОМ, например "chicken", "salmon", "potato".
    """
    async with httpx.AsyncClient(timeout=10, headers=_HEADERS) as client:
        data = await _get_json(client, f"{_MEALDB}/filter.php", i=ingredient)
        meals = data.get("meals")

    if not meals:
        return f"Блюд с ингредиентом '{ingredient}' не найдено (попробуй английское название)."

    lines = [f"- {m['strMeal']}" for m in meals[:10]]
    more = f"\n…и ещё {len(meals) - 10}." if len(meals) > 10 else ""
    return f"Блюда с '{ingredient}' ({len(meals)} шт.):\n" + "\n".join(lines) + more


@tool
async def get_recipe(name: str) -> str:
    """Получить полный рецепт блюда по названию: ингредиенты и шаги приготовления.

    Args:
        name: Название блюда НА АНГЛИЙСКОМ, например "Arrabiata", "Beef Wellington".
    """
    async with httpx.AsyncClient(timeout=10, headers=_HEADERS) as client:
        data = await _get_json(client, f"{_MEALDB}/search.php", s=name)
        meals = data.get("meals")

    if not meals:
        return f"Рецепт '{name}' не найден (укажи английское название блюда)."
    return _format_meal(meals[0])


@tool
async def random_meal() -> str:
    """Предложить случайное блюдо с полным рецептом — на случай «что бы приготовить»."""
    async with httpx.AsyncClient(timeout=10, headers=_HEADERS) as client:
        data = await _get_json(client, f"{_MEALDB}/random.php")
        meals = data.get("meals")

    if not meals:
        return "Не удалось получить случайное блюдо, попробуй ещё раз."
    return _format_meal(meals[0])


CHEF_TOOLS = [product_nutrition, find_recipes_by_ingredient, get_recipe, random_meal]
