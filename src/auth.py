"""Bearer-авторизация для сервера LangGraph.

Хендлер читает секрет из AUTH_SECRET_TOKEN и проверяет заголовок Authorization:
- нет заголовка или не формата "Bearer <token>" -> 401 (+ WWW-Authenticate);
- токен есть, но не совпал с секретом -> 403;
- токен верный -> запрос пропускается, идентити = "user".

Про статус-коды. LangGraph-сервер нормализует `Auth.exceptions.HTTPException`
в HTTP 403 на любой отказ (его detail сохраняется, но код и заголовки теряются).
Чтобы корректно различать «не аутентифицирован» (401) и «нет доступа» (403),
поднимаем `starlette.exceptions.HTTPException` — кастомный auth-бэкенд LangGraph
пробрасывает его как есть, поэтому статус и заголовки доходят до клиента.
"""

import os

from dotenv import load_dotenv
from langgraph_sdk import Auth
from starlette.exceptions import HTTPException

load_dotenv()

auth = Auth()

SECRET_TOKEN = os.getenv("AUTH_SECRET_TOKEN", "")


@auth.authenticate
async def authenticate(authorization: str | None) -> str:
    """Проверяет Bearer-токен и возвращает идентити пользователя."""
    if not authorization or not authorization.startswith("Bearer "):
        raise HTTPException(
            status_code=401,
            detail="Missing or malformed Authorization header",
            headers={"WWW-Authenticate": "Bearer"},
        )

    token = authorization.removeprefix("Bearer ").strip()

    if not SECRET_TOKEN or token != SECRET_TOKEN:
        raise HTTPException(status_code=403, detail="Invalid token")

    return "user"
