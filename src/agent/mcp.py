"""Подключение внешних MCP-серверов и загрузка их инструментов."""

from __future__ import annotations

from langchain_mcp_adapters.client import MultiServerMCPClient

# Конфиг MCP-серверов. Каждый поднимается как отдельный stdio-процесс через uvx —
# это значит, что пакет скачается из PyPI при первом запуске и запустится в изоляции.
MCP_SERVERS = {
    # fetch — умеет ходить по произвольным URL и возвращать контент страницы.
    # Примитивно, но универсально: агент может читать любую веб-страницу.
    "fetch": {
        "command": "uvx",
        "args": ["mcp-server-fetch"],
        "transport": "stdio",
    },
    # time — текущее время, таймзоны, конвертация между ними.
    # Нулевые зависимости — хороший пример чистого stateless-сервера.
    "time": {
        "command": "uvx",
        "args": ["mcp-server-time", "--local-timezone=Asia/Tashkent"],
        "transport": "stdio",
    },
}

_client = MultiServerMCPClient(MCP_SERVERS)


async def load_mcp_tools():
    """Поднимает MCP-серверы и возвращает их инструменты как LangChain tools."""
    return await _client.get_tools()
