from typing import Protocol, runtime_checkable


@runtime_checkable
class ExplorationTool(Protocol):
    name: str
    description: str

    async def execute(self, query: str) -> str: ...


class WebSearchTool:
    name = "web_search"
    description = "Search the web for current information about a topic"

    def __init__(self, api_key: str | None = None) -> None:
        self._api_key = api_key

    async def execute(self, query: str) -> str:
        if not self._api_key:
            return f"[web_search not configured — query logged: {query}]"
        # Future: integrate a real search API (Brave, Tavily, SerpAPI, etc.)
        raise NotImplementedError("web_search API integration pending")
