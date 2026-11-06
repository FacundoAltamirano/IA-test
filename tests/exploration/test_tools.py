import pytest
from ia_test.exploration.tools import WebSearchTool, ExplorationTool


async def test_web_search_without_api_key_returns_stub():
    tool = WebSearchTool(api_key=None)
    result = await tool.execute("test query")
    assert "web_search not configured" in result
    assert "test query" in result


async def test_web_search_logs_the_query():
    tool = WebSearchTool()
    result = await tool.execute("quantum entanglement")
    assert "quantum entanglement" in result


def test_web_search_implements_tool_protocol():
    tool = WebSearchTool()
    assert isinstance(tool, ExplorationTool)
    assert tool.name == "web_search"
    assert tool.description
