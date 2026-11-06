import pytest
from unittest.mock import AsyncMock, MagicMock, patch
from ia_test.providers.openai import OpenAIProvider
from ia_test.providers.base import Message, ModelProvider


@pytest.fixture
def provider():
    return OpenAIProvider(api_key="test-key", model="gpt-4o-mini")


def test_provider_model_name(provider):
    assert provider.model_name == "gpt-4o-mini"


def test_provider_implements_protocol(provider):
    assert isinstance(provider, ModelProvider)


async def test_complete_returns_response(provider):
    mock_response = MagicMock()
    mock_response.choices[0].message.content = "Hello from AI"
    mock_response.model = "gpt-4o-mini"
    mock_response.usage.prompt_tokens = 10
    mock_response.usage.completion_tokens = 5
    mock_response.usage.total_tokens = 15

    with patch.object(
        provider._client.chat.completions,
        "create",
        new=AsyncMock(return_value=mock_response),
    ):
        response = await provider.complete([Message(role="user", content="Hi")])

    assert response.content == "Hello from AI"
    assert response.model == "gpt-4o-mini"
    assert response.usage.prompt_tokens == 10
    assert response.usage.completion_tokens == 5
    assert response.usage.total_tokens == 15


async def test_complete_passes_messages_correctly(provider):
    mock_response = MagicMock()
    mock_response.choices[0].message.content = "ok"
    mock_response.model = "gpt-4o-mini"
    mock_response.usage.prompt_tokens = 5
    mock_response.usage.completion_tokens = 2
    mock_response.usage.total_tokens = 7

    create_mock = AsyncMock(return_value=mock_response)
    with patch.object(provider._client.chat.completions, "create", new=create_mock):
        messages = [
            Message(role="system", content="You are a test assistant."),
            Message(role="user", content="Hello"),
        ]
        await provider.complete(messages)

    call_kwargs = create_mock.call_args.kwargs
    assert call_kwargs["model"] == "gpt-4o-mini"
    assert len(call_kwargs["messages"]) == 2
    assert call_kwargs["messages"][0]["role"] == "system"
    assert call_kwargs["messages"][1]["role"] == "user"
