import pytest
from unittest.mock import MagicMock
from ia_test.core.orchestrator import Core
from ia_test.core.events import SystemStarted, SystemStopped


@pytest.fixture
def mock_provider():
    provider = MagicMock()
    provider.model_name = "gpt-4o-mini"
    return provider


async def test_core_starts_and_emits_system_started(test_settings, mock_provider):
    core = Core(settings=test_settings, provider=mock_provider)
    received = []

    async def capture(event):
        received.append(event)

    core.event_bus.subscribe(SystemStarted, capture)
    await core.start()

    assert core.running
    assert len(received) == 1
    assert isinstance(received[0], SystemStarted)


async def test_core_stop_emits_system_stopped(test_settings, mock_provider):
    core = Core(settings=test_settings, provider=mock_provider)
    received = []

    async def capture(event):
        received.append(event)

    core.event_bus.subscribe(SystemStopped, capture)
    await core.start()
    await core.stop()

    assert not core.running
    assert len(received) == 1
    assert isinstance(received[0], SystemStopped)


async def test_core_exposes_provider(test_settings, mock_provider):
    core = Core(settings=test_settings, provider=mock_provider)
    assert core.provider is mock_provider
    assert core.provider.model_name == "gpt-4o-mini"
