import pytest
from ia_test.core.settings import Settings


@pytest.fixture
def test_settings() -> Settings:
    return Settings(
        model_provider="openai",
        model_name="gpt-4o-mini",
        openai_api_key="test-key",
        max_autonomous_cycles=5,
        max_api_calls_per_hour=10,
        max_api_calls_per_day=50,
    )
