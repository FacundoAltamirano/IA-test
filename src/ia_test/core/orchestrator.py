import logging

from .event_bus import EventBus
from .events import SystemStarted, SystemStopped
from .settings import Settings
from ..providers.base import ModelProvider

logger = logging.getLogger(__name__)


class Core:
    """Central orchestrator. Owns the event bus and wires all components."""

    def __init__(self, settings: Settings, provider: ModelProvider) -> None:
        self.settings = settings
        self.provider = provider
        self.event_bus = EventBus()
        self._running = False

    @classmethod
    def from_settings(cls, settings: Settings | None = None) -> "Core":
        if settings is None:
            settings = Settings()
        provider = _build_provider(settings)
        return cls(settings=settings, provider=provider)

    async def start(self) -> None:
        self._running = True
        logger.info(
            "Core started — provider: %s, model: %s",
            self.settings.model_provider,
            self.settings.model_name,
        )
        await self.event_bus.publish(SystemStarted())

    async def stop(self) -> None:
        self._running = False
        await self.event_bus.publish(SystemStopped())
        logger.info("Core stopped")

    @property
    def running(self) -> bool:
        return self._running


def _build_provider(settings: Settings) -> ModelProvider:
    if settings.model_provider == "openai":
        from ..providers.openai import OpenAIProvider
        return OpenAIProvider(
            api_key=settings.openai_api_key,
            model=settings.model_name,
        )
    raise ValueError(f"Unknown model provider: {settings.model_provider!r}")
