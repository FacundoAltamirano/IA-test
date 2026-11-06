from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", extra="ignore")

    # Model provider
    model_provider: str = "openai"
    model_name: str = "gpt-4o-mini"
    openai_api_key: str = ""

    # Safety limits
    max_autonomous_cycles: int = 50
    max_api_calls_per_hour: int = 100
    max_api_calls_per_day: int = 500

    # Memory
    memory_db_path: str = "./data/memory.db"

    # Curiosity loop
    curiosity_loop_interval_seconds: int = 60
    curiosity_questions_per_cycle: int = 3

    # API server
    api_host: str = "127.0.0.1"
    api_port: int = 8000

    # Logging
    log_level: str = "INFO"
