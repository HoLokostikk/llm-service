import os


class Settings:
    ollama_url: str = os.getenv("OLLAMA_URL", "http://localhost:11434")
    model: str = os.getenv("MODEL", "qwen2.5:0.5b-instruct-q8_0")
    request_timeout: float = float(os.getenv("REQUEST_TIMEOUT", "120"))
    log_level: str = os.getenv("LOG_LEVEL", "INFO")


settings = Settings()
