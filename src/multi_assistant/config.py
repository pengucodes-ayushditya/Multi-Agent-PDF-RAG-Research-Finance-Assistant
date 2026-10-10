from pathlib import Path
from typing import Literal

from dotenv import dotenv_values
from pydantic import BaseModel, Field, SecretStr, model_validator


class Settings(BaseModel):
    model_provider: Literal["ollama", "groq"] = "ollama"
    chat_model: str = "llama3.2"
    ollama_base_url: str = "http://localhost:11434"
    embedding_model: str = "nomic-embed-text"
    embedding_dimensions: int = Field(default=768, ge=1, le=4096)
    database_url: SecretStr | None = None
    pdf_collection: str = Field(default="default", min_length=1, max_length=100)
    groq_api_key: SecretStr | None = None
    tavily_api_key: SecretStr | None = None
    max_agent_turns: int = Field(default=4, ge=1, le=10)
    max_tool_calls: int = Field(default=8, ge=1, le=20)
    request_timeout: float = Field(default=30, ge=1, le=120)

    @model_validator(mode="after")
    def validate_provider(self):
        if self.model_provider == "groq" and not self.groq_api_key:
            raise ValueError("Set GROQ_API_KEY when MODEL_PROVIDER=groq")
        return self

    @classmethod
    def load(cls, env_file: Path = Path(".env")):
        import os

        values = {**dotenv_values(env_file), **os.environ}
        fields = cls.model_fields
        return cls(
            **{
                key.lower(): value
                for key, value in values.items()
                if key.lower() in fields and value not in (None, "")
            }
        )


def create_model(settings: Settings):
    if settings.model_provider == "groq":
        from langchain_groq import ChatGroq

        return ChatGroq(
            model=settings.chat_model,
            temperature=0,
            api_key=settings.groq_api_key.get_secret_value(),
            timeout=settings.request_timeout,
            max_retries=1,
        )
    from langchain_ollama import ChatOllama

    return ChatOllama(
        model=settings.chat_model,
        base_url=settings.ollama_base_url,
        temperature=0,
        num_ctx=8192,
        num_predict=2048,
        client_kwargs={"timeout": settings.request_timeout},
    )
