from functools import lru_cache
from pydantic import BaseModel
import os


class Settings(BaseModel):
    app_version: str = "1.0.0"
    dataset_path: str = os.getenv("DATASET_PATH", "data/ecommerce_faq.csv")
    chroma_persist_dir: str = os.getenv("CHROMA_PERSIST_DIR", "chroma_data")
    chroma_collection: str = os.getenv("CHROMA_COLLECTION", "ecommerce_faq")
    embedding_model: str = os.getenv("EMBEDDING_MODEL", "all-MiniLM-L6-v2")
    llm_provider: str = os.getenv("LLM_PROVIDER", "ollama")
    llm_model: str = os.getenv("LLM_MODEL", "llama3")
    openrouter_api_key: str | None = os.getenv("OPENROUTER_API_KEY")
    gemini_api_key: str | None = os.getenv("GEMINI_API_KEY")
    ollama_base_url: str = os.getenv("OLLAMA_BASE_URL", "http://ollama:11434")


@lru_cache
def get_settings() -> Settings:
    return Settings()
