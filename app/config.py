from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    database_url: str = "sqlite:///./consilium.db"
    secret_key: str = "change-me-in-production-use-a-long-random-string"
    algorithm: str = "HS256"
    access_token_expire_minutes: int = 60 * 24 * 7  # 7 days
    storage_base_url: str = "https://storage.consilium.kz/files"
    upload_dir: str = "./uploads"
    chroma_dir: str = "./uploads/chroma_db"
    chunks_dir: str = "./uploads/knowledge/chunks"
    openrouter_consilium_key: str = ""

    model_config = SettingsConfigDict(env_file=".env", extra="ignore")


settings = Settings()
