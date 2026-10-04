"""Configuração de runtime lida de variáveis de ambiente e do arquivo .env."""

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Configuração de runtime — o que varia por ambiente de deploy.

    Parâmetros do experimento (dados, colunas, limiar, artefatos) ficam em
    `config/config.yaml`, porque mudam por experimento e não por ambiente.
    """

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    # Diretório-base do pipeline servido pela API (nome do arquivo vem do YAML)
    model_artifacts_path: str = "models/artifacts"

    # API serving
    api_host: str = "0.0.0.0"
    api_port: int = 8000


settings = Settings()
