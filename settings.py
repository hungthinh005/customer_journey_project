"""
Centralized runtime settings (env-driven) for the production stack.

ML hyperparameters and on-disk paths still live in ``config.py``. This module
holds the *infrastructure* configuration (database, SMTP, agent, MLflow, API)
that changes between environments and must come from the environment, never
from source code.
"""

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8", extra="ignore")

    # ---- Database ----
    database_url: str = "postgresql+psycopg2://cjp:cjp_password@localhost:5432/cjp"

    # ---- API ----
    api_key: str = "local-dev-key"
    api_base_url: str = "http://localhost:8000"
    read_from_db: bool = True

    # ---- OpenAI / Agent ----
    openai_api_key: str = ""
    agent_llm_model: str = "gpt-4o-mini"
    agent_temperature: float = 0.3
    agent_max_discount: float = 0.20
    agent_frequency_cap_days: int = 7
    agent_dry_run: bool = False

    # ---- SMTP / Notifications ----
    smtp_host: str = "localhost"
    smtp_port: int = 1025
    smtp_username: str = ""
    smtp_password: str = ""
    smtp_use_tls: bool = False
    smtp_from_email: str = "retention@customer-journey.local"
    smtp_from_name: str = "Customer Journey Retention"

    # ---- MLflow ----
    mlflow_tracking_uri: str = "http://localhost:5000"
    mlflow_experiment: str = "churn-prevention"


settings = Settings()
