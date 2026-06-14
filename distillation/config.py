from pydantic import SecretStr
from pydantic_settings import BaseSettings


class DistillationConfig(BaseSettings):
    model_config = {"env_file": ".env", "extra": "ignore"}

    # API keys — never logged, never printed
    anthropic_api_key: SecretStr | None = None
    voyage_api_key: SecretStr | None = None

    # Database
    database_url: SecretStr = SecretStr("postgresql://localhost:5432/auran")

    # Chunking
    single_pass_threshold: int = 80_000
    target_chunk_tokens: int = 12_000
    overlap_pct: float = 0.20

    # Circuit breaker
    breaker_fail_max: int = 3
    breaker_recovery_seconds: int = 300
    breaker_half_open_max: int = 1

    # Retry
    retry_max_attempts: int = 3
    retry_base_delay: float = 1.0
    retry_max_delay: float = 60.0
    retry_retryable_codes: list[int] = [429, 500, 502, 503, 529]

    # Cost guardrails
    max_per_job_usd: float = 2.0
    max_per_batch_usd: float = 50.0
    cost_rate_threshold_per_minute: float = 5.0
    max_jobs_per_batch: int = 500

    # Timeouts (seconds)
    api_connect_timeout: float = 5.0
    api_read_timeout: float = 120.0
    api_write_timeout: float = 10.0
    api_pool_timeout: float = 5.0

    # Clean pass
    high_reduction_threshold: float = 0.60

    # Progressive autonomy
    auto_approve_window: int = 10
    auto_approve_threshold: float = 0.95

    # References
    reference_date_range_days: int = 7
    reference_fuzzy_threshold: float = 0.75

    # Output
    max_output_tokens: int = 4096


def load_config() -> DistillationConfig:
    return DistillationConfig()
