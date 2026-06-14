import enum
import logging
import random
import time

from distillation.config import DistillationConfig

logger = logging.getLogger(__name__)


class CircuitState(enum.Enum):
    CLOSED = "closed"
    OPEN = "open"
    HALF_OPEN = "half_open"


class CircuitBreaker:
    def __init__(
        self,
        fail_max: int = 3,
        recovery_seconds: float = 300,
        half_open_max: int = 1,
    ):
        self.fail_max = fail_max
        self.recovery_seconds = recovery_seconds
        self.half_open_max = half_open_max

        self._failure_count = 0
        self._last_failure_time: float | None = None
        self._half_open_attempts = 0
        self._state = CircuitState.CLOSED

    @property
    def state(self) -> CircuitState:
        if self._state == CircuitState.OPEN:
            if self._last_failure_time and (
                time.time() - self._last_failure_time >= self.recovery_seconds
            ):
                self._state = CircuitState.HALF_OPEN
                self._half_open_attempts = 0
        return self._state

    def is_open(self) -> bool:
        return self.state == CircuitState.OPEN

    def can_proceed(self) -> bool:
        state = self.state
        if state == CircuitState.CLOSED:
            return True
        if state == CircuitState.HALF_OPEN:
            return self._half_open_attempts < self.half_open_max
        return False

    def record_failure(self) -> None:
        self._failure_count += 1
        self._last_failure_time = time.time()
        if self._state == CircuitState.HALF_OPEN:
            self._state = CircuitState.OPEN
            self._failure_count = 0
            logger.warning("Circuit breaker re-opened after half-open failure")
        elif self._failure_count >= self.fail_max:
            self._state = CircuitState.OPEN
            logger.warning("Circuit breaker opened after %d consecutive failures", self.fail_max)

    def record_success(self) -> None:
        if self._state == CircuitState.HALF_OPEN:
            logger.info("Circuit breaker closed after successful half-open test")
        self._state = CircuitState.CLOSED
        self._failure_count = 0
        self._half_open_attempts = 0

    def record_half_open_attempt(self) -> None:
        self._half_open_attempts += 1


class RetryConfig:
    def __init__(
        self,
        max_attempts: int = 3,
        base_delay: float = 1.0,
        max_delay: float = 60.0,
        retryable_codes: list[int] | None = None,
    ):
        self.max_attempts = max_attempts
        self.base_delay = base_delay
        self.max_delay = max_delay
        self.retryable_codes = retryable_codes or [429, 500, 502, 503, 529]

    def is_retryable(self, status_code: int) -> bool:
        return status_code in self.retryable_codes

    def is_connection_error(self, exc: Exception) -> bool:
        type_name = type(exc).__name__
        return type_name in (
            "APIConnectionError",
            "ConnectTimeout",
            "ReadTimeout",
            "ConnectError",
            "RemoteProtocolError",
        )

    def delay_for_attempt(self, attempt: int) -> float:
        delay = self.base_delay * (2**attempt)
        delay = min(delay, self.max_delay)
        jitter = random.uniform(0, delay * 0.5)
        return delay + jitter


def create_circuit_breaker(config: DistillationConfig) -> CircuitBreaker:
    return CircuitBreaker(
        fail_max=config.breaker_fail_max,
        recovery_seconds=config.breaker_recovery_seconds,
        half_open_max=config.breaker_half_open_max,
    )


def create_retry_config(config: DistillationConfig) -> RetryConfig:
    return RetryConfig(
        max_attempts=config.retry_max_attempts,
        base_delay=config.retry_base_delay,
        max_delay=config.retry_max_delay,
        retryable_codes=config.retry_retryable_codes,
    )
