import time

from distillation.resilience import CircuitBreaker, CircuitState


class TestCircuitBreaker:
    def test_starts_closed(self):
        cb = CircuitBreaker(fail_max=3, recovery_seconds=5)
        assert cb.state == CircuitState.CLOSED

    def test_opens_after_consecutive_failures(self):
        cb = CircuitBreaker(fail_max=3, recovery_seconds=5)
        cb.record_failure()
        cb.record_failure()
        assert cb.state == CircuitState.CLOSED
        cb.record_failure()
        assert cb.state == CircuitState.OPEN

    def test_open_breaker_blocks_requests(self):
        cb = CircuitBreaker(fail_max=3, recovery_seconds=5)
        for _ in range(3):
            cb.record_failure()
        assert cb.is_open()

    def test_success_resets_failure_count(self):
        cb = CircuitBreaker(fail_max=3, recovery_seconds=5)
        cb.record_failure()
        cb.record_failure()
        cb.record_success()
        assert cb.state == CircuitState.CLOSED
        cb.record_failure()
        assert cb.state == CircuitState.CLOSED

    def test_half_open_after_recovery_timeout(self):
        cb = CircuitBreaker(fail_max=3, recovery_seconds=0.1)
        for _ in range(3):
            cb.record_failure()
        assert cb.state == CircuitState.OPEN
        time.sleep(0.15)
        assert cb.state == CircuitState.HALF_OPEN

    def test_half_open_success_closes(self):
        cb = CircuitBreaker(fail_max=3, recovery_seconds=0.1)
        for _ in range(3):
            cb.record_failure()
        time.sleep(0.15)
        assert cb.state == CircuitState.HALF_OPEN
        cb.record_success()
        assert cb.state == CircuitState.CLOSED

    def test_half_open_failure_reopens(self):
        cb = CircuitBreaker(fail_max=3, recovery_seconds=0.1)
        for _ in range(3):
            cb.record_failure()
        time.sleep(0.15)
        assert cb.state == CircuitState.HALF_OPEN
        cb.record_failure()
        assert cb.state == CircuitState.OPEN

    def test_half_open_allows_one_request(self):
        cb = CircuitBreaker(fail_max=3, recovery_seconds=0.1, half_open_max=1)
        for _ in range(3):
            cb.record_failure()
        time.sleep(0.15)
        assert cb.can_proceed()
        cb.record_half_open_attempt()
        assert not cb.can_proceed()
