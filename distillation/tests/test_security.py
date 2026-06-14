import logging

import pytest

from distillation.security import (
    BatchBudgetExhaustedError,
    CostGuardrail,
    CostRateExceededError,
    JobCostExceededError,
    SecretRedactingFilter,
    sanitize_error,
    validate_input,
)


class TestSecretRedactingFilter:
    def test_redacts_anthropic_key(self):
        filt = SecretRedactingFilter()
        record = logging.LogRecord(
            "test",
            logging.INFO,
            "",
            0,
            "Using key sk-ant-abc123-def456_ghi",
            None,
            None,
        )
        filt.filter(record)
        assert "sk-ant-" not in record.msg
        assert "[REDACTED]" in record.msg

    def test_redacts_voyage_key(self):
        filt = SecretRedactingFilter()
        record = logging.LogRecord(
            "test",
            logging.INFO,
            "",
            0,
            "Voyage key: pa-abc123xyz",
            None,
            None,
        )
        filt.filter(record)
        assert "pa-abc" not in record.msg
        assert "[REDACTED]" in record.msg

    def test_redacts_aws_key(self):
        filt = SecretRedactingFilter()
        record = logging.LogRecord(
            "test",
            logging.INFO,
            "",
            0,
            "AWS: AKIAIOSFODNN7EXAMPLE",
            None,
            None,
        )
        filt.filter(record)
        assert "AKIA" not in record.msg

    def test_redacts_bearer_token(self):
        filt = SecretRedactingFilter()
        record = logging.LogRecord(
            "test",
            logging.INFO,
            "",
            0,
            "Header: Bearer eyJhbGciOiJSUzI1NiJ9.payload.sig",
            None,
            None,
        )
        filt.filter(record)
        assert "eyJ" not in record.msg

    def test_passes_clean_messages(self):
        filt = SecretRedactingFilter()
        record = logging.LogRecord(
            "test",
            logging.INFO,
            "",
            0,
            "Processing job for transcript 20260603",
            None,
            None,
        )
        filt.filter(record)
        assert record.msg == "Processing job for transcript 20260603"


class TestSanitizeError:
    def test_strips_api_key_from_exception(self):
        exc = Exception("Request failed with key sk-ant-secret123-value456")
        result = sanitize_error(exc)
        assert "sk-ant-" not in result
        assert "[REDACTED]" in result

    def test_strips_authorization_header(self):
        exc = Exception("Authorization: Bearer token123abc")
        result = sanitize_error(exc)
        assert "token123abc" not in result


class TestValidateInput:
    def test_rejects_empty_content(self):
        with pytest.raises(ValueError, match="too short"):
            validate_input("", max_tokens=100_000)

    def test_rejects_too_short(self):
        with pytest.raises(ValueError, match="too short"):
            validate_input("hi", max_tokens=100_000)

    def test_rejects_oversized_content(self):
        huge = "word " * 500_000
        with pytest.raises(ValueError, match="too large"):
            validate_input(huge, max_tokens=100_000)

    def test_accepts_valid_content(self):
        content = "word " * 1000
        validate_input(content, max_tokens=100_000)


class TestCostGuardrail:
    def test_estimates_cost(self):
        guard = CostGuardrail()
        cost = guard.estimate_cost(10_000, 1_000, "claude-sonnet-4-6")
        assert cost == pytest.approx((10_000 * 3.0 + 1_000 * 15.0) / 1_000_000)

    def test_opus_pricing(self):
        guard = CostGuardrail()
        cost = guard.estimate_cost(10_000, 1_000, "claude-opus-4-8")
        assert cost == pytest.approx((10_000 * 5.0 + 1_000 * 25.0) / 1_000_000)

    def test_job_cost_exceeded(self):
        guard = CostGuardrail(max_per_job_usd=0.001)
        with pytest.raises(JobCostExceededError):
            guard.check_job_cost(1_000_000, 100_000, "claude-opus-4-8")

    def test_batch_budget_exhausted(self):
        guard = CostGuardrail(max_per_batch_usd=0.001)
        guard.record_usage(100_000, 10_000, "claude-opus-4-8")
        with pytest.raises(BatchBudgetExhaustedError):
            guard.check_batch_budget()

    def test_cost_rate_exceeded(self):
        guard = CostGuardrail(cost_rate_threshold_per_minute=0.0001)
        guard.record_usage(100_000, 10_000, "claude-opus-4-8")
        with pytest.raises(CostRateExceededError):
            guard.check_cost_rate()

    def test_job_count_limit(self):
        guard = CostGuardrail(max_jobs_per_batch=3)
        guard.start_job()
        guard.start_job()
        guard.start_job()
        with pytest.raises(BatchBudgetExhaustedError):
            guard.start_job()

    def test_reset_clears_state(self):
        guard = CostGuardrail()
        guard.record_usage(100_000, 10_000, "claude-opus-4-8")
        guard.start_job()
        guard.reset()
        assert guard.batch_spent_usd == 0.0
        assert guard.job_count == 0
