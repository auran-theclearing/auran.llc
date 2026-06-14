import logging
import re
import time
from dataclasses import dataclass, field

SECRET_PATTERNS = [
    re.compile(r"sk-ant-[a-zA-Z0-9_-]+"),
    re.compile(r"pa-[a-zA-Z0-9_-]+"),
    re.compile(r"AKIA[A-Z0-9]{16}"),
    re.compile(r"(?i)(bearer\s+)\S+"),
    re.compile(r"(?i)(authorization[:\s]+)\S+"),
]


class SecretRedactingFilter(logging.Filter):
    def filter(self, record: logging.LogRecord) -> bool:
        if record.args:
            try:
                record.msg = record.msg % record.args
            except (TypeError, ValueError):
                record.msg = str(record.msg)
            record.args = None
        msg = str(record.msg)
        for pattern in SECRET_PATTERNS:
            msg = pattern.sub("[REDACTED]", msg)
        record.msg = msg
        return True


def sanitize_error(exc: Exception) -> str:
    msg = str(exc)
    for pattern in SECRET_PATTERNS:
        msg = pattern.sub("[REDACTED]", msg)
    return msg


def validate_input(content: str, max_tokens: int, token_estimate_divisor: float = 3.5) -> None:
    if not content or len(content) < 10:
        raise ValueError("Content too short to distill")
    estimated_tokens = int(len(content) / token_estimate_divisor)
    if estimated_tokens > max_tokens:
        raise ValueError(
            f"Input too large: ~{estimated_tokens} estimated tokens exceeds max {max_tokens}"
        )


class BudgetExhaustedError(Exception):
    pass


class JobCostExceededError(Exception):
    pass


class BatchBudgetExhaustedError(Exception):
    pass


class CostRateExceededError(Exception):
    pass


# Pricing per million tokens (input, output) — looked up by model prefix
MODEL_PRICING = {
    "claude-opus": (5.0, 25.0),
    "claude-sonnet": (3.0, 15.0),
    "claude-haiku": (1.0, 5.0),
}


logger = logging.getLogger(__name__)


def _get_pricing(model: str) -> tuple[float, float]:
    for prefix, pricing in MODEL_PRICING.items():
        if model.startswith(prefix):
            return pricing
    logger.warning("Unknown model %r — falling back to Opus pricing ($5/$25)", model)
    return (5.0, 25.0)


@dataclass
class CostGuardrail:
    max_per_job_usd: float = 2.0
    max_per_batch_usd: float = 50.0
    cost_rate_threshold_per_minute: float = 5.0
    max_jobs_per_batch: int = 500

    batch_spent_usd: float = field(default=0.0, init=False)
    job_count: int = field(default=0, init=False)
    _cost_events: list[tuple[float, float]] = field(default_factory=list, init=False)

    def estimate_cost(self, input_tokens: int, output_tokens: int, model: str) -> float:
        input_price, output_price = _get_pricing(model)
        return (input_tokens * input_price + output_tokens * output_price) / 1_000_000

    def record_usage(self, input_tokens: int, output_tokens: int, model: str) -> float:
        cost = self.estimate_cost(input_tokens, output_tokens, model)
        self.batch_spent_usd += cost
        self._cost_events.append((time.time(), cost))
        return cost

    def check_job_cost(self, input_tokens: int, output_tokens: int, model: str) -> None:
        cost = self.estimate_cost(input_tokens, output_tokens, model)
        if cost > self.max_per_job_usd:
            raise JobCostExceededError(
                f"Estimated job cost ${cost:.4f} exceeds max ${self.max_per_job_usd}"
            )

    def check_batch_budget(self) -> None:
        if self.batch_spent_usd >= self.max_per_batch_usd:
            raise BatchBudgetExhaustedError(
                f"Batch budget exhausted: ${self.batch_spent_usd:.2f} >= ${self.max_per_batch_usd}"
            )
        if self.job_count > self.max_jobs_per_batch:
            raise BatchBudgetExhaustedError(
                f"Batch job limit reached: {self.job_count} > {self.max_jobs_per_batch}"
            )

    def check_cost_rate(self) -> None:
        now = time.time()
        cutoff = now - 60.0
        self._cost_events = [(t, c) for t, c in self._cost_events if t > cutoff]
        recent_cost = sum(c for _, c in self._cost_events)
        if recent_cost > self.cost_rate_threshold_per_minute:
            threshold = self.cost_rate_threshold_per_minute
            raise CostRateExceededError(
                f"Cost rate ${recent_cost:.2f}/min exceeds ${threshold}/min"
            )

    def start_job(self) -> None:
        self.job_count += 1
        self.check_batch_budget()
        self.check_cost_rate()

    def finish_job(self, input_tokens: int, output_tokens: int, model: str) -> float:
        cost = self.record_usage(input_tokens, output_tokens, model)
        self.check_batch_budget()
        return cost

    def reset(self) -> None:
        self.batch_spent_usd = 0.0
        self.job_count = 0
        self._cost_events.clear()
