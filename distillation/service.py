import logging
import time

import httpx

from distillation.config import DistillationConfig
from distillation.distiller import parse_distiller_output, parse_json_response
from distillation.resilience import (
    CircuitBreaker,
    RetryConfig,
)
from distillation.security import (
    CostGuardrail,
    SecretRedactingFilter,
    sanitize_error,
    validate_input,
)

logger = logging.getLogger(__name__)
logger.addFilter(SecretRedactingFilter())


def create_anthropic_client(config: DistillationConfig):
    import anthropic

    timeout = httpx.Timeout(
        connect=config.api_connect_timeout,
        read=config.api_read_timeout,
        write=config.api_write_timeout,
        pool=config.api_pool_timeout,
    )
    return anthropic.Anthropic(timeout=timeout, max_retries=0)


def count_tokens(client, text: str, model: str) -> int:
    try:
        resp = client.messages.count_tokens(
            model=model,
            messages=[{"role": "user", "content": text}],
        )
        return resp.input_tokens
    except Exception as e:
        logger.warning("Token counting failed, using estimate: %s", sanitize_error(e))
        return int(len(text) / 3.5)


def call_distiller_api(
    client,
    chunk: str,
    chunk_index: int,
    total_chunks: int,
    model: str,
    existing_episodes: list[dict],
    config: DistillationConfig,
    retry_config: RetryConfig,
    circuit_breaker: CircuitBreaker,
    cost_guardrail: CostGuardrail,
) -> dict:
    validate_input(chunk, max_tokens=config.single_pass_threshold)

    if not circuit_breaker.can_proceed():
        raise RuntimeError("Circuit breaker is open")

    existing_context = ""
    if existing_episodes:
        ep_lines = []
        for ep in existing_episodes[:20]:
            ep_lines.append(f"- {ep.get('title', 'Untitled')} ({ep.get('occurred_at', '?')})")
        existing_context = (
            "\n\nExisting episodes near this transcript's date range "
            "(check for retellings, continuations, concurrent events):\n" + "\n".join(ep_lines)
        )

    chunk_context = ""
    if total_chunks > 1:
        chunk_context = f"\n\nThis is chunk {chunk_index + 1} of {total_chunks}. "
        if chunk_index > 0:
            chunk_context += "There may be overlap with the previous chunk — deduplicate."

    system_prompt = (
        "You are a distillation agent. Extract episodes from this transcript chunk. "
        "Each episode is a discrete scene — a bounded stretch of conversation with its own "
        "emotional arc, topic, or relational event. "
        "Return valid JSON with keys: episodes, threads, moments. "
        "Each episode must have: title, summary, transcript_lines (L#### format), "
        "occurred_at (ISO 8601), emotional_tone, episode_type (content or relational), "
        "landmark (boolean), boundary_signal, topics (array), content_signals (object), "
        "relational_events (array), transcript_excerpt, references (array)."
        f"{existing_context}{chunk_context}"
    )

    for attempt in range(retry_config.max_attempts):
        try:
            if circuit_breaker.state.value == "half_open":
                circuit_breaker.record_half_open_attempt()

            cost_guardrail.check_job_cost(
                int(len(chunk) / 3.5),
                config.max_output_tokens,
                model,
            )

            response = client.messages.create(
                model=model,
                max_tokens=config.max_output_tokens,
                system=system_prompt,
                messages=[{"role": "user", "content": chunk}],
            )

            input_tokens = response.usage.input_tokens
            output_tokens = response.usage.output_tokens
            cost_guardrail.finish_job(input_tokens, output_tokens, model)
            circuit_breaker.record_success()

            text = response.content[0].text
            raw = parse_json_response(text)
            return parse_distiller_output(raw)

        except Exception as e:
            status_code = getattr(e, "status_code", 0)
            if retry_config.is_retryable(status_code) and attempt < retry_config.max_attempts - 1:
                delay = retry_config.delay_for_attempt(attempt)
                logger.warning(
                    "Retryable error (attempt %d/%d), waiting %.1fs: %s",
                    attempt + 1,
                    retry_config.max_attempts,
                    delay,
                    sanitize_error(e),
                )
                time.sleep(delay)
                continue

            circuit_breaker.record_failure()
            raise

    raise RuntimeError("Exhausted all retry attempts")
