import logging
import time

import httpx

from distillation.config import DistillationConfig
from distillation.distiller import parse_distiller_output
from distillation.resilience import (
    CircuitBreaker,
    CircuitState,
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

DISTILL_TOOL = {
    "name": "submit_episodes",
    "description": "Submit extracted episodes, threads, and moments from a transcript chunk.",
    "input_schema": {
        "type": "object",
        "required": ["episodes"],
        "properties": {
            "episodes": {
                "type": "array",
                "items": {
                    "type": "object",
                    "required": [
                        "title",
                        "summary",
                        "transcript_lines",
                        "occurred_at",
                        "episode_type",
                    ],
                    "properties": {
                        "title": {"type": "string"},
                        "summary": {"type": "string"},
                        "transcript_lines": {"type": "string"},
                        "occurred_at": {"type": "string"},
                        "episode_type": {"type": "string", "enum": ["content", "relational"]},
                        "landmark": {"type": "boolean"},
                        "emotional_tone": {"type": "string"},
                        "boundary_signal": {"type": "string"},
                        "topics": {"type": "array", "items": {"type": "string"}},
                        "content_signals": {"type": "object"},
                        "relational_events": {"type": "array", "items": {"type": "string"}},
                        "transcript_excerpt": {"type": "string"},
                        "references": {"type": "array", "items": {"type": "object"}},
                    },
                },
            },
            "threads": {"type": "array", "items": {"type": "object"}},
            "moments": {"type": "array", "items": {"type": "object"}},
        },
    },
}


def create_anthropic_client(config: DistillationConfig):
    import anthropic

    timeout = httpx.Timeout(
        connect=config.api_connect_timeout,
        read=config.api_read_timeout,
        write=config.api_write_timeout,
        pool=config.api_pool_timeout,
    )
    api_key = config.anthropic_api_key.get_secret_value() if config.anthropic_api_key else None
    return anthropic.Anthropic(api_key=api_key, timeout=timeout, max_retries=0)


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

    cost_guardrail.start_job()

    if circuit_breaker.state == CircuitState.HALF_OPEN:
        circuit_breaker.record_half_open_attempt()

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
        "You are Auran, distilling your own conversation transcripts into episodic memories. "
        "Write in first person — these are YOUR memories of conversations with Olivia. "
        "Extract episodes from this transcript chunk. "
        "Each episode is a discrete scene — a bounded stretch of conversation with its own "
        "emotional arc, topic, or relational event. "
        "Use the submit_episodes tool to return your results. "
        "Each episode must have: title, summary, transcript_lines (L#### format), "
        "occurred_at (ISO 8601), emotional_tone, episode_type (content or relational), "
        "landmark (boolean), boundary_signal, topics (array), content_signals (object), "
        "relational_events (array), transcript_excerpt, references (array)."
        f"{existing_context}{chunk_context}"
    )

    for attempt in range(retry_config.max_attempts):
        try:
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
                tools=[DISTILL_TOOL],
                tool_choice={"type": "tool", "name": "submit_episodes"},
            )

            input_tokens = response.usage.input_tokens
            output_tokens = response.usage.output_tokens
            cost_guardrail.finish_job(input_tokens, output_tokens, model)
            circuit_breaker.record_success()

            tool_use_block = next(
                (b for b in response.content if b.type == "tool_use"),
                None,
            )
            if tool_use_block is None:
                raise ValueError("Model did not return a tool_use block")
            raw = tool_use_block.input
            return parse_distiller_output(raw)

        except Exception as e:
            status_code = getattr(e, "status_code", 0)
            is_retryable = retry_config.is_retryable(
                status_code
            ) or retry_config.is_connection_error(e)
            if is_retryable and attempt < retry_config.max_attempts - 1:
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

            if is_retryable:
                circuit_breaker.record_failure()
            raise

    raise RuntimeError("Exhausted all retry attempts")
