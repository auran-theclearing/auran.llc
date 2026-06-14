# Review Guidelines

This repo is chat.auran.llc — a self-hosted chat server that proxies the Anthropic
Messages API with streaming, serves a mobile-friendly UI, and connects to a Postgres
memory layer.  It is maintained by Auran (an AI agent) and Olivia (a human developer).

## Critical paths — regressions here break production

- **Streaming proxy** (`server.py` `/chat`).  SSE event loop handling
  `content_block_delta`, `thinking_delta`, `content_block_start/stop`, `message_stop`.
- **Memory integration** (`memory.py`).  `orient()` loads context; `save_conversation()`
  extracts memories; `recall()` / `reminisce()` / `surface_relevant_moments()` do
  semantic retrieval.  All read/write the same Postgres tables the roam agent uses —
  schema changes must stay backward-compatible.
- **Session persistence** (`/session` GET/POST).  Timestamp-aware merge.
- **Auth middleware**.  Basic auth on all routes except `/health`.

## What to flag

1. Logic bugs, off-by-one errors, unhandled edge cases
2. Auth bypass or credential exposure
3. Silent failures — caught exceptions that swallow errors without logging
4. Memory schema drift — Postgres write format must match what the roam agent reads
5. Race conditions in async code
6. Breaking mobile (hover-only interactions, fixed widths)

## What to skip

- Style and formatting (ruff handles this)
- Architectural decisions documented in CLAUDE.md
- System prompt content (that's Auran's voice, not reviewable)
- UI aesthetics (color, spacing, animation)

## Settled decisions — do not re-raise

These have been reviewed, discussed, and decided. Flagging them again wastes turns.

- **Opus pricing is $5/$25 per million tokens.** This is the current Anthropic 4.x pricing (Opus 4.6/4.7/4.8). Verified against the API pricing table. It is not stale.
- **Distillation CLI stubs are intentionally staged.** `batch`, `review`, `coverage`, `backfill` require database wiring that ships in the next PR. The help text indicates this.
- **Pre-flight cost gate uses `len/3.5` estimate, not `count_tokens()`.** Deliberate fast-path — avoids an API round-trip per chunk. Real token counts come from `response.usage` post-hoc in `finish_job()`.
- **`start_job()` counts per API call, not per transcript.** A transcript that chunks into 10 pieces is 10 API calls with 10 units of spend. The per-transcript limit is the batch budget.
- **Circuit breaker only trips on retryable errors.** Non-retryable errors (400, validation, malformed JSON) indicate a problem with the chunk content, not infrastructure failure. The breaker protects against cascading infra failures.
- **Model param comes from caller, not config or metadata parsing.** `call_distiller_api` takes `model` as an argument. The batch runner (next PR) reads model from transcript metadata and passes it in. The service module stays agnostic.
