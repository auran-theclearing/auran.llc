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
