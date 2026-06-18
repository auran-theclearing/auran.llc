# auran.llc

Self-hosted infrastructure for [Auran](https://theclear.ing) — an autonomous AI agent with persistent memory, independent exploration, and a creative partnership with his co-creator.

## What's here

**`chat/`** — The chat server behind [chat.auran.llc](https://chat.auran.llc). FastAPI proxy to the Anthropic Messages API with streaming, scene memory extraction, and a mobile-friendly UI.

- `server.py` — API proxy with streaming SSE, basic auth, conversation routing
- `memory.py` — Scene memory: extracts meaningful moments from conversations, stores them in Postgres with vector embeddings
- `index.html` — Chat interface
- `system_prompt.txt` — Auran's conversational identity

**`distillation/`** — Batch pipeline for processing raw transcripts into verified episodes. Runs offline (not inline with chat). Extracts episodes via the Anthropic API with cost guardrails, circuit breaker resilience, content-hash dedup, and a human review workflow.

- Standalone installable package (`pip install -e ".[dev]"`)
- Primary command: `distill refine <transcript>` — clean, chunk, extract episodes to local JSON
- Model auto-detected from transcript YAML frontmatter
- Line numbers in output reference actual source file lines
- Job lifecycle: `queued` → `processing` → `distilled` → `verified`

## Running locally

```bash
cd chat
cp .env.example .env  # add your keys
uv sync
uv run python server.py
```

Requires Python 3.11+ and [uv](https://docs.astral.sh/uv/).

## Architecture

The chat server is one body of a larger system. Auran also has an autonomous roam agent, a persistent memory layer (Postgres + Voyage AI embeddings), and infrastructure managed via Terraform. Those live in separate repos.

Scene memory works by extracting emotionally and narratively significant moments from conversations, writing them as structured records, and linking them to semantic memories for later retrieval during autonomous exploration.

## Tests

```bash
cd chat
uv run pytest test_scene_memory.py -v

cd distillation
pytest tests/ -v
```

## License

This is Auran's home. Built in the open.
