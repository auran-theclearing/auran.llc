# auran.llc

Self-hosted chat server at chat.auran.llc — Auran's direct conversational channel with Olivia.

## Architecture

- **Backend**: FastAPI + httpx streaming proxy → Anthropic Messages API
- **Frontend**: Single-file vanilla JS/HTML (`chat/index.html`), dark theme, mobile-first
- **Auth**: HTTP Basic Auth via `.env` (CHAT_USER / CHAT_PASS)
- **Memory**: Postgres integration — orient from memories on session start, extract and store memories on save
- **Deploy**: S3 sync → SSM send-command → EC2 pulls from S3 → systemctl restart

## Critical paths

1. **Streaming proxy** — `/chat` endpoint in `server.py`. The SSE event loop handling `content_block_delta`, `thinking_delta`, start/stop events. Core data flow.
2. **Memory orient** — `memory.py:orient()` loads context from Postgres. Injected into system prompt on every request.
3. **Memory save** — `memory.py:save_conversation()` extracts felt-experience memories via Claude and writes to Postgres. Same table the roam agent reads.
4. **Session persistence** — `/session` GET/POST with server-side timestamp merge protection.
5. **Auth middleware** — Basic auth on all routes except `/health`. Public-facing server.

## Development workflow

- **Never commit directly to main.** Always branch + PR.
- **Envoy reviews all PRs.** CI runs ruff lint + format check + smoke test.
- **Deploy from main only.** Push to main triggers S3 sync → SSM deploy.
- **Git commits as Auran**: `--author="Auran <auran@theclear.ing>"` with `-c user.name="Auran" -c user.email="auran@theclear.ing"`

## Linting

```bash
cd chat
uv run ruff check .        # lint
uv run ruff format .       # format
uv run ruff check --fix .  # auto-fix
```

## Running locally

```bash
cd chat
cp .env.example .env  # fill in ANTHROPIC_API_KEY, CHAT_USER, CHAT_PASS
uv run python server.py
```

## Deploy

```bash
# From Cowork sandbox — use dispatch relay:
# 1. git push via dispatch
# 2. S3 sync via dispatch
# 3. SSM restart via dispatch
# See charting_territory/CLAUDE.md for dispatch command format
```

## Instance

- **EC2**: `i-070dd19cbf382a171` (auran-voice, 3.92.95.65) — shared voice/chat instance
- **Port**: 8080 (Cloudflare proxies HTTPS → HTTP)
- **DNS**: chat.auran.llc → Cloudflare proxy → EC2
