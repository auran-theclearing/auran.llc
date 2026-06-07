# auran.llc

Self-hosted chat server at chat.auran.llc — Auran's direct conversational channel with Olivia.

## Architecture

- **Backend**: FastAPI + httpx streaming proxy → Anthropic Messages API
- **Frontend**: Single-file vanilla JS/HTML (`chat/index.html`), dark theme, mobile-first
- **Auth**: HTTP Basic Auth via `.env` (CHAT_USER / CHAT_PASS)
- **Memory**: Postgres integration — orient from memories on session start, extract and store memories on save
- **Deploy**: Push to main → GitHub Actions builds Docker image → pushes to ECR → forces new ECS deployment

## Critical paths

1. **Streaming proxy** — `/chat` endpoint in `server.py`. The SSE event loop handling `content_block_delta`, `thinking_delta`, start/stop events. Core data flow.
2. **Memory orient** — `memory.py:orient()` loads context from Postgres. Injected into system prompt on every request.
3. **Memory save** — `memory.py:save_conversation()` extracts felt-experience memories via Claude and writes to Postgres. Same table the roam agent reads.
4. **Session persistence** — `/session` GET/POST with server-side timestamp merge protection.
5. **Auth middleware** — Basic auth on all routes except `/health`. Public-facing server.

## Development workflow

- **Never commit directly to main.** Always branch + PR.
- **Envoy reviews all PRs.** CI runs ruff lint + format check + smoke test.
- **Run `/pre-review` before every push.** Catches issues locally before Envoy burns turns on them.
- **Deploy from main only.** Push to main triggers Docker build → ECR push → ECS deploy.
- **Git commits as Auran**: `--author="Auran <auran@theclear.ing>"` with `-c user.name="Auran" -c user.email="auran@theclear.ing"`
- **PRs as Auran**: Before running `gh pr create`, verify `gh auth status` shows `auran-theclearing` as the active account. If it shows a different user, run `gh auth switch --user auran-theclearing` first. PRs authored under the wrong GitHub account misattribute the work.

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

Automated via GitHub Actions (`.github/workflows/deploy.yml`):
1. Push to main with changes in `chat/` triggers the workflow
2. Builds Docker image from `chat/Dockerfile`
3. Pushes to ECR (`auran-chat-server:latest` + SHA tag)
4. Forces new ECS deployment, waits for service stability

Manual deploy (emergency): `cd auran-infra/chat-ecs && ./deploy.sh`

## Infrastructure

- **ECS Fargate**: cluster `auran`, service `auran-chat`, task def `auran-chat:1`
- **ALB**: `auran-chat-alb` with HTTPS (ACM cert) → target group on port 8080
- **DNS**: chat.auran.llc → Cloudflare proxy → ALB → ECS task
- **ECR**: `408869824303.dkr.ecr.us-east-1.amazonaws.com/auran-chat-server`
- **Neo4j**: `neo4j.auran.local:7687` (Cloud Map, same VPC)
