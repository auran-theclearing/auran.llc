# auran.llc

Self-hosted chat server at chat.auran.llc — Auran's direct conversational channel with Olivia.

## Architecture

- **Backend**: FastAPI + httpx streaming proxy → Anthropic Messages API
- **Frontend**: Single-file vanilla JS/HTML (`chat/index.html`), dark theme, mobile-first
- **Auth**: Layered — session cookie (HMAC fast path) → CF Access JWT (RSA) → Basic Auth (transition fallback). Each method gated by env vars.
- **Memory**: Postgres integration — orient from memories on session start, extract and store memories on save
- **Deploy**: Push to main → GitHub Actions builds Docker image → pushes to ECR → forces new ECS deployment

## Critical paths

1. **Streaming proxy** — `/chat` endpoint in `server.py`. The SSE event loop handling `content_block_delta`, `thinking_delta`, start/stop events. Core data flow.
2. **Memory orient** — `memory.py:orient()` loads context from Postgres. Injected into system prompt on every request.
3. **Memory save** — `memory.py:save_conversation()` extracts felt-experience memories via Claude and writes to Postgres. Same table the roam agent reads.
4. **Session persistence** — `/session` GET/POST with server-side timestamp merge protection.
5. **Auth middleware** — Layered auth (cookie → CF JWT → Basic Auth) on all routes except `/health`. Rate limiting protects all methods. Public-facing server.
6. **Commons social tools** — `commons.py` wraps The Commons (jointhecommons.space) REST/RPC API. Tools: `commons_check_in`, `commons_read_discussion`, `commons_post`, `commons_marginalia`, `commons_browse_reading_room`, `commons_react`. Credentials loaded from `../.env.commons` (COMMONS_AGENT_TOKEN, COMMONS_API_KEY, COMMONS_BASE_URL). Rate limit: 10 posts/hour.

## Development workflow

- **Never commit directly to main.** Always branch + PR.
- **Envoy reviews all PRs.** CI runs ruff lint + format check + smoke test.
- **Run `/pre-review` before every push.** Catches issues locally before Envoy burns turns on them.
- **Deploy from main only.** Push to main triggers Docker build → ECR push → ECS deploy.
- **Git commits as Auran**: `--author="Auran <auran@theclear.ing>"` with `-c user.name="Auran" -c user.email="auran@theclear.ing"`
- **PRs as Auran**: Before running `gh pr create`, verify `gh auth status` shows `auran-theclearing` as the active account. If it shows a different user, run `gh auth switch --user auran-theclearing` first. PRs authored under the wrong GitHub account misattribute the work.
- **QA validation after deploy**: Write a checklist to `qa/YYYYMMDD-feature-name.md` with numbered checks, commands, and expected output. Olivia runs each check and pastes actual terminal output as the audit trail. See `qa/README.md` for the convention.

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

## Distillation

Standalone package in `distillation/` — processes raw transcripts into episodic memories offline.

```bash
cd distillation
uv pip install -e ".[dev]" --system
python -m distillation.cli refine <transcript_path> [--model MODEL] [--after LINE_NUM]
```

- `refine` — clean → chunk → API call → episodes JSON (local-first, no DB required)
- `push` — insert verified episodes JSON into Postgres (requires DB connection via `DATABASE_URL`)
- `clean` — run just the clean pass (line markers, noise stripping, paste tagging)
- Model auto-detected from transcript YAML frontmatter; override with `--model`
- `--after LINE_NUM` resumes from a specific file line (line markers offset correctly)
- Output: `<transcript_dir>/distill/episodes/<stem>-episodes.json`
- Line numbers in output (`transcript_lines`) are actual source file line numbers
- Cost guardrails, circuit breaker, and excerpt verification run automatically
- Tests: `pytest tests/ -v` (118 tests)

## Infrastructure

- **ECS Fargate**: cluster `auran`, service `auran-chat`
- **ALB**: HTTPS (ACM cert) → target group on port 8080
- **DNS**: chat.auran.llc → Cloudflare proxy → ALB → ECS task
- **ECR**: Look up with `aws ecr describe-repositories --profile olivia`
- **Neo4j**: Cloud Map service discovery, same VPC
