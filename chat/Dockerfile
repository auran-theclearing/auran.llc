FROM python:3.12-slim

COPY --from=ghcr.io/astral-sh/uv:latest /uv /uvx /bin/

WORKDIR /app

# Install dependencies first (cache-friendly layer ordering)
COPY pyproject.toml uv.lock ./
RUN uv sync --frozen --no-dev --no-install-project

# Copy application code
COPY server.py memory.py persistence.py graph_recall.py system_prompt.txt ./
COPY migrations/ ./migrations/
COPY index.html ./

ARG BUILD_SHA=dev
ENV PYTHONUNBUFFERED=1
ENV BUILD_SHA=${BUILD_SHA}

EXPOSE 8080

# Health check — hit the /health endpoint
HEALTHCHECK --interval=30s --timeout=5s --start-period=10s --retries=3 \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:8080/health')" || exit 1

CMD ["uv", "run", "uvicorn", "server:app", "--host", "0.0.0.0", "--port", "8080"]
