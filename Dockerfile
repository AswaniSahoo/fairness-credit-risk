FROM python:3.11-slim

WORKDIR /app

# Build toolchain is needed for the boosting library wheels on slim images.
RUN apt-get update && apt-get install -y --no-install-recommends \
    gcc \
    g++ \
    && rm -rf /var/lib/apt/lists/*

COPY requirements-api.txt .

RUN --mount=type=cache,target=/root/.cache/pip \
    pip install --default-timeout=1000 -r requirements-api.txt

COPY api/ ./api/
COPY artifacts/ ./artifacts/
COPY config/ ./config/
COPY src/ ./src/
# /metrics serves the recorded test-block metrics straight from this artifact rather than
# from hardcoded values, so the image is broken without it.
COPY reports/track_comparison.json ./reports/track_comparison.json

EXPOSE 8000

# `requests` is pinned in requirements-api.txt so this probe can actually run, and a
# non-2xx response must fail the check rather than pass silently.
HEALTHCHECK --interval=30s --timeout=10s --start-period=15s --retries=3 \
    CMD python -c "import requests, sys; sys.exit(0 if requests.get('http://localhost:8000/health', timeout=5).ok else 1)"

CMD ["uvicorn", "api.main:app", "--host", "0.0.0.0", "--port", "8000"]
