# ══════════════════════════════════════════════════════════════════════════════
# STAGE 1: Builder (Install dependencies)
# ══════════════════════════════════════════════════════════════════════════════
FROM python:3.10-slim AS builder

WORKDIR /app

# Install system dependencies needed to build some python libraries
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Install Poetry
ENV POETRY_VERSION=2.0.0
ENV POETRY_HOME=/opt/poetry
RUN curl -sSL https://install.python-poetry.org | python3 -
ENV PATH="${POETRY_HOME}/bin:${PATH}"

# Copy only project dependency definitions
COPY pyproject.toml poetry.lock ./

# Configure poetry to not create a virtual env (or create it inside /app/.venv)
RUN poetry config virtualenvs.in-project true \
    && poetry install --no-root --no-interaction --no-ansi

# ══════════════════════════════════════════════════════════════════════════════
# STAGE 2: Final Production Image
# ══════════════════════════════════════════════════════════════════════════════
FROM python:3.10-slim AS runner

WORKDIR /app

# Create a non-privileged system user for running the app securely
RUN groupadd -g 999 appuser && \
    useradd -r -u 999 -g appuser appuser

# Copy virtualenv and installed packages from builder
COPY --from=builder /app/.venv /app/.venv

# Copy source code, models, and reference data
COPY src/ /app/src/
COPY data/ /app/data/

# Set environment variables
ENV PATH="/app/.venv/bin:${PATH}"
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

# Expose standard FastAPI port
EXPOSE 8000

# Set ownership to non-root user
RUN chown -R appuser:appuser /app

# Switch to non-privileged user
USER appuser

# Healthcheck to verify FastAPI is up (uses built-in python to avoid installing curl on slim image)
HEALTHCHECK --interval=30s --timeout=5s --start-period=5s --retries=3 \
    CMD python3 -c "import urllib.request; urllib.request.urlopen('http://localhost:8000/')" || exit 1

# Command to run FastAPI server under Uvicorn
CMD ["uvicorn", "src.app:app", "--host", "0.0.0.0", "--port", "8000"]
