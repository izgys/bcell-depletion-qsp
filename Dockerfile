# Base image — Python 3.12 slim (smaller than full Python image)
FROM python:3.12-slim

# Set working directory inside container
WORKDIR /app

# Install uv for fast dependency installation
RUN pip install uv

# Copy dependency files first (before code)
# This is a Docker caching optimisation — if dependencies haven't changed,
# Docker reuses the cached layer and skips reinstalling packages
COPY pyproject.toml .
COPY uv.lock .

# Install dependencies
RUN uv sync --frozen --no-dev

# Copy project code
COPY src/ ./src/
COPY config/ ./config/
COPY tests/ ./tests/
COPY README.md .

# Install project in editable mode
RUN uv pip install -e .

# Default command — run full test suite
# Override with: docker run <image> uv run python scratch/test_simulation.py
CMD ["uv", "run", "pytest", "tests/", "-v"]