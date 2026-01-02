# --- Stage 1: The "Builder" ---
FROM python:3.12-bookworm AS builder

WORKDIR /app

# Create virtual environment
RUN python -m venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"

# Install dependencies
COPY requirements.txt .
# upgrade pip to ensure wheel compatibility for awscrt
RUN pip install --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

# --- Stage 2: The "Runner" ---
FROM python:3.12-slim-bookworm AS runner

# Create a non-root user (Security Best Practice)
ARG UID=1001
ARG GID=1001
RUN groupadd --gid ${GID} appgroup && useradd --uid ${UID} --gid appgroup --shell /bin/bash --create-home appuser

WORKDIR /app

# Copy the virtual environment from builder
COPY --from=builder /opt/venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"

# --- COPY APPLICATION CODE ---
# We explicitly copy folders to keep the layer clean
COPY --chown=appuser:appgroup backend ./backend
COPY --chown=appuser:appgroup core ./core
COPY --chown=appuser:appgroup config ./config
COPY --chown=appuser:appgroup prompts ./prompts
COPY --chown=appuser:appgroup data ./data

# (Optional) ChromaDB is currently ignored in .dockerignore for the 'Hybrid' demo.
# If you want to bake it in later, remove it from .dockerignore and uncomment this:
# COPY --chown=appuser:appgroup chroma_db ./chroma_db

# Switch to non-root user
USER appuser

# Expose the port
EXPOSE 8000

# Run the application
# We use shell expansion for the PORT variable (Default: 8000)
CMD ["/bin/sh", "-c", "uvicorn backend.main:app --host 0.0.0.0 --port ${PORT:-8000}"]

