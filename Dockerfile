# --- Stage 1: The "Builder" ---
# Use the full, official Python image to build our dependencies.
# Using a specific version is a best practice.
FROM python:3.12-bullseye AS builder

# Set the working directory
WORKDIR /app

# Create a virtual environment inside the builder stage. This is a best practice.
RUN python -m venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"

# Copy ONLY the requirements file first.
# This leverages Docker's layer caching. If requirements.txt doesn't change,
# Docker won't re-run the time-consuming pip install.
COPY requirements.txt .

# Install dependencies into the virtual environment.
# --no-cache-dir reduces image size.
RUN pip install --no-cache-dir -r requirements.txt
# RUN pip install --no-cache-dir --upgrade -r ./requirements.txt


# --- Stage 2: The "Runner" ---
# Use a slim, lightweight base image. `python:3.12-slim-bullseye` is excellent.
# It doesn't have all the build tools, making it smaller and more secure.
FROM python:3.12-slim-bullseye AS runner

# --- TEMPORARY DEBUGGING STEP ---
# As root, update the package list and install networking tools if needed for debugging.
# We will remove this section before creating the final production image.
# USER root
# RUN apt-get update && apt-get install -y iputils-ping curl dnsutils && rm -rf /var/lib/apt/lists/*
# USER python # Switch back to a non-root user for security
# --- END OF DEBUGGING STEP ---

# Create a non-root user and group
ARG UID=1001
ARG GID=1001
RUN groupadd --gid ${GID} appgroup && useradd --uid ${UID} --gid appgroup --shell /bin/bash --create-home appuser    

WORKDIR /app
COPY --chown=appuser:appgroup . /app

# Copy the virtual environment from the builder stage.
# This is the key step: we only copy the installed packages, not the build tools.
COPY --from=builder /opt/venv /opt/venv

# Set the path to use the venv's Python and packages.
ENV PATH="/opt/venv/bin:$PATH"

# Set the user to the official, non-root user provided by the base image.
# This is a critical security best practice.
USER appuser
# WORKDIR /home/worker
# USER python

# Now, copy our application code into the final image.
COPY backend ./backend
COPY core ./core
COPY config ./config
COPY prompts ./prompts
# Add any other necessary source directories.

# Expose the port the app runs on
EXPOSE 8000

# This the command to run the application for development purposes only.
# CMD ["uvicorn", "backend.main:app", "--host", "0.0.0.0", "--port", "8000"]

# This is the command to run the app in production.  
# The syntax for PORT means "use the environment variable PORT if set, otherwise use 8000"
CMD ["/bin/sh", "-c", "uvicorn backend.main:app --host 0.0.0.0 --port ${PORT:-8000}"]
