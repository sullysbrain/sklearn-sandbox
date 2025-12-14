# syntax=docker/dockerfile:1.7-labs
FROM mcr.microsoft.com/devcontainers/python:3.14-trixie

RUN apt-get update && apt-get install -y --no-install-recommends \
    ca-certificates curl \
 && rm -rf /var/lib/apt/lists/*

# Install uv for all users
RUN curl -LsSf https://astral.sh/uv/install.sh | UV_INSTALL_DIR=/usr/local/bin sh

# Make /opt/venv the default Python and ensure it exists at build time
ENV VIRTUAL_ENV=/opt/venv
RUN python -m venv "$VIRTUAL_ENV"

# Ensure every shell starts with the venv on PATH
ENV PATH="$VIRTUAL_ENV/bin:${PATH}"
RUN printf 'export VIRTUAL_ENV=%s\nexport PATH=%s/bin:$PATH\n' "$VIRTUAL_ENV" "$VIRTUAL_ENV" \
    > /etc/profile.d/venv.sh

# OPTIONAL: run uv sync on container start after the repo is mounted
ENTRYPOINT ["/bin/bash","-lc","set -e; \
  if [ -f pyproject.toml ] || [ -f uv.lock ]; then \
    UV_LINK_MODE=${UV_LINK_MODE:-copy} uv sync --active; \
  fi; \
  exec \"$@\""]
CMD ["bash"]
