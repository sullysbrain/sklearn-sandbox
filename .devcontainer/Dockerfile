FROM mcr.microsoft.com/devcontainers/python:3.14-trixie

RUN apt-get update && apt-get install -y --no-install-recommends \
    ca-certificates curl \
    && rm -rf /var/lib/apt/lists/*

# install uv
RUN curl -LsSf https://astral.sh/uv/install.sh | sh
ENV PATH="/root/.local/bin:${PATH}"

# use /opt/venv as the project venv and make it the default
ENV VIRTUAL_ENV=/opt/venv
ENV PATH="${VIRTUAL_ENV}/bin:${PATH}"

# create an empty venv so PATH is valid immediately
RUN python -m venv "${VIRTUAL_ENV}"

# entrypoint that runs `uv sync` after your repo is mounted
COPY .devcontainer/entrypoint.sh /usr/local/bin/entrypoint.sh
RUN chmod +x /usr/local/bin/entrypoint.sh
ENTRYPOINT ["/usr/local/bin/entrypoint.sh"]

# devcontainers normally override CMD; bash is a good default
CMD ["bash"]