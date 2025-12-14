#!/usr/bin/env bash
set -euo pipefail

# ensure venv exists (idempotent)
if [ ! -d "${VIRTUAL_ENV:-/opt/venv}" ]; then
  uv venv "${VIRTUAL_ENV:-/opt/venv}"
fi

# if project metadata exists in the mounted workspace, sync deps
if [ -f "uv.lock" ] || [ -f "pyproject.toml" ]; then
  # avoid hardlink warnings on macOS/volumes
  UV_LINK_MODE=${UV_LINK_MODE:-copy} uv sync
fi

exec "$@"
