#!/usr/bin/env bash
set -euo pipefail

VSCODE_HOME="/home/vscode"
CACHE_DIR="${VSCODE_HOME}/.cache"
ZSHRC="${VSCODE_HOME}/.zshrc"
P10K_DST="${VSCODE_HOME}/.p10k.zsh"
P10K_SRC="${PWD}/.devcontainer/p10k/.p10k.zsh"
OMZ_DIR="${VSCODE_HOME}/.oh-my-zsh"
P10K_THEME_DIR="${OMZ_DIR}/custom/themes/powerlevel10k"
DOTFILES_P10K_URL="${DOTFILES_P10K_URL:-}"

log() { echo "[postCreate] $*"; }

ensure_cache_permissions() {
  log "Ensuring cache dirs exist and are writable..."
  sudo mkdir -p "${CACHE_DIR}/uv" "${CACHE_DIR}/pip"
  sudo chown -R vscode:vscode "${CACHE_DIR}"
  chmod 700 "${CACHE_DIR}" || true
}

ensure_zsh_installed() {
  if ! command -v zsh >/dev/null 2>&1; then
    log "Installing zsh..."
    sudo apt-get update
    sudo apt-get install -y zsh git curl
  fi
}

ensure_oh_my_zsh() {
  if [[ ! -d "${OMZ_DIR}" ]]; then
    log "Installing Oh My Zsh..."
    # Prevent installer from auto-switching shells / launching zsh
    RUNZSH=no CHSH=no KEEP_ZSHRC=yes \
      sh -c "$(curl -fsSL https://raw.githubusercontent.com/ohmyzsh/ohmyzsh/master/tools/install.sh)"
  else
    log "Oh My Zsh already installed."
  fi

  # Ensure .zshrc exists
  if [[ ! -f "${ZSHRC}" ]]; then
    log "Creating ${ZSHRC}"
    touch "${ZSHRC}"
    chown vscode:vscode "${ZSHRC}" || true
  fi
}

ensure_p10k_theme() {
  if [[ ! -d "${P10K_THEME_DIR}" ]]; then
    log "Installing Powerlevel10k theme..."
    git clone --depth=1 https://github.com/romkatv/powerlevel10k.git "${P10K_THEME_DIR}"
  else
    log "Powerlevel10k theme already installed."
  fi

  # Set theme in .zshrc
  if grep -qE '^ZSH_THEME=' "${ZSHRC}"; then
    sed -i 's|^ZSH_THEME=.*|ZSH_THEME="powerlevel10k/powerlevel10k"|' "${ZSHRC}"
  else
    echo 'ZSH_THEME="powerlevel10k/powerlevel10k"' >> "${ZSHRC}"
  fi
}

install_p10k_config() {
  local installed="false"

  # 1) Prefer dotfiles repo single-file download (if URL provided)
  if [[ -n "${DOTFILES_P10K_URL}" ]]; then
    log "Fetching p10k config from dotfiles URL -> ${P10K_DST}"
    if curl -fsSL "${DOTFILES_P10K_URL}" -o "${P10K_DST}"; then
      chown vscode:vscode "${P10K_DST}" || true
      log "Downloaded p10k config from dotfiles URL ${DOTFILES_P10K_URL}"
      installed="true"
    else
      log "Warning: failed to fetch DOTFILES_P10K_URL (${DOTFILES_P10K_URL}); will try repo fallback"
    fi
  fi

  # 2) Fallback: repo-local saved config
  if [[ "${installed}" != "true" && -f "${P10K_SRC}" ]]; then
    log "Copying saved p10k config -> ${P10K_DST}"
    cp -f "${P10K_SRC}" "${P10K_DST}"
    chown vscode:vscode "${P10K_DST}" || true
    installed="true"
  fi

  # 3) No config found
  if [[ "${installed}" != "true" ]]; then
    log "No p10k config found (dotfiles or repo); leaving default (you can run: p10k configure)"
  fi

  # Ensure .zshrc loads it
  if ! grep -qE '^\s*\[\[\s+-f\s+~\/\.p10k\.zsh\s+\]\]\s+&&\s+source\s+~\/\.p10k\.zsh' "${ZSHRC}"; then
    log "Ensuring .zshrc sources ~/.p10k.zsh"
    cat >> "${ZSHRC}" <<'EOF'

# Load Powerlevel10k config
[[ -f ~/.p10k.zsh ]] && source ~/.p10k.zsh
EOF
  fi
}


set_default_shell_zsh() {
  local zsh_path
  zsh_path="$(command -v zsh)"
  log "Setting default shell for vscode -> ${zsh_path}"
  sudo chsh -s "${zsh_path}" vscode || true
}

uv_sync_if_project() {
  if [[ -f "${PWD}/pyproject.toml" || -f "${PWD}/uv.lock" ]]; then
    log "Running: uv sync --active"
    uv sync --active
  else
    log "No pyproject.toml/uv.lock found; skipping uv sync"
  fi
}

main() {
  ensure_cache_permissions
  ensure_zsh_installed
  ensure_oh_my_zsh
  ensure_p10k_theme
  install_p10k_config
  set_default_shell_zsh
  uv_sync_if_project
  log "Done. Open a new terminal (zsh) to see p10k."
}

main "$@"
