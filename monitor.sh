#!/usr/bin/env bash
# ==============================================================================
# monitor.sh - Live TUI Dashboard for llama-server Tasks & Slots
# ==============================================================================
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
exec python3 "${SCRIPT_DIR}/scripts/monitor.py" "$@"
