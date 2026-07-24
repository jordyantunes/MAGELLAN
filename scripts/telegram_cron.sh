#!/usr/bin/env bash
# Fires a one-shot headless Claude Code agent to analyze the current MAGELLAN
# run and send a Telegram report. Invoked by system cron (see `crontab -l`),
# independent of any interactive Claude Code session.
set -euo pipefail

PROJECT_DIR="/home/jordy/projects/MAGELLAN"
LOG_DIR="$PROJECT_DIR/scripts/.telegram_cron_logs"
mkdir -p "$LOG_DIR"

cd "$PROJECT_DIR"

/home/jordy/.local/bin/claude \
  --print \
  --allowedTools "Bash" \
  --permission-mode bypassPermissions \
  --model sonnet \
  "$(cat scripts/telegram_cron_prompt.txt)" \
  >> "$LOG_DIR/$(date +%Y%m%d).log" 2>&1
