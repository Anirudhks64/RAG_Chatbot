#!/usr/bin/env bash
# Uploads the chatbot API to your Hugging Face Space. The Space then rebuilds and restarts.
#
# Once:  .venv/bin/hf auth login      (paste a WRITE token from huggingface.co/settings/tokens)
# Every time you want the server updated:
#        ./deploy.sh <your-username>/<space-name>
#
# Only the files the server needs are uploaded (no .env, no PDFs, no git history),
# because Hugging Face refuses git pushes that contain binary files.
set -euo pipefail
SPACE="${1:?Usage: ./deploy.sh <your-username>/<space-name>}"
cd "$(dirname "$0")"

.venv/bin/hf upload "$SPACE" . . --repo-type space \
  --include "Dockerfile" \
  --include ".dockerignore" \
  --include "README.md" \
  --include "requirements.txt" \
  --include "src/*.py" \
  --include "data/websites.txt" \
  --include "data/web_cache/*" \
  --include ".chainlit/*" \
  --delete "src/*" \
  --delete "data/*" \
  --commit-message "Deploy $(git rev-parse --short HEAD 2>/dev/null || echo local)"

echo
echo "Uploaded. The Space is now building (about 10 minutes the first time)."
echo "Then check: https://$(echo "${SPACE/\//-}" | tr "A-Z_." "a-z--").hf.space/health"
