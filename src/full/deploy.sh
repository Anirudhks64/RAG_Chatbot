#!/usr/bin/env bash
# Uploads the FULL chatbot API to a Hugging Face Space (Docker Spaces need HF PRO since mid-2026).
# The Space then rebuilds and restarts.
#
# Once:  .venv/bin/hf auth login      (paste a WRITE token from huggingface.co/settings/tokens)
# Every time you want the server updated (from the project folder):
#        src/full/deploy.sh <your-username>/<space-name>
#
# Only the files the server needs are uploaded (no .env, no PDFs, no git history),
# because Hugging Face refuses git pushes that contain binary files.
set -euo pipefail
SPACE="${1:?Usage: src/full/deploy.sh <your-username>/<space-name>}"
cd "$(dirname "$0")/../.."   # the project folder

.venv/bin/hf upload "$SPACE" . . --repo-type space \
  --include ".dockerignore" \
  --include "README.md" \
  --include "src/full/requirements.txt" \
  --include "src/full/*.py" \
  --include "src/shared/*.py" \
  --include "data/websites.txt" \
  --include "data/web_cache/*" \
  --include ".chainlit/*" \
  --delete "src/*" \
  --delete "data/*" \
  --commit-message "Deploy $(git rev-parse --short HEAD 2>/dev/null || echo local)"

# A Space needs its Dockerfile at the top, so src/full/Dockerfile is uploaded as "Dockerfile"
.venv/bin/hf upload "$SPACE" src/full/Dockerfile Dockerfile --repo-type space \
  --commit-message "Dockerfile $(git rev-parse --short HEAD 2>/dev/null || echo local)"

echo
echo "Uploaded. The Space is now building (about 10 minutes the first time)."
echo "Then check: https://$(echo "${SPACE/\//-}" | tr "A-Z_." "a-z--").hf.space/health"
